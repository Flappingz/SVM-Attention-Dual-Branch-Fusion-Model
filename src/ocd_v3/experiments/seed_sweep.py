from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.evaluation.metrics import mean_sd_ci95
from ocd_v3.experiments.full_config import FullExperimentConfig
from ocd_v3.experiments.train_full import TrainFullResult, train_hierarchical_full
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import read_jsonl, write_json_atomic

_METRIC_NAMES = ("f1", "roc_auc", "accuracy", "precision", "recall")


@dataclass(frozen=True)
class SeedSweepResult:
    sweep_dir: Path
    summary: dict[str, Any]
    runs: tuple[TrainFullResult, ...]


def parse_seed_spec(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("Seed range end precedes start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seed specification must contain unique integers")
    return tuple(seeds)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _seed_invariant_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    invariant = deepcopy(manifest)
    invariant.pop("created_at", None)
    invariant.pop("run_id", None)
    invariant.get("seeds", {}).pop("training_base", None)
    training = (
        invariant.get("parameters", {})
        .get("full_experiment", {})
        .get("training", {})
    )
    training.pop("seed", None)
    return invariant


def summarize_full_seed_runs(
    runs: Sequence[TrainFullResult],
    *,
    expected_base_seeds: Sequence[int],
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    seeds = tuple(int(value) for value in expected_base_seeds)
    if len(runs) != len(seeds) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Seed sweep requires one unique completed run per seed")

    per_seed: list[dict[str, Any]] = []
    reference_manifest: dict[str, Any] | None = None
    reference_subjects: dict[str, tuple[int, int]] | None = None
    run_ids: set[str] = set()
    for expected_seed, result in zip(seeds, runs, strict=True):
        summary = _read_json(result.run_dir / "summary.json")
        manifest = _read_json(result.run_dir / "run_manifest.json")
        if not summary.get("complete_outer_cv"):
            raise ValueError("Seed aggregation requires complete outer CV runs")
        run_id = str(summary["run_id"])
        if run_id != manifest.get("run_id") or run_id in run_ids:
            raise ValueError("Seed run IDs are inconsistent or duplicated")
        run_ids.add(run_id)
        if int(manifest["seeds"]["training_base"]) != expected_seed:
            raise ValueError("Run training seed differs from the requested seed order")
        if int(manifest["parameters"]["full_experiment"]["training"]["seed"]) != expected_seed:
            raise ValueError("Full configuration seed differs from the run seed")
        invariant = _seed_invariant_manifest(manifest)
        if reference_manifest is None:
            reference_manifest = invariant
        elif invariant != reference_manifest:
            raise ValueError("Seed runs differ in fields other than training seed")

        predictions = read_jsonl(result.run_dir / "oof_predictions.jsonl")
        subjects = {
            str(row["subject_id"]): (int(row["label"]), int(row["outer_fold"]))
            for row in predictions
        }
        if len(predictions) != len(subjects):
            raise ValueError("A seed run has duplicate OOF subjects")
        if reference_subjects is None:
            reference_subjects = subjects
        elif subjects != reference_subjects:
            raise ValueError("Seed runs do not cover identical OOF subjects and folds")

        metrics = summary["oof_metrics"]
        if any(metrics.get(name) is None for name in _METRIC_NAMES):
            raise ValueError("A required OOF metric is undefined")
        per_seed.append(
            {
                "base_seed": expected_seed,
                "fold_seeds": [expected_seed + fold for fold in range(5)],
                "run_id": run_id,
                "oof_metrics": metrics,
            }
        )

    assert reference_manifest is not None and reference_subjects is not None
    metric_summary = {
        name: asdict(
            mean_sd_ci95(
                [float(row["oof_metrics"][name]) for row in per_seed]
            )
        )
        for name in _METRIC_NAMES
    }
    identity = {
        "dataset_id": reference_manifest["dataset_id"],
        "feature_id": reference_manifest["parameters"]["feature_id"],
        "split_id": reference_manifest["split_id"],
        "base_seeds": list(seeds),
        "run_ids": [row["run_id"] for row in per_seed],
        "source_code": reference_manifest["source_code"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    sweep_id = f"seed-sweep-{digest}"
    sweep_root = output_root.resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_dir = sweep_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "seed_sweep_id": sweep_id,
        **identity,
        "keyword_condition": reference_manifest["parameters"]["keyword_condition"],
        "classification_threshold": reference_manifest["parameters"][
            "classification_threshold"
        ],
        "n_seed_repetitions": len(seeds),
        "outer_folds_per_seed": 5,
        "trained_fold_models": len(seeds) * 5,
        "oof_subjects_per_seed": len(reference_subjects),
        "statistical_unit": "training-seed repetition over the same fixed five folds and subjects",
        "ci_method": "two-sided 95% Student's t interval over seed-level OOF metrics",
        "interpretation_boundary": (
            "Intervals quantify optimization-seed variability conditional on this fixed "
            "dataset and split; seeds are not independent user samples."
        ),
        "metric_summary": metric_summary,
        "per_seed": per_seed,
    }
    write_json_atomic(sweep_dir / "summary.json", summary)
    return sweep_dir, summary


def run_full_seed_sweep(
    *,
    feature_set: PreparedFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    full_config: FullExperimentConfig,
    run_output_root: Path,
    sweep_output_root: Path,
    base_seeds: Sequence[int],
    on_run_complete: Callable[[int, TrainFullResult], None] | None = None,
) -> SeedSweepResult:
    seeds = tuple(int(value) for value in base_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique training seeds are required")
    runs: list[TrainFullResult] = []
    for seed in seeds:
        seeded_config = replace(
            full_config,
            training=replace(full_config.training, seed=seed),
        )
        result = train_hierarchical_full(
            feature_set=feature_set,
            split_assignments_path=split_assignments_path,
            study_config=study_config,
            full_config=seeded_config,
            output_root=run_output_root,
        )
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(seed, result)
    sweep_dir, summary = summarize_full_seed_runs(
        runs,
        expected_base_seeds=seeds,
        output_root=sweep_output_root,
    )
    return SeedSweepResult(sweep_dir=sweep_dir, summary=summary, runs=tuple(runs))
