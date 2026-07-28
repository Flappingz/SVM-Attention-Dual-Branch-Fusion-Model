"""Validate and summarize matched modality and architecture ablations."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from ocd_v3.evaluation.metrics import mean_sd_ci95
from ocd_v3.io import write_json_atomic

METRICS = ("f1", "roc_auc", "accuracy", "precision", "recall")
MODALITY_CONDITIONS = (
    "text",
    "image",
    "metadata",
    "text+image",
    "text+metadata",
    "text+image+metadata",
)
ARCHITECTURE_CONDITIONS = (
    "full",
    "without_post_cross_attention",
    "without_user_self_attention",
    "mean_pooling_at_both_levels",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    observed = abs(fmean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(
            fmean(
                sign * value
                for sign, value in zip(signs, differences, strict=True)
            )
        )
        extreme += permuted >= observed - 1e-15
        total += 1
    return extreme / total


def _holm_adjust(raw_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def _protocol(summary: Mapping[str, Any]) -> dict[str, Any]:
    parameters = summary.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "dataset_id": summary.get("dataset_id"),
        "keyword_condition": summary.get(
            "keyword_condition", parameters.get("keyword_condition")
        ),
        "classification_threshold": summary.get(
            "classification_threshold", parameters.get("classification_threshold")
        ),
        "split_seeds": summary.get("split_seeds"),
        "split_ids": summary.get("split_ids"),
        "oof_subjects_per_repetition": summary.get("oof_subjects_per_repetition"),
    }


def _validate_summary(summary: Mapping[str, Any], name: str) -> None:
    rows = summary.get("per_repetition")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{name} is not a repeated subject-level OOF summary")
    seeds = [int(row["split_seed"]) for row in rows]
    split_ids = [str(row["split_id"]) for row in rows]
    if seeds != [int(value) for value in summary.get("split_seeds", ())]:
        raise ValueError(f"{name} split seed order is inconsistent")
    if split_ids != [str(value) for value in summary.get("split_ids", ())]:
        raise ValueError(f"{name} split ID order is inconsistent")
    for row in rows:
        metrics = row.get("oof_metrics", {})
        if any(metrics.get(metric) is None for metric in METRICS):
            raise ValueError(f"{name} has an undefined required metric")


def _paired_family(
    summaries: Mapping[str, Mapping[str, Any]], *, reference: str
) -> dict[str, Any]:
    reference_rows = summaries[reference]["per_repetition"]
    reference_seeds = [int(row["split_seed"]) for row in reference_rows]
    reference_splits = [str(row["split_id"]) for row in reference_rows]
    candidates = [name for name in summaries if name != reference]
    comparisons: dict[str, dict[str, Any]] = {name: {} for name in candidates}
    for metric in METRICS:
        raw_p: dict[str, float] = {}
        differences_by_candidate: dict[str, list[float]] = {}
        for name in candidates:
            rows = summaries[name]["per_repetition"]
            if [int(row["split_seed"]) for row in rows] != reference_seeds:
                raise ValueError(f"{name} does not use the reference seed order")
            if [str(row["split_id"]) for row in rows] != reference_splits:
                raise ValueError(f"{name} does not use the reference split IDs")
            differences = [
                float(candidate["oof_metrics"][metric])
                - float(full["oof_metrics"][metric])
                for candidate, full in zip(rows, reference_rows, strict=True)
            ]
            differences_by_candidate[name] = differences
            raw_p[name] = _exact_sign_flip_pvalue(differences)
        adjusted = _holm_adjust(raw_p)
        for name in candidates:
            differences = differences_by_candidate[name]
            sample_sd = stdev(differences)
            comparisons[name][metric] = {
                "direction": f"{name} minus {reference}",
                "differences_by_split_seed": dict(
                    zip((str(seed) for seed in reference_seeds), differences, strict=True)
                ),
                "summary": asdict(mean_sd_ci95(differences)),
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
                "paired_standardized_mean_difference": (
                    fmean(differences) / sample_sd if sample_sd > 0 else None
                ),
                "exact_two_sided_sign_flip_p": raw_p[name],
                "holm_adjusted_p_within_metric": adjusted[name],
            }
    return {
        "reference": reference,
        "comparison_unit": "matched split-seed repetition-level subject OOF metric",
        "test": "exact two-sided sign-flip randomization",
        "multiplicity": "Holm adjustment across planned contrasts within each metric",
        "by_condition": comparisons,
    }


def _condition_table(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            metric: summary["metric_summary"][metric]
            for metric in METRICS
        }
        for name, summary in summaries.items()
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Ablation report",
        "",
        "All rows use the same 177-user Keyword-post-removed cohort, matched outer ",
        "partitions, fixed threshold 0.5, and one complete subject-level OOF result per ",
        "split-seed repetition. Intervals describe partition sensitivity on this cohort; ",
        "they are not population or external-validation intervals.",
        "",
    ]
    for family, title in (
        ("modality", "Modality ablations"),
        ("architecture", "Architecture ablations"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Condition | F1 mean | F1 95% CI | ROC-AUC mean | ROC-AUC 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for condition, metrics in report[family]["metrics"].items():
            f1 = metrics["f1"]
            auc = metrics["roc_auc"]
            lines.append(
                f"| {condition} | {f1['mean']:.6f} | "
                f"[{f1['ci95_low']:.6f}, {f1['ci95_high']:.6f}] | "
                f"{auc['mean']:.6f} | "
                f"[{auc['ci95_low']:.6f}, {auc['ci95_high']:.6f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "## Verification",
            "",
            "- Status: ANALYZED (upgrade to VERIFIED only after "
            "independent artifact audit)",
            "- Statistical unit: one complete subject-level OOF result per split seed",
            "- Interpretation boundary: repeated partitions share the same users",
            "",
        ]
    )
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def build_ablation_report(
    *,
    modality_summary_paths: Mapping[str, Path],
    architecture_summary_paths: Mapping[str, Path],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if set(modality_summary_paths) != set(MODALITY_CONDITIONS):
        raise ValueError("Modality report requires the six pre-specified conditions")
    if set(architecture_summary_paths) != set(ARCHITECTURE_CONDITIONS):
        raise ValueError("Architecture report requires the four pre-specified conditions")
    modality = {
        name: _read_json(path.resolve()) for name, path in modality_summary_paths.items()
    }
    architecture = {
        name: _read_json(path.resolve())
        for name, path in architecture_summary_paths.items()
    }
    for name, summary in {**modality, **architecture}.items():
        _validate_summary(summary, name)
    reference_protocol = _protocol(modality["text+image+metadata"])
    if any(_protocol(summary) != reference_protocol for summary in modality.values()):
        raise ValueError("Modality summaries do not share one locked evaluation protocol")
    if any(_protocol(summary) != reference_protocol for summary in architecture.values()):
        raise ValueError("Architecture summaries do not share the modality protocol")
    if (
        modality["text+image+metadata"]["run_ids"]
        != architecture["full"]["run_ids"]
    ):
        raise ValueError("The modality-full and architecture-full references differ")

    report = {
        "schema_version": 1,
        "verification": {
            "verification_status": "ANALYZED",
            "statistical_unit": "one complete subject-level OOF result per split seed",
        },
        "protocol": reference_protocol,
        "modality": {
            "metrics": _condition_table(modality),
            "paired_comparisons": _paired_family(
                modality, reference="text+image+metadata"
            ),
            "summary_paths": {
                name: str(path.resolve()) for name, path in modality_summary_paths.items()
            },
        },
        "architecture": {
            "metrics": _condition_table(architecture),
            "paired_comparisons": _paired_family(architecture, reference="full"),
            "summary_paths": {
                name: str(path.resolve())
                for name, path in architecture_summary_paths.items()
            },
        },
        "interpretation_boundary": (
            "All repetitions share the same subjects. Results quantify matched partition "
            "sensitivity and do not establish external, temporal, platform, or clinical "
            "generalization."
        ),
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    write_json_atomic(output_path, report)
    _write_markdown(output_dir / "report.md", report)
    return output_path, report
