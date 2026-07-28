from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ocd_v3.config import StudyConfig
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof
from ocd_v3.experiments.full_config import FullExperimentConfig
from ocd_v3.features.training_data import (
    PreparedFeatureSet,
    UserFeatureDataset,
    collate_user_features,
    fit_metadata_standardizer,
)
from ocd_v3.io import write_json_atomic, write_jsonl_atomic
from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier
from ocd_v3.provenance import create_run_manifest, save_run_manifest


@dataclass(frozen=True)
class TrainFullResult:
    run_dir: Path
    summary: dict[str, Any]


class HierarchicalExperimentConfiguration(Protocol):
    encoder: Any
    model: Any
    training: Any

    def public_dict(self) -> dict[str, Any]: ...


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(name: str) -> Any:
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for training but is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    collate_fn: Callable[[Sequence[Any]], Any] = collate_user_features,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    device: Any,
    loss_function: Any,
    optimizer: Any | None,
    gradient_clip_norm: float,
    batch_forward: Callable[[Any, dict[str, Any], Any], dict[str, Any]] | None = None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    import torch

    training = optimizer is not None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    labels: list[int] = []
    scores: list[float] = []
    subject_ids: list[str] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            batch_labels = batch["labels"].to(device, non_blocking=True)
            outputs = (
                model(
                    batch["content_embeddings"].to(device, non_blocking=True),
                    batch["metadata"].to(device, non_blocking=True),
                    batch["post_mask"].to(device, non_blocking=True),
                )
                if batch_forward is None
                else batch_forward(model, batch, device)
            )
            loss = loss_function(outputs["logits"], batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite training/evaluation loss")
            if training:
                group_start = (
                    batch_index // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_size = min(
                    gradient_accumulation_steps,
                    total_batches - group_start,
                )
                (loss / group_size).backward()
                completes_group = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if completes_group:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            sample_count = int(batch_labels.shape[0])
            total_loss += float(loss.item()) * sample_count
            total_samples += sample_count
            labels.extend(int(value) for value in batch_labels.detach().cpu().tolist())
            scores.extend(
                float(value)
                for value in outputs["probabilities"].detach().cpu().tolist()
            )
            subject_ids.extend(batch["subject_ids"])
    if total_samples == 0:
        raise ValueError("Data loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "labels": labels,
        "scores": scores,
        "subject_ids": subject_ids,
    }


def _metrics_payload(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "loss": result["loss"],
        **binary_metrics(result["labels"], result["scores"], threshold).to_dict(),
    }


def _prediction_rows(
    result: dict[str, Any], *, outer_fold: int, role: str, threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            "subject_id": subject_id,
            "outer_fold": outer_fold,
            "role": role,
            "label": label,
            "score": score,
            "prediction": int(score >= threshold),
            "threshold": threshold,
        }
        for subject_id, label, score in zip(
            result["subject_ids"], result["labels"], result["scores"], strict=True
        )
    ]


def _load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assignments"), list):
        raise ValueError("Unsupported split assignment artifact")
    return payload


def _fold_summary(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        values = [float(row[metric]) for row in fold_metrics if row[metric] is not None]
        if len(values) != len(fold_metrics):
            raise ValueError(f"Fold metric {metric} is undefined in at least one test fold")
        if len(values) == 1:
            summary[metric] = {
                "n": 1,
                "mean": values[0],
                "sample_sd": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        else:
            summary[metric] = asdict(mean_sd_ci95(values))
    return summary


def train_hierarchical_full(
    *,
    feature_set: PreparedFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    full_config: FullExperimentConfig | HierarchicalExperimentConfiguration,
    output_root: Path,
    selected_folds: Iterable[int] | None = None,
    epoch_limit: int | None = None,
    experiment_id: str = "hierarchical_attention_full",
    configuration_parameter_name: str = "full_experiment",
    additional_parameters: dict[str, Any] | None = None,
    summary_fields: dict[str, Any] | None = None,
    model_factory: Callable[[], Any] | None = None,
    include_metadata: bool = True,
    dataset_factory: Callable[
        [dict[str, list[dict[str, Any]]]], dict[str, Any]
    ]
    | None = None,
    collate_fn: Callable[[Sequence[Any]], Any] = collate_user_features,
    batch_forward: Callable[[Any, dict[str, Any], Any], dict[str, Any]] | None = None,
    optimizer_factory: Callable[[Any], Any] | None = None,
    gradient_accumulation_steps: int = 1,
) -> TrainFullResult:
    import torch
    from safetensors.torch import save_file

    split = _load_split(split_assignments_path.resolve())
    if split["dataset_id"] != feature_set.dataset_id:
        raise ValueError("Feature set and split artifact refer to different datasets")
    if tuple(full_config.encoder.representations) != feature_set.schema.representation_names:
        raise ValueError("Full config representations differ from the feature bundle schema")
    all_fold_ids = sorted({int(row["outer_fold"]) for row in split["assignments"]})
    fold_ids = all_fold_ids if selected_folds is None else sorted(set(selected_folds))
    if not fold_ids or any(fold not in all_fold_ids for fold in fold_ids):
        raise ValueError("selected_folds contains an unknown or empty fold selection")
    epochs = full_config.training.epochs if epoch_limit is None else int(epoch_limit)
    if epochs < 1:
        raise ValueError("epoch_limit must be positive")

    parameters = {
        configuration_parameter_name: full_config.public_dict(),
        "feature_id": feature_set.feature_id,
        "embedding_input_modalities": feature_set.manifest.get(
            "embedding_input_modalities", ["text", "image"]
        ),
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "selected_folds": fold_ids,
        "epoch_limit": epochs,
        "threshold_protocol": "fixed_from_study_config",
        "classification_threshold": study_config.evaluation.classification_threshold,
        "cohort_eligibility": {
            "minimum_available_posts_per_subject": int(
                split.get("minimum_available_posts_per_subject", 1)
            ),
            "subjects": len(
                {
                    str(row["subject_id"])
                    for row in split["assignments"]
                }
            ),
        },
        "split_protocol": {
            key: split.get(key)
            for key in (
                "split_protocol",
                "strategy",
                "train_ratio",
                "validation_ratio",
                "test_ratio",
                "post_count_bins",
            )
            if key in split
        },
    }
    if additional_parameters:
        overlapping = set(parameters).intersection(additional_parameters)
        if overlapping:
            raise ValueError(
                "Additional parameters overlap existing keys: "
                + ", ".join(sorted(overlapping))
            )
        parameters.update(additional_parameters)
    parameters.setdefault(
        "metadata_condition", "included" if include_metadata else "excluded"
    )
    repository = Path(__file__).resolve().parents[3]
    run_manifest = create_run_manifest(
        repository=repository,
        dataset_id=feature_set.dataset_id,
        split_id=str(split["split_id"]),
        experiment_id=experiment_id,
        parameters=parameters,
        seeds={
            "split": int(split["seed"]),
            "training_base": full_config.training.seed,
        },
    )
    run_dir = output_root.resolve() / str(run_manifest["run_id"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return TrainFullResult(
            run_dir=run_dir,
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            "An incomplete run directory already exists; inspect it before retrying"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(run_dir / "run_manifest.json", run_manifest)

    threshold = study_config.evaluation.classification_threshold
    device = _device(full_config.encoder.device)
    fold_summaries: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    trainable_parameter_count: int | None = None
    trainable_parameter_counts: dict[str, int] | None = None
    trainable_parameter_breakdown: dict[str, int] | None = None

    for outer_fold in fold_ids:
        fold_dir = run_dir / f"fold-{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            row for row in split["assignments"] if int(row["outer_fold"]) == outer_fold
        ]
        by_role = {
            role: [row for row in rows if row["role"] == role]
            for role in ("train", "validation", "test")
        }
        if any(not by_role[role] for role in by_role):
            raise ValueError(f"Fold {outer_fold} has an empty train/validation/test role")
        training_ids = [str(row["subject_id"]) for row in by_role["train"]]
        standardizer = (
            fit_metadata_standardizer(feature_set, training_ids)
            if include_metadata
            else None
        )
        if standardizer is not None:
            write_json_atomic(
                fold_dir / "metadata_standardizer.json",
                {"mean": list(standardizer.mean), "scale": list(standardizer.scale)},
            )
        fold_seed = full_config.training.seed + outer_fold
        _seed_everything(fold_seed)
        datasets = (
            {
                role: UserFeatureDataset(
                    feature_set=feature_set,
                    assignments=role_rows,
                    standardizer=standardizer,
                )
                for role, role_rows in by_role.items()
            }
            if dataset_factory is None
            else dataset_factory(by_role)
        )
        if set(datasets) != set(by_role):
            raise ValueError("dataset_factory must return train/validation/test datasets")
        loaders = {
            role: _loader(
                dataset,
                batch_size=full_config.training.batch_size,
                shuffle=role == "train",
                num_workers=full_config.training.num_workers,
                seed=fold_seed,
                collate_fn=collate_fn,
            )
            for role, dataset in datasets.items()
        }
        model = (
            HierarchicalAttentionClassifier(
                config=full_config.model,
                representation_names=feature_set.schema.representation_names,
                embedding_dimension=feature_set.schema.embedding_dimension,
                metadata_dimension=(
                    feature_set.schema.metadata_dimension if include_metadata else 0
                ),
            )
            if model_factory is None
            else model_factory()
        ).to(device)
        count = model.trainable_parameter_count()
        if trainable_parameter_count is None:
            trainable_parameter_count = count
        elif trainable_parameter_count != count:
            raise RuntimeError("Trainable parameter count changed between folds")
        component_counts = (
            model.trainable_parameter_counts()
            if hasattr(model, "trainable_parameter_counts")
            else {"total": count}
        )
        if trainable_parameter_counts is None:
            trainable_parameter_counts = component_counts
        elif trainable_parameter_counts != component_counts:
            raise RuntimeError("Trainable component parameter counts changed between folds")
        architectural_breakdown = (
            model.trainable_parameter_breakdown()
            if hasattr(model, "trainable_parameter_breakdown")
            else {"total": count}
        )
        if trainable_parameter_breakdown is None:
            trainable_parameter_breakdown = architectural_breakdown
        elif trainable_parameter_breakdown != architectural_breakdown:
            raise RuntimeError(
                "Trainable architectural parameter breakdown changed between folds"
            )
        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=full_config.training.learning_rate,
                weight_decay=full_config.training.weight_decay,
            )
            if optimizer_factory is None
            else optimizer_factory(model)
        )
        loss_function = torch.nn.BCEWithLogitsLoss()
        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_result = _run_epoch(
                model=model,
                loader=loaders["train"],
                device=device,
                loss_function=loss_function,
                optimizer=optimizer,
                gradient_clip_norm=full_config.training.gradient_clip_norm,
                batch_forward=batch_forward,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            validation_result = _run_epoch(
                model=model,
                loader=loaders["validation"],
                device=device,
                loss_function=loss_function,
                optimizer=None,
                gradient_clip_norm=full_config.training.gradient_clip_norm,
                batch_forward=batch_forward,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": _metrics_payload(train_result, threshold),
                    "validation": _metrics_payload(validation_result, threshold),
                }
            )
            if validation_result["loss"] < best_loss:
                best_loss = float(validation_result["loss"])
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= full_config.training.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("Training ended without a finite validation checkpoint")
        model.load_state_dict(best_state)
        save_file(
            {name: tensor.contiguous() for name, tensor in best_state.items()},
            fold_dir / "best_model.safetensors",
        )
        write_json_atomic(fold_dir / "history.json", history)
        write_json_atomic(
            fold_dir / "checkpoint.json",
            {
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
                "selection_metric": "validation_bce_loss",
                "fold_seed": fold_seed,
                "threshold": threshold,
            },
        )

        final_results = {
            role: _run_epoch(
                model=model,
                loader=loader,
                device=device,
                loss_function=loss_function,
                optimizer=None,
                gradient_clip_norm=full_config.training.gradient_clip_norm,
                batch_forward=batch_forward,
            )
            for role, loader in loaders.items()
        }
        prediction_rows = [
            prediction
            for role, result in final_results.items()
            for prediction in _prediction_rows(
                result, outer_fold=outer_fold, role=role, threshold=threshold
            )
        ]
        write_jsonl_atomic(fold_dir / "predictions.jsonl", prediction_rows)
        test_rows = [row for row in prediction_rows if row["role"] == "test"]
        all_oof_rows.extend(test_rows)
        metrics = {
            role: _metrics_payload(result, threshold)
            for role, result in final_results.items()
        }
        fold_summary = {
            "outer_fold": outer_fold,
            "best_epoch": best_epoch,
            "fold_seed": fold_seed,
            "role_counts": {role: len(role_rows) for role, role_rows in by_role.items()},
            "metrics": metrics,
        }
        write_json_atomic(fold_dir / "metrics.json", fold_summary)
        fold_summaries.append(fold_summary)

        del model, optimizer, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl_atomic(run_dir / "oof_predictions.jsonl", all_oof_rows)
    complete_outer_cv = fold_ids == all_fold_ids
    is_holdout = split.get("split_protocol") == "single_group_stratified_holdout"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "dataset_id": feature_set.dataset_id,
        "feature_id": feature_set.feature_id,
        "split_id": split["split_id"],
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "embedding_input_modalities": feature_set.manifest.get(
            "embedding_input_modalities", ["text", "image"]
        ),
        "metadata_condition": "included" if include_metadata else "excluded",
        "architecture_variant": getattr(full_config.model, "architecture_variant", "full"),
        "complete_outer_cv": complete_outer_cv and not is_holdout,
        "holdout_evaluation": complete_outer_cv and is_holdout,
        "selected_folds": fold_ids,
        "threshold": threshold,
        "threshold_protocol": "fixed_before outer-CV evaluation",
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_counts": trainable_parameter_counts,
        "trainable_parameter_breakdown": trainable_parameter_breakdown,
        "folds": fold_summaries,
    }
    if summary_fields:
        overlapping = set(summary).intersection(summary_fields)
        conflicting = {
            key
            for key in overlapping
            if summary[key] != summary_fields[key]
        }
        if conflicting:
            raise ValueError(
                "Additional summary fields overlap existing keys: "
                + ", ".join(sorted(conflicting))
            )
        summary.update(
            {key: value for key, value in summary_fields.items() if key not in overlapping}
        )
    if complete_outer_cv:
        expected_subject_ids = {
            str(row["subject_id"])
            for row in split["assignments"]
            if row["role"] == "test"
        }
        oof_predictions = [
            OOFPrediction(
                subject_id=str(row["subject_id"]),
                outer_fold=int(row["outer_fold"]),
                label=int(row["label"]),
                score=float(row["score"]),
            )
            for row in all_oof_rows
        ]
        summary["fold_test_metrics"] = _fold_summary(
            [row["metrics"]["test"] for row in fold_summaries]
        )
        metrics = evaluate_oof(
            oof_predictions, expected_subject_ids, threshold
        ).to_dict()
        summary["holdout_metrics" if is_holdout else "oof_metrics"] = metrics
    write_json_atomic(summary_path, summary)
    return TrainFullResult(run_dir=run_dir, summary=summary)
