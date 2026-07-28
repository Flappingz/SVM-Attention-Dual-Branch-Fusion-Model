from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.data.dataset import PreparedDataset, SubjectRecord
from ocd_v3.io import write_json_atomic

SPLIT_ALGORITHM_VERSION = 3


@dataclass(frozen=True)
class SplitAssignment:
    outer_fold: int
    subject_id: str
    label_name: str
    label_id: int
    role: str
    available_post_count: int
    selected_post_count: int


def _stable_random(seed: int, *values: object) -> float:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    for value in values:
        digest.update(b"\x1f")
        digest.update(str(value).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") / 2**64


def _allocate_counts(total: int, weights: Sequence[int]) -> list[int]:
    if total < 0 or not weights or sum(weights) <= 0:
        raise ValueError("Invalid proportional allocation request")
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def assign_outer_folds(
    subjects: Iterable[SubjectRecord], fold_count: int, seed: int
) -> dict[str, int]:
    materialized = list(subjects)
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    grouped: dict[int, list[SubjectRecord]] = defaultdict(list)
    for subject in materialized:
        grouped[subject.label_id].append(subject)
    if not grouped:
        raise ValueError("No eligible subjects are available")
    for label_id, rows in grouped.items():
        if len(rows) < fold_count:
            raise ValueError(f"Label {label_id} has fewer subjects than outer folds")

    assignments: dict[str, int] = {}
    for label_id, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -row.split_post_count,
                _stable_random(seed, "outer", label_id, row.subject_id),
            ),
        )
        counts = [0] * fold_count
        post_totals = [0] * fold_count
        fold_tiebreak = list(range(fold_count))
        random.Random(seed + label_id).shuffle(fold_tiebreak)
        tie_rank = {fold: rank for rank, fold in enumerate(fold_tiebreak)}
        for row in ordered:
            selected = min(
                range(fold_count),
                key=lambda fold: (counts[fold], post_totals[fold], tie_rank[fold]),
            )
            assignments[row.subject_id] = selected
            counts[selected] += 1
            post_totals[selected] += row.split_post_count
    return assignments


def _select_validation_subjects(
    candidates: list[SubjectRecord], target: int, seed: int, outer_fold: int, label_id: int
) -> set[str]:
    if target <= 0:
        return set()
    ordered = sorted(candidates, key=lambda row: (row.split_post_count, row.subject_id))
    bin_count = min(5, len(ordered))
    bins: list[list[SubjectRecord]] = [[] for _ in range(bin_count)]
    for index, row in enumerate(ordered):
        bin_index = min(index * bin_count // len(ordered), bin_count - 1)
        bins[bin_index].append(row)
    allocations = _allocate_counts(target, [len(bin_rows) for bin_rows in bins])
    selected: set[str] = set()
    for bin_index, (bin_rows, count) in enumerate(zip(bins, allocations, strict=True)):
        ranked = sorted(
            bin_rows,
            key=lambda row: _stable_random(
                seed, "validation", outer_fold, label_id, bin_index, row.subject_id
            ),
        )
        selected.update(row.subject_id for row in ranked[:count])
    return selected


def build_nested_assignments(
    subjects: Iterable[SubjectRecord],
    outer_folds: dict[str, int],
    fold_count: int,
    validation_fraction: float,
    seed: int,
) -> list[SplitAssignment]:
    materialized = list(subjects)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    assignments: list[SplitAssignment] = []
    for outer_fold in range(fold_count):
        validation_ids: set[str] = set()
        grouped_candidates: dict[int, list[SubjectRecord]] = defaultdict(list)
        for subject in materialized:
            if outer_folds[subject.subject_id] != outer_fold:
                grouped_candidates[subject.label_id].append(subject)
        for label_id, candidates in grouped_candidates.items():
            target = round(len(candidates) * validation_fraction)
            target = max(1, min(target, len(candidates) - 1))
            validation_ids.update(
                _select_validation_subjects(candidates, target, seed, outer_fold, label_id)
            )
        for subject in materialized:
            if outer_folds[subject.subject_id] == outer_fold:
                role = "test"
            elif subject.subject_id in validation_ids:
                role = "validation"
            else:
                role = "train"
            assignments.append(
                SplitAssignment(
                    outer_fold=outer_fold,
                    subject_id=subject.subject_id,
                    label_name=subject.label_name,
                    label_id=subject.label_id,
                    role=role,
                    available_post_count=subject.post_count,
                    selected_post_count=subject.split_post_count,
                )
            )
    validate_assignments(assignments, materialized, fold_count)
    return assignments


def validate_assignments(
    assignments: Iterable[SplitAssignment], subjects: Iterable[SubjectRecord], fold_count: int
) -> None:
    assignment_rows = list(assignments)
    subject_rows = list(subjects)
    expected_ids = {subject.subject_id for subject in subject_rows}
    by_fold: dict[int, list[SplitAssignment]] = defaultdict(list)
    for row in assignment_rows:
        if row.role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {row.role}")
        by_fold[row.outer_fold].append(row)
    if set(by_fold) != set(range(fold_count)):
        raise ValueError("Outer fold IDs are incomplete")
    for outer_fold, rows in by_fold.items():
        row_ids = [row.subject_id for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"Duplicate subject assignment in outer fold {outer_fold}")
        if set(row_ids) != expected_ids:
            raise ValueError(f"Outer fold {outer_fold} does not cover every subject")
        roles = Counter(row.role for row in rows)
        if any(roles[role] == 0 for role in ("train", "validation", "test")):
            raise ValueError(f"Outer fold {outer_fold} has an empty role")
    test_counts = Counter(
        row.subject_id for row in assignment_rows if row.role == "test"
    )
    if set(test_counts) != expected_ids or any(count != 1 for count in test_counts.values()):
        raise ValueError("Each subject must occur in exactly one outer test fold")


def split_summary(assignments: Iterable[SplitAssignment]) -> dict[str, Any]:
    rows = list(assignments)
    summary: dict[str, Any] = {}
    for outer_fold in sorted({row.outer_fold for row in rows}):
        fold_rows = [row for row in rows if row.outer_fold == outer_fold]
        role_summary: dict[str, Any] = {}
        for role in ("train", "validation", "test"):
            role_rows = [row for row in fold_rows if row.role == role]
            role_summary[role] = {
                "subjects": len(role_rows),
                "available_posts": sum(row.available_post_count for row in role_rows),
                "selected_posts": sum(row.selected_post_count for row in role_rows),
                "labels": dict(sorted(Counter(row.label_name for row in role_rows).items())),
            }
        summary[str(outer_fold)] = role_summary
    return summary


def create_split_artifact(
    dataset: PreparedDataset,
    fold_count: int,
    validation_fraction: float,
    seed: int,
    output_root: Path,
    minimum_posts_per_subject: int = 1,
) -> tuple[Path, dict[str, Any]]:
    if minimum_posts_per_subject < 1:
        raise ValueError("minimum_posts_per_subject must be positive")
    all_eligible_subjects = list(dataset.subjects())
    subjects = [
        subject
        for subject in all_eligible_subjects
        if subject.post_count >= minimum_posts_per_subject
    ]
    outer = assign_outer_folds(subjects, fold_count, seed)
    assignments = build_nested_assignments(
        subjects, outer, fold_count, validation_fraction, seed
    )
    identity = {
        "dataset_id": dataset.dataset_id,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "outer_folds": fold_count,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "minimum_available_posts_per_subject": minimum_posts_per_subject,
    }
    split_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    split_id = f"splits-{split_digest}"
    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    split_dir = resolved_output_root / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "assignments.json"
    payload = {
        "schema_version": 1,
        "split_id": split_id,
        **identity,
        "assignments": [asdict(row) for row in assignments],
    }
    write_json_atomic(split_path, payload)
    summary = {
        "schema_version": 1,
        "split_id": split_id,
        **identity,
        "cohort": {
            "subjects_before_post_filter": len(all_eligible_subjects),
            "subjects_included": len(subjects),
            "subjects_excluded_by_post_filter": (
                len(all_eligible_subjects) - len(subjects)
            ),
            "included_labels": dict(
                sorted(Counter(subject.label_name for subject in subjects).items())
            ),
            "excluded_labels": dict(
                sorted(
                    Counter(
                        subject.label_name
                        for subject in all_eligible_subjects
                        if subject.post_count < minimum_posts_per_subject
                    ).items()
                )
            ),
        },
        "summary": split_summary(assignments),
    }
    write_json_atomic(split_path.parent / "summary.json", summary)
    return split_path, summary
