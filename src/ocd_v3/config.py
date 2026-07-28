from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a study configuration is incomplete or inconsistent."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PSEUDONYMIZATION_KEY_ENV = "OCD_PSEUDONYMIZATION_KEY"


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                missing.add(name)
                return match.group(0)
            return os.environ[name]

        expanded = _ENV_PATTERN.sub(replace, value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ConfigurationError(f"Missing required environment variable(s): {names}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {name} key(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class GroupConfig:
    directory: str
    label_name: str
    label_id: int


@dataclass(frozen=True)
class PostSelectionConfig:
    maximum_posts_per_subject: int
    strategy: str


@dataclass(frozen=True)
class DatasetConfig:
    groups: tuple[GroupConfig, ...]
    keywords: tuple[str, ...]
    keyword_mask: str
    post_selection: PostSelectionConfig


@dataclass(frozen=True)
class EvaluationConfig:
    outer_folds: int
    split_seed: int
    validation_fraction_within_outer_train: float
    classification_threshold: float


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    raw_root: Path
    artifact_root: Path
    private_mapping_root: Path
    dataset: DatasetConfig
    evaluation: EvaluationConfig

    @property
    def audit_dir(self) -> Path:
        return self.artifact_root / "audits"

    @property
    def dataset_dir(self) -> Path:
        return self.artifact_root / "datasets"

    @property
    def split_dir(self) -> Path:
        return self.artifact_root / "splits"

    @property
    def feature_dir(self) -> Path:
        return self.artifact_root / "features"

    @property
    def run_dir(self) -> Path:
        return self.artifact_root / "runs"

    @property
    def private_mapping_dir(self) -> Path:
        """Private, non-versioned identity mappings for this study."""
        return self.private_mapping_root / "identity-mappings"

    def pseudonymization_key(self) -> bytes:
        """Return the local secret used for stable, keyed pseudonyms.

        The secret is deliberately not part of the JSON configuration, manifests, or
        repository. It must be supplied by the governed execution environment.
        """
        value = os.environ.get(PSEUDONYMIZATION_KEY_ENV)
        if not value:
            raise ConfigurationError(
                f"Missing required environment variable: {PSEUDONYMIZATION_KEY_ENV}"
            )
        return value.encode("utf-8")

    def resolved_dict(self) -> dict[str, Any]:
        """Return the complete local configuration, including resolved paths."""
        payload = asdict(self)
        payload["raw_root"] = str(self.raw_root)
        payload["artifact_root"] = str(self.artifact_root)
        payload["private_mapping_root"] = str(self.private_mapping_root)
        return payload

    def public_dict(self) -> dict[str, Any]:
        """Return the path-independent study protocol used in public identities."""
        payload = asdict(self)
        payload.pop("raw_root", None)
        payload.pop("artifact_root", None)
        payload.pop("private_mapping_root", None)
        return payload

    def digest(self) -> str:
        encoded = json.dumps(
            self.public_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> StudyConfig:
    config_path = Path(path)
    raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload = _require_mapping(_expand_environment(raw_payload), "configuration")
    _reject_unknown(
        payload,
        {"schema_version", "paths", "dataset", "evaluation"},
        "top-level",
    )
    if payload.get("schema_version") != 2:
        raise ConfigurationError("Only study configuration schema_version=2 is supported")

    paths = _require_mapping(payload.get("paths"), "paths")
    _reject_unknown(paths, {"raw_root", "artifact_root", "private_mapping_root"}, "paths")
    raw_root = Path(paths["raw_root"]).expanduser().resolve()
    artifact_root = Path(paths["artifact_root"]).expanduser().resolve()
    private_mapping_root = Path(paths["private_mapping_root"]).expanduser().resolve()
    if raw_root == artifact_root or raw_root in artifact_root.parents:
        raise ConfigurationError("artifact_root must not be the raw data directory or its child")
    if private_mapping_root == raw_root or raw_root in private_mapping_root.parents:
        raise ConfigurationError(
            "private_mapping_root must not be the raw data directory or its child"
        )
    if (
        private_mapping_root == artifact_root
        or private_mapping_root in artifact_root.parents
        or artifact_root in private_mapping_root.parents
    ):
        raise ConfigurationError(
            "private_mapping_root must be separate from artifact_root"
        )

    # Example configurations live in the source tree. Refuse any data or identity
    # mapping root below that worktree so an accidental local path cannot be staged.
    for candidate in (config_path.resolve().parent, *config_path.resolve().parents):
        if not (candidate / ".git").exists():
            continue
        if artifact_root == candidate or candidate in artifact_root.parents:
            raise ConfigurationError("artifact_root must be outside the Git worktree")
        if private_mapping_root == candidate or candidate in private_mapping_root.parents:
            raise ConfigurationError("private_mapping_root must be outside the Git worktree")
        break

    dataset_payload = _require_mapping(payload.get("dataset"), "dataset")
    _reject_unknown(
        dataset_payload,
        {"groups", "keywords", "keyword_mask", "post_selection"},
        "dataset",
    )
    group_payloads = dataset_payload.get("groups")
    if not isinstance(group_payloads, list) or len(group_payloads) != 2:
        raise ConfigurationError("dataset.groups must contain exactly the two study groups")
    groups: list[GroupConfig] = []
    for index, group_value in enumerate(group_payloads):
        group = _require_mapping(group_value, f"dataset.groups[{index}]")
        _reject_unknown(group, {"directory", "label_name", "label_id"}, "group")
        groups.append(
            GroupConfig(
                directory=str(group["directory"]),
                label_name=str(group["label_name"]),
                label_id=int(group["label_id"]),
            )
        )
    if len({group.directory for group in groups}) != len(groups):
        raise ConfigurationError("Group directories must be unique")
    if len({group.label_id for group in groups}) != len(groups):
        raise ConfigurationError("Group label IDs must be unique")

    keywords = dataset_payload.get("keywords")
    if not isinstance(keywords, list) or not all(str(item).strip() for item in keywords):
        raise ConfigurationError("dataset.keywords must be a non-empty string list")
    selection_payload = _require_mapping(
        dataset_payload.get("post_selection"), "dataset.post_selection"
    )
    _reject_unknown(
        selection_payload,
        {"maximum_posts_per_subject", "strategy"},
        "post_selection",
    )
    selection = PostSelectionConfig(
        maximum_posts_per_subject=int(selection_payload["maximum_posts_per_subject"]),
        strategy=str(selection_payload["strategy"]),
    )
    if selection.maximum_posts_per_subject < 1:
        raise ConfigurationError("maximum_posts_per_subject must be positive")
    if selection.strategy not in {"most_recent"}:
        raise ConfigurationError("post_selection.strategy currently supports only most_recent")
    dataset = DatasetConfig(
        groups=tuple(groups),
        keywords=tuple(dict.fromkeys(str(item).strip() for item in keywords)),
        keyword_mask=str(dataset_payload.get("keyword_mask", "[MASK]")),
        post_selection=selection,
    )

    evaluation_payload = _require_mapping(payload.get("evaluation"), "evaluation")
    _reject_unknown(
        evaluation_payload,
        {
            "outer_folds",
            "split_seed",
            "validation_fraction_within_outer_train",
            "classification_threshold",
        },
        "evaluation",
    )
    evaluation = EvaluationConfig(
        outer_folds=int(evaluation_payload["outer_folds"]),
        split_seed=int(evaluation_payload["split_seed"]),
        validation_fraction_within_outer_train=float(
            evaluation_payload["validation_fraction_within_outer_train"]
        ),
        classification_threshold=float(evaluation_payload["classification_threshold"]),
    )
    if evaluation.outer_folds < 2:
        raise ConfigurationError("outer_folds must be at least 2")
    if not 0 < evaluation.validation_fraction_within_outer_train < 1:
        raise ConfigurationError("validation fraction must be between 0 and 1")
    if not 0 <= evaluation.classification_threshold <= 1:
        raise ConfigurationError("classification threshold must be in [0, 1]")

    return StudyConfig(
        schema_version=2,
        raw_root=raw_root,
        artifact_root=artifact_root,
        private_mapping_root=private_mapping_root,
        dataset=dataset,
        evaluation=evaluation,
    )
