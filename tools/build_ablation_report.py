"""Build a matched ablation table from repeated-CV summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocd_v3.experiments.ablation_report import build_ablation_report


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Summary arguments must be unique NAME=PATH values")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", action="append", required=True)
    parser.add_argument("--architecture", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_path, report = build_ablation_report(
        modality_summary_paths=_named_paths(args.modality),
        architecture_summary_paths=_named_paths(args.architecture),
        output_dir=Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "protocol": report["protocol"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
