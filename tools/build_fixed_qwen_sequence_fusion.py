"""Build a pre-specified 90/10 pure-Qwen sequence fusion artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocd_v3.experiments.qwen_fixed_fusion import build_fixed_qwen_sequence_fusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--secondary-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--selection-provenance", required=True)
    args = parser.parse_args()
    output_dir, summary = build_fixed_qwen_sequence_fusion(
        artifact_root=args.artifact_root,
        primary_summary_path=args.primary_summary,
        secondary_summary_path=args.secondary_summary,
        output_root=args.output_root,
        repository=args.repository,
        secondary_weight=0.1,
        selection_provenance=args.selection_provenance,
        threshold=0.5,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "fusion_id": summary["fusion_id"],
                "metric_summary": summary["metric_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
