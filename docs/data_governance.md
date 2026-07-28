# Data governance

The study uses governed Chinese social-media records and derived user-level
research artifacts. Raw records and participant-linked outputs must remain in
controlled storage outside the Git worktree. The original study cohort is not
distributed with the repository.

## Controlled data and artifacts

The following materials require governed storage and access controls:

- raw social-media exports, post text, images, and dynamic media;
- `identity-map.jsonl` and any table linking pseudonyms to platform identifiers;
- `subjects_summary.json` when it contains pseudonymous subject IDs, labels, or
  per-user counts;
- split assignments containing subject IDs, labels, folds, or evaluation roles;
- `predictions.jsonl`, `oof_predictions.jsonl`, scores, and decision values tied
  to subject IDs;
- generated feature bundles, checkpoints, local model snapshots, and run
  directories;
- manifests containing local paths, environment-specific details, or governed
  artifact identifiers.

Pseudonymization reduces direct identifiability but does not remove the
re-identification risk of user-level mental-health data. Access decisions should
follow the approved research protocol, platform terms, and applicable ethics and
data-protection requirements. Do not paste governed records, identifiers, model
outputs, or screenshots into public issues or pull requests.

## Version-controlled materials

Version control is appropriate for source code, configuration templates,
aggregate manuscript results, documentation, and synthetic fixtures that cannot
be linked to real users. Data-dependent outputs should remain under the external
artifact root and be reviewed before they are shared beyond the research team.

Researchers using separately authorized data should adapt it to the documented
[`input format`](input_format.md) and apply the governance requirements of their
own protocol and jurisdiction.