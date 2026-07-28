# Development and validation

Before merging changes, run:

```bash
python -m unittest discover -s tests -v
python -m ruff check src tests tools
python tools/repository_check.py
```

The repository check scans tracked paths and text for governed research
artifacts, local filesystem paths, and credential-like values.

Keep raw data, identity mappings, feature bundles, predictions, checkpoints, and
run directories outside the Git worktree. When code changes affect experiment
configuration or artifact identities, document the compatibility impact and
retain historical artifacts needed to interpret earlier runs.

Review the following together when updating an experiment workflow:

- configuration examples and command-line interfaces;
- manifest fields and artifact identity derivation;
- tests for user-level splitting and fold-local preprocessing;
- `README.md`, `CITATION.cff`, and environment documentation;
- links and example commands from a clean checkout.

See [`data_governance.md`](data_governance.md) for storage and access rules for
study data and derived artifacts.
