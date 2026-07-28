# Contributing

Contributions should preserve the scientific protocol, user-level evaluation
boundary, and data-governance assumptions documented in the repository.

Before opening a pull request:

1. Run `python -m unittest discover -s tests -v`.
2. Run `python -m ruff check src tests tools`.
3. Run `python tools/repository_check.py`.
4. Update configuration examples, manifests, tests, and documentation together
   when an experiment interface or artifact identity changes.

Use synthetic fixtures for tests and examples. Do not commit or paste raw
social-media records, user or post identifiers, identity mappings, feature
bundles, predictions, checkpoints, credentials, local paths, or other governed
artifacts. Suspected exposure of personal data or credentials should not be
reported with the sensitive material in a public issue.

Changes to cohort construction, label policy, split logic, preprocessing,
thresholds, model selection, or reported-result generation should explain their
scientific impact and include regression tests. See
[`docs/development.md`](docs/development.md) and
[`docs/data_governance.md`](docs/data_governance.md) for additional guidance.