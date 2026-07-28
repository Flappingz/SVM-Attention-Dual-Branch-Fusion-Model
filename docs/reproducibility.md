# Reproducibility guide

The paper evaluates a fixed complete dual-branch model on user-level repeated
five-fold outer evaluations. The implementation keeps every user in one role,
fits preprocessing within each training fold, records out-of-fold predictions,
and computes repetition-level summaries.

1. Set the five local environment variables documented in the README.
2. Run `audit-raw` and `prepare` with the example study configuration.
3. Build frozen Qwen3-VL features for the required input condition.
4. Run `tools/run_qwen_sequence_svm_repeated_cv.py` for the temporal-summary
   primary branch.
5. Run `train-full-repeated-cv` for the hierarchical-attention auxiliary branch.
6. Run `tools/build_fixed_qwen_sequence_fusion.py` to apply the fixed 0.9/0.1
   score fusion.
7. Preserve the generated JSON summaries and out-of-fold predictions in the
   governed external artifact root; they are the source for metric tables.

## Experiment identities and cached artifacts

Dataset identities exclude machine-specific storage roots. Feature identities
include a SHA-256 content fingerprint of the local encoder snapshot, the relevant
inference-package versions, the selected Qwen attention implementation, and the
source-code state. When the Git worktree has uncommitted changes, a digest of the
tracked diff and untracked file contents is included in feature, run, and fusion
identities. Existing feature manifests are compared with the complete requested
identity before cached artifacts are reused.

The current feature-set schema is version 6. Feature sets created under an older
identity scheme may be retained as historical artifacts, but path-bearing legacy
manifests are rejected by the current training and reuse paths. Regenerate those
features before running new experiments. The controlled benchmark configuration
permits `expected_dataset_id` to be `null` when the dataset identity is supplied
by the local preparation step; provide an explicit ID when a study requires an
exact frozen-dataset check. Dataset and cross-feature alignment checks remain
active in either case.

The controlled baseline configuration is
`configs/controlled_benchmark.keyword-post-removed.gt20.json`. It uses a shared
cohort and split construction for character TF-IDF+SVM, frozen Chinese-RoBERTa
mean+LR, Qwen mean+MLP, and the hierarchical-attention baseline. Use
`run-controlled-benchmark` after preparing matching Qwen and Chinese-RoBERTa
features.

`tools/run_ablations.py` performs the modality and structural experiments. It
always retains the complete dual-branch reference, the temporal-summary primary
branch, and the fixed fusion rule. The keyword sensitivity comparison likewise
retrains the complete dual-branch model while retaining the same users,
candidate window, partitions, and fusion rule.

Use `tools/audit_ablation_report.py`,
`tools/audit_controlled_benchmark.py`, and
`tools/audit_qwen_sequence_fusion_confirmation.py` to independently recompute
and validate result layouts. Generated artifact directories remain
outside the Git worktree.