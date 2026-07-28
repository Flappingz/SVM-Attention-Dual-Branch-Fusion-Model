# Paper protocol

The reported evaluation uses a fixed cohort of users with more than 20 usable
historical posts and a fixed most-recent window of at most 64 posts per user.
The positive class is the self-reported OCD group. Users are the unit of
splitting, training, prediction, and reported metrics.

The complete model is a fixed-weight fusion of two Qwen-based branches:

- a temporal-summary linear SVM primary branch (weight 0.9); and
- a compact hierarchical-attention auxiliary branch (weight 0.1) using frozen
  multimodal Qwen post embeddings and fold-local post metadata scaling.

The branch structure, fusion weight, user window, classification threshold,
keyword rule, and confirmation partitions are fixed before the final
evaluation. The keyword sensitivity condition removes an entire keyword-hit
post after fixing the candidate window and does not backfill older posts.

Modality and structural conditions always use the complete dual-branch model as
the common reference. The structural conditions replace only the specified
hierarchical operation; they retain the temporal-summary SVM primary branch and
the fixed fusion rule.

All reported metrics are user-level out-of-fold F1, ROC-AUC, accuracy,
precision, and recall. Paired comparisons use matched evaluation partitions,
two-sided exact sign-flip tests, and Holm adjustment. Repetitions reuse one
cohort, so intervals describe partition sensitivity rather than independent
population sampling or external validation.
