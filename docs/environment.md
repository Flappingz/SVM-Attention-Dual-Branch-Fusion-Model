# Experiment environment

The experiments reported in the paper used Python 3.12 on Windows with a
CUDA-capable PyTorch build. The direct package versions used for the experiments
are recorded in [`requirements-deep.lock.txt`](../requirements-deep.lock.txt).
This file is a reference constraint set, not a complete transitive lock with
hashes.

Install the matching CUDA/PyTorch wheel from the official PyTorch index before
installing the remaining requirements. The reproducibility commands set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` when deterministic neural execution is
required. Hardware and CUDA driver differences can still affect throughput and
low-level kernel availability.

The experiment configurations record the expected Qwen3-VL model identifier and
revision. The model snapshot is obtained separately under its own license. The
feature pipeline computes a SHA-256 content fingerprint over the local snapshot
and includes it, the relevant inference-package versions, and the selected
attention implementation in the feature identity. This detects local snapshot or
runtime changes without storing the local model path in generated manifests.

Only load a model snapshot from a trusted source. The Qwen loader enables
Transformers' `trust_remote_code` option, so model-supplied Python code may be
executed during loading. Provide the trusted local snapshot through
`QWEN3_VL_EMBEDDING_MODEL` and preserve its upstream license and revision
metadata.