# Multimodal OCD Risk-Signal Research

This project contains the implementation and experimental workflows for the EAI
PervasiveHealth 2026 paper *Multimodal OCD Risk Identification on Chinese
Social Media Using Vision--Language Embeddings*.

The study models user-level, self-reported OCD risk signals in a governed
research cohort. It is not a clinical diagnostic system, a calibrated risk
score, or a tool for screening, moderation, employment, insurance, or
individual surveillance.

## Model and evaluation

| Component | Implementation |
| --- | --- |
| Pseudonymized data preparation | `ocd_v3.data` and `configs/study.keyword-post-removed.example.json` |
| Frozen Qwen3-VL features | `ocd_v3.features.build` |
| Controlled baselines | TF-IDF+SVM, frozen Chinese-RoBERTa mean+LR, and Qwen mean+MLP modules |
| Complete model | temporal-summary SVM + hierarchical-attention auxiliary branch + fixed 0.9/0.1 fusion |
| Analyses | modality and structure ablations plus keyword-hit-post removal sensitivity |

Every modality, structural, and keyword-hit-post analysis uses the **complete
dual-branch model** as its reference. Structural changes affect only the stated
hierarchical aggregation operation; the temporal-summary SVM branch and fixed
fusion rule remain in place.

## Installation

Create an isolated Python 3.12 environment, then install the recorded package
set and this project:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-deep.lock.txt
python -m pip install -e ".[dev]"
```

For GPU reproduction, install the recorded CUDA-compatible PyTorch build first;
see [docs/environment.md](docs/environment.md). The Qwen3-VL encoder is obtained
separately under its own license.

## Configuration and data governance

The example study configuration is
`configs/study.keyword-post-removed.example.json`. Local storage locations are
provided through environment variables:

```powershell
$env:OCD_RAW_ROOT = '<governed-raw-root>'
$env:OCD_ARTIFACT_ROOT = '<external-artifact-root>'
$env:OCD_PRIVATE_MAPPING_ROOT = '<external-private-mapping-root>'
$env:OCD_PSEUDONYMIZATION_KEY = '<long-random-secret-kept-outside-the-repository>'
$env:QWEN3_VL_EMBEDDING_MODEL = '<local-qwen3-vl-model>'
```

On POSIX shells, set the same variables with `export NAME=value`.

`prepare` creates stable HMAC-SHA256 pseudonymous subject, post, and media IDs.
The raw identity map is written only under `OCD_PRIVATE_MAPPING_ROOT`, which must
be outside both the Git worktree and `OCD_ARTIFACT_ROOT`. Raw data, identity
mappings, model inputs, predictions, and generated artifacts remain in governed
storage; see [docs/data_governance.md](docs/data_governance.md).

## Data availability

The original Weibo cohort is not distributed with this repository because it
contains user-level mental-health signals, text, and media governed by the study
protocol, platform terms, and data-protection requirements. Consequently, the
repository alone supports software inspection, synthetic testing, and reuse on
separately authorized data; it does not permit independent reconstruction of the
paper's reported numerical results without access to the governed cohort.

The test suite uses synthetic fixtures only. Researchers working with authorized
or independently collected data can adapt it to the documented
[input layout](docs/input_format.md) and should apply their own ethics, consent,
platform-policy, and data-protection requirements.

## Reproducing the dual-branch evaluation

The following commands place generated outputs outside the repository. The
dataset and feature paths are emitted by the preceding command. The examples use
PowerShell line continuation.

```powershell
python -m ocd_v3 --config configs/study.keyword-post-removed.example.json audit-raw
python -m ocd_v3 --config configs/study.keyword-post-removed.example.json prepare

python -m ocd_v3 --config configs/study.keyword-post-removed.example.json build-features `
  --dataset-dir <prepared-dataset> `
  --full-config configs/full_experiment.compact-head.raw-mean-skip.json `
  --keyword-condition removed

python tools/run_qwen_sequence_svm_repeated_cv.py `
  --study-config configs/study.keyword-post-removed.example.json `
  --dataset-dir <prepared-dataset> `
  --feature-dir <qwen-feature-set> `
  --model-config configs/qwen_sequence_temporal_pyramid_svm.json `
  --split-seeds 58-65 --minimum-posts-per-subject 21

python -m ocd_v3 --config configs/study.keyword-post-removed.example.json train-full-repeated-cv `
  --dataset-dir <prepared-dataset> `
  --feature-dir <qwen-feature-set> `
  --full-config configs/full_experiment.compact-head.raw-mean-skip.json `
  --split-seeds 58-65 --minimum-posts-per-subject 21

python tools/build_fixed_qwen_sequence_fusion.py `
  --artifact-root $env:OCD_ARTIFACT_ROOT `
  --primary-summary <temporal-summary> `
  --secondary-summary <hierarchical-summary> `
  --output-root <external-fusion-output> `
  --repository (Resolve-Path .) `
  --selection-provenance "model family finalized on development partitions before evaluation on separate confirmation partitions"
```

For the controlled benchmark, ablations, audit sequence, and interpretation
limits, see [docs/reproducibility.md](docs/reproducibility.md) and
[docs/paper_protocol.md](docs/paper_protocol.md).

## Tests and consistency checks

```powershell
python -m unittest discover -s tests -v
python -m ruff check src tests tools
python tools/repository_check.py
```

Contribution and validation guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).

## License and citation

Licensed under the [MIT License](LICENSE). Citation metadata are in
[CITATION.cff](CITATION.cff).