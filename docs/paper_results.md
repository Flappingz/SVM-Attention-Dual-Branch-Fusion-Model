# Paper result record

This concise record mirrors the manuscript's formal numerical results. It does
not include raw predictions, artifact identifiers, or internal experiment logs.

## Controlled baselines and confirmation evaluation

Values are mean user-level out-of-fold metrics across eight complete evaluations.
The model family and fusion rule were finalized using development partitions
before evaluation on separate confirmation partitions.

| Model | F1 | ROC-AUC | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Character TF-IDF + SVM (development) | 0.7319 | 0.7913 | 0.7196 | 0.7053 | 0.7612 |
| Frozen RoBERTa mean + LR (development) | 0.6621 | 0.7129 | 0.6766 | 0.6984 | 0.6320 |
| Qwen3-VL mean + MLP (development) | 0.7150 | 0.7614 | 0.7076 | 0.7006 | 0.7317 |
| Hierarchical attention (development) | 0.6977 | 0.7488 | 0.6836 | 0.6717 | 0.7261 |
| TF-IDF + SVM (confirmation) | 0.7239 | 0.7923 | 0.7119 | 0.6985 | 0.7514 |
| Temporal-summary SVM | 0.7432 | 0.7867 | 0.7352 | 0.7248 | 0.7626 |
| Hierarchical-attention auxiliary branch | 0.7058 | 0.7636 | 0.6984 | 0.6937 | 0.7205 |
| **Complete dual-branch fusion** | **0.7647** | **0.8091** | **0.7564** | **0.7432** | **0.7879** |

The complete dual-branch model exceeded the same-partition TF-IDF baseline by
mean paired gains of 0.0408 F1 and 0.0168 ROC-AUC; both differences were positive
in all eight partition sets.

## Complete-model ablations and sensitivity

Values are mean ± SD across the eight evaluations. Each condition refers to the
complete dual-branch model, not a single branch.

| Group | Condition | F1 | ROC-AUC |
| --- | --- | ---: | ---: |
| Modality | Image only | 0.6452 ± 0.0103 | 0.6617 ± 0.0111 |
| Modality | Metadata only | 0.6788 ± 0.0124 | 0.7276 ± 0.0131 |
| Modality | Text only | 0.7288 ± 0.0145 | 0.7786 ± 0.0101 |
| Modality | Text + image | 0.7361 ± 0.0221 | 0.7800 ± 0.0094 |
| Modality | Text + metadata | 0.7421 ± 0.0182 | 0.8061 ± 0.0102 |
| Modality | Text + image + metadata | 0.7647 ± 0.0138 | 0.8091 ± 0.0087 |
| Structure | Without post-level cross-attention | 0.7638 ± 0.0133 | 0.8100 ± 0.0078 |
| Structure | Without user-level self-attention | 0.7668 ± 0.0154 | 0.8104 ± 0.0076 |
| Structure | Mean pooling at both levels | 0.7595 ± 0.0158 | 0.8073 ± 0.0116 |
| Keyword | Keyword-hit posts retained | 0.7743 ± 0.0134 | 0.8260 ± 0.0076 |
| Keyword | Keyword-hit posts removed | 0.7647 ± 0.0138 | 0.8091 ± 0.0087 |

Structural differences received no Holm-corrected support. Retaining
keyword-hit posts increased F1 by 0.0096 and ROC-AUC by 0.0169. This is a
sensitivity analysis of deleting entire posts and associated modalities, not a
causal attribution to keyword tokens.
