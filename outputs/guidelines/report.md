# Guideline classification evaluation

Training used 12,283 hearings, validation used 3,071, and testing used 3,837.

| Split | Model | Micro F1 | Macro F1 | Exact match |
|---|---|---:|---:|---:|
| validation | model | 0.894 | 0.708 | 0.742 |
| validation | majority | 0.489 | 0.058 | 0.443 |
| test | model | 0.896 | 0.775 | 0.746 |
| test | majority | 0.479 | 0.062 | 0.438 |

Macro F1 averages letters with at least one positive in that split. The majority baseline predicts each letter only if it occurs on at least half of training cases.

| Guideline | Train positives | Validation positives | Test positives | Precision | Recall | F1 | Average precision | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 1 | 0 | 0.000 | — | — | — | 0.50 |
| B | 1,681 | 416 | 541 | 0.908 | 0.970 | 0.938 | 0.969 | 0.35 |
| C | 608 | 139 | 225 | 0.781 | 0.840 | 0.809 | 0.872 | 0.80 |
| D | 259 | 62 | 73 | 0.718 | 0.836 | 0.772 | 0.808 | 0.70 |
| E | 3,838 | 892 | 1,138 | 0.823 | 0.808 | 0.815 | 0.898 | 0.55 |
| F | 7,430 | 1,849 | 2,290 | 0.982 | 0.988 | 0.985 | 0.995 | 0.20 |
| G | 926 | 215 | 286 | 0.899 | 0.867 | 0.883 | 0.942 | 0.80 |
| H | 1,606 | 440 | 541 | 0.899 | 0.935 | 0.917 | 0.949 | 0.55 |
| I | 111 | 32 | 41 | 0.792 | 0.927 | 0.854 | 0.933 | 0.70 |
| J | 1,628 | 368 | 503 | 0.716 | 0.751 | 0.733 | 0.799 | 0.70 |
| K | 130 | 37 | 41 | 0.738 | 0.756 | 0.747 | 0.808 | 0.80 |
| L | 20 | 5 | 5 | 0.103 | 0.600 | 0.176 | 0.289 | 0.50 |
| M | 126 | 36 | 42 | 0.722 | 0.619 | 0.667 | 0.743 | 0.90 |

A dash indicates an undefined metric. Rare-label results have high sampling uncertainty. Scores are uncalibrated.

Labels come from numbered guideline paragraphs in the Formal Findings of historical DOHA decisions. The model reads the earlier Findings of Fact. The targets represent recorded guideline mentions across the archive, with possible missing parser matches, historical guideline differences, and retrospective signals in the facts. They are not independently annotated applicability labels.

Exact duplicate facts are deduplicated or excluded when labels conflict. Related case groups and exact text hashes do not cross partitions. TF-IDF and each Logistic Regression model are fitted on training data only. The fixed threshold grid is selected on validation data when the label has at least 10 positives and negatives there. Other thresholds remain 0.50. Held-out testing does not change the models or thresholds.
