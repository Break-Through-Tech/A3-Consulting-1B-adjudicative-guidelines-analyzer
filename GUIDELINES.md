# How the two classifiers work

The outcome model predicts one stored hearing result, `GRANTED` or `DENIED`. The guideline model predicts a set of letters from A through M. For example, `[E, F]` means both letters were selected. Each letter has its own score and can be selected independently.

## Follow one historical case through training

1. **Separate the input from the answers.** The input is the case's earlier Findings of Fact. The outcome answer comes from the stored DOHA result. The guideline answers come from numbered paragraphs in the later Formal Findings section. A guideline counts as present whether the formal finding favors or goes against the applicant. We skip cases when we cannot extract a usable narrative or a recognizable numbered guideline list.
2. **Split by related case groups.** Case numbers and exact text matches link records together, including indirect links. Each group stays entirely in training, validation, or test. We remove duplicate factual inputs and exclude copies with conflicting guideline labels. Each model builds its own split for its eligible population, so its scores should be read separately.
3. **Convert words into numbers.** TF-IDF gives weight to words and two-word phrases. A phrase that occurs often in one case and rarely across training cases generally gets more weight than a phrase that appears everywhere. The vocabulary and frequency weights come only from training text.
4. **Learn one decision per target.** Logistic Regression learns weights on those text features. The outcome model learns a single binary decision. The guideline version learns a separate yes/no decision for each letter, reusing the same text features. Its letter scores are independent and do not sum to 100%.
5. **Turn scores into guideline selections.** For letters with enough validation examples, we choose the cutoff that maximizes F1 on a fixed grid. Sparse letters retain a cutoff of 0.50. Imagine E scores 0.68 with a cutoff of 0.55 and F scores 0.87 with a cutoff of 0.20. Both letters are selected. The scores are uncalibrated.
6. **Test against unseen groups.** Once the cutoffs are fixed, run the test set. Precision measures how often a selected letter appears in the recorded answers. Recall measures how many recorded letters the model finds. F1 balances the two. Micro F1 pools all letter decisions. Macro F1 averages the score across letters with test positives. Exact match requires the entire predicted set to equal the recorded set.

On a synthetic narrative about repaying overdue accounts, F received a score of 0.460, above its learned cutoff of 0.20, and the command returned `[F]`. On the first validation hearing with multiple recorded letters, the saved model returned `[E, F]`, matching the stored set. Those examples show how to read the result. The full test set measures performance.

## Run both models

From the 1B project root with its Python environment activated:

```bash
python -m src.train_outcome
python -m src.train_guidelines
python -m pytest -q
python -m src.predict_outcome --input /absolute/path/to/facts.txt
python -m src.predict_guidelines --input /absolute/path/to/facts.txt
```

Use a UTF-8 file of at least 200 characters containing factual narrative. For a full parsed decision, add `--input-kind decision` to either prediction command. The model files, split manifests, and per-case predictions are generated locally under `data/processed/`. Only load trusted joblib files.

The path through the code is `src/data_loader.py` → `src/guideline_data.py` → `src/train_guidelines.py` → `src/predict_guidelines.py`. The outcome counterpart uses `src/outcome_data.py`, `src/train_outcome.py`, and `src/predict_outcome.py`. [Guideline test results](outputs/guidelines/report.md) · [Outcome test results](outputs/classifier/report.md)

## How to present the results

The guideline model was evaluated on 3,837 held-out hearings and achieved 0.896 micro F1 and 74.6% exact set match. The test had no positive A cases. L had only five positives and an F1 of 0.176. Review the per-letter table before drawing conclusions about individual guidelines. The majority baseline scored 0.479 micro F1 and 43.8% exact set match.

These experiments compare predictions with historical records, using fact sections written after adjudication. The guideline letters describe what the archive records across different eras, and the parser can miss entries. Evaluating current applications or claiming current SEAD-4 applicability would require independently reviewed labels and representative new cases.
