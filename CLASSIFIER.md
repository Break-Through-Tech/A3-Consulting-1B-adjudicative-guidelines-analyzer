# Outcome classifier

This implements the team's binary **DENIED / GRANTED outcome baseline** using
TF-IDF and Logistic Regression. It is separate from multi-label guideline
classification, LLM extraction, and RAG.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m src.train_outcome
python -m pytest -q tests/test_outcome.py
```

`notebooks/01_baseline.ipynb` presents the saved results and demonstrates model
loading. It trains if results are missing. Run the training command explicitly
after changing preprocessing or source data; existing notebook results are reused.

## Data policy

- Load the two canonical source shards through `src.data_loader`.
- Keep stored `GRANTED`/`DENIED` outcomes and hearing cases only; reject documents
  whose introductory text identifies an Appeal Board decision.
- Extract only a standalone **Findings of Fact** section with a recognized ending
  heading. Reject unknown boundaries and inputs shorter than 200 characters.
- Remove sentences with explicit clearance-verdict patterns. Never substitute
  full decision text when extraction fails.
- Exclude identical factual inputs carrying contradictory labels. Connect case
  numbers, normalized document hashes, and factual-text hashes into transitive
  groups **before** discarding duplicate inputs.
- Split those groups using the first fold of seeded stratified group splitting:
  approximately 64% train / 16% validation / 20% test. Whole groups take priority
  over exact ratios. Seed 42; development split seed 43.

These replace the older exploratory row splits for classifier evaluation. The
existing `data_split.ipynb` and `feature_extraction.py` continue to support their
separate exploratory workflow; this classifier does not consume those splits.

## Model and evaluation

Fixed configuration: TF-IDF with 20,000 unigram/bigram features and `min_df=2`;
Logistic Regression with `C=1`, balanced class weights, liblinear, and 1,000
maximum iterations. Fit vocabulary and model on training data only. No parameter
search or test-based selection is performed.

Report accuracy, macro/micro F1, per-class precision/recall/F1, GRANTED ROC AUC,
and confusion matrices. Compare against a training-majority dummy classifier.
The model is not refitted on validation or test rows after reporting metrics.

## Outputs

Versionable aggregate results:

- `outputs/classifier/report.md`
- `outputs/classifier/metrics.json`
- `outputs/classifier/top_features.json`
- Validation and test confusion-matrix images in the same directory

Ignored local artifacts (do not commit raw text or model binaries):

- `data/processed/classifier/outcome_model.joblib`: pipeline plus metadata
- `split_manifest.csv`: source IDs, hashes, boundaries, groups, assignments
- `row_audit.csv`: explicit inclusion/exclusion reason for every source row
- `validation_predictions.csv`, `test_predictions.csv`: outcome, prediction,
  uncalibrated probability, and source row ID
- `metadata.json`: versions, settings, split fingerprint, and counts

Use validation mistakes for development and qualitative review. Preserve test
results for final reporting; repeated changes driven by test errors invalidate
the held-out claim. Source row IDs let a reviewer inspect errors locally without
publishing case narratives in the report.

## Predict

Provide a UTF-8 text file containing a factual narrative:

```bash
python -m src.predict_outcome --input /absolute/path/to/facts.txt
```

For a full parsed decision with supported section headings:

```bash
python -m src.predict_outcome --input /absolute/path/to/decision.txt --input-kind decision
```

Prediction uses the same text normalization and explicit-verdict filtering as
training. Only load trusted local joblib artifacts. The model expects factual
input; guideline labels and outcome metadata are not features.

## Interpretation and outstanding review

The baseline predicts **stored historical labels**, not actual clearance
eligibility. Those labels have not been independently verified. Facts are
written after adjudication and can contain retrospective signals even when
explicit verdict sections are excluded. Strict filtering changes the evaluated
population, and grouping does not eliminate unidentified relationships or
near-duplicates. Results do not establish performance on future years, appeals,
or fresh applications. Probabilities are not calibrated confidence scores.

Before claiming prospective validity, manually review source labels, factual
section boundaries and errors; investigate top features; and plan an independent
or temporal validation. This implementation does not claim those research checks
are complete.
