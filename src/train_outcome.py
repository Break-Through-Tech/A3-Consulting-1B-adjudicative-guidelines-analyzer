"""Train and evaluate the fixed outcome baseline: python -m src.train_outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, load_cases
from src.outcome_data import LABELS, PREPROCESSING_VERSION, prepare_cases, split_cases


def evaluate(model, frame):
    target = frame.outcome
    prediction = model.predict(frame.model_text)
    probability = model.predict_proba(frame.model_text)[:, list(model.classes_).index("GRANTED")]
    return {
        "rows": len(frame), "accuracy": accuracy_score(target, prediction),
        "macro_f1": f1_score(target, prediction, average="macro", zero_division=0),
        "micro_f1": f1_score(target, prediction, average="micro", zero_division=0),
        "roc_auc_granted": roc_auc_score(target.eq("GRANTED"), probability),
        "classification_report": classification_report(target, prediction, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(target, prediction, labels=LABELS).tolist(),
    }, prediction, probability


def train(artifact_dir: Path, report_dir: Path, seed=42):
    print("Loading source cases...", flush=True)
    raw = load_cases()
    cases, audit = prepare_cases(raw)
    cases = split_cases(cases, seed)
    print(f"Retained {len(cases):,}/{len(raw):,} rows; splits: {cases['split'].value_counts().to_dict()}", flush=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_columns = ["row_id", "case_number", "outcome", "document_hash", "model_text_hash", "split_group", "facts_start", "facts_end", "split"]
    manifest = cases[manifest_columns]
    manifest.to_csv(artifact_dir / "split_manifest.csv", index=False)
    audit.to_csv(artifact_dir / "row_audit.csv", index=False)
    manifest_hash = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
    train_df = cases[cases["split"].eq("train")]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)),
    ])
    dummy = DummyClassifier(strategy="most_frequent")
    print("Fitting fixed TF-IDF + Logistic Regression on training data only...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_df.model_text, train_df.outcome)
    dummy.fit(train_df.model_text, train_df.outcome)
    results = {}
    for split in ["validation", "test"]:
        frame = cases[cases["split"].eq(split)]
        metrics, predictions, probabilities = evaluate(model, frame)
        dummy_metrics, _, _ = evaluate(dummy, frame)
        results[split] = {"logistic_regression": metrics, "majority_class": dummy_metrics}
        pred = frame[["row_id", "outcome"]].copy()
        pred["prediction"] = predictions
        pred["probability_granted"] = probabilities
        pred.to_csv(artifact_dir / f"{split}_predictions.csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=LABELS).plot(ax=ax, colorbar=False)
        ax.set_title(f"{split.title()} outcome predictions")
        fig.tight_layout()
        fig.savefig(report_dir / f"{split}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(split, "accuracy=", round(metrics["accuracy"], 4), "macro F1=", round(metrics["macro_f1"], 4), flush=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target": "Stored source outcome: DENIED vs GRANTED, hearing decisions only",
        "source_rows": len(raw), "included_rows": len(cases),
        "row_disposition": {str(k): int(v) for k, v in audit.reason.value_counts().items()},
        "split_counts": {name: {str(k): int(v) for k, v in frame.outcome.value_counts().items()} for name, frame in cases.groupby("split")},
        "split_manifest_sha256": manifest_hash,
        "split_policy": "First fold of seeded StratifiedGroupKFold(5), then first fold on development with seed+1; approximate 64/16/20. No fold selection by performance.",
        "group_policy": "Connected components of normalized case number, full document hash and factual input hash; remove identical factual inputs and ambiguous duplicate labels.",
        "python": platform.python_version(),
        "packages": {n: version(n) for n in ["pandas", "scikit-learn", "numpy", "joblib"]},
        "model_parameters": {"max_features": 20000, "ngram_range": [1, 2], "min_df": 2, "C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 1000},
        "evaluation_policy": "Fixed settings; train only on training partition; validation reported before test. No tuning or refit after test evaluation.",
    }
    bundle = {"pipeline": model, "metadata": metadata}
    joblib.dump(bundle, artifact_dir / "outcome_model.joblib")
    # Verify that serialization preserves real held-out predictions.
    restored = joblib.load(artifact_dir / "outcome_model.joblib")["pipeline"]
    sample = cases.loc[cases["split"].eq("test"), "model_text"].head(8)
    assert np.array_equal(model.predict(sample), restored.predict(sample))
    payload = {"metadata": metadata, "results": results}
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = model["tfidf"].get_feature_names_out()
    coefficients = model["classifier"].coef_[0]
    order = np.argsort(coefficients)
    top = {"DENIED": [{"term": str(names[i]), "coefficient": float(coefficients[i])} for i in order[:20]],
           "GRANTED": [{"term": str(names[i]), "coefficient": float(coefficients[i])} for i in order[-20:][::-1]]}
    (report_dir / "top_features.json").write_text(json.dumps(top, indent=2) + "\n")
    lines = ["# Outcome classification baseline", "", f"Generated: {metadata['created_utc']}", "",
             "Binary TF-IDF + Logistic Regression baseline for stored DOHA hearing outcomes.", "",
             "## Data and exclusions", "", f"Source rows: **{len(raw):,}**. Included rows: **{len(cases):,}**.", "",
             "| Disposition | Rows |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in metadata["row_disposition"].items()]
    lines += ["", "Only bounded Findings of Fact sections are used. No whole-document fallback. Explicit clearance-verdict sentences are filtered. Stored labels are not independently adjudicated.", "",
              "## Split design", "", metadata["split_policy"], "", metadata["group_policy"], "",
              "| Partition | DENIED | GRANTED | Total |", "|---|---:|---:|---:|"]
    for name in ["train", "validation", "test"]:
        counts = metadata["split_counts"][name]
        lines.append(f"| {name} | {counts.get('DENIED', 0):,} | {counts.get('GRANTED', 0):,} | {sum(counts.values()):,} |")
    lines += ["", "Assertions verify no group, normalized document, or normalized model text crosses partitions. The vectorizer sees training text only.", "",
              "## Held-out results", "", "| Partition | Model | Accuracy | Macro F1 | Micro F1 | ROC AUC (GRANTED) |", "|---|---|---:|---:|---:|---:|"]
    for split, models in results.items():
        for name, m in models.items():
            lines.append(f"| {split} | {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['roc_auc_granted']:.4f} |")
    lines += ["", "Micro F1 equals accuracy for this single-label task. No parameters were selected using test performance.", ""]
    for split in results:
        lines += [f"### {split.title()} per-class metrics", "", "| Outcome | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label in LABELS:
            m = results[split]["logistic_regression"]["classification_report"][label]
            lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support']):,} |")
        lines += ["", f"![{split} confusion matrix]({split}_confusion_matrix.png)", ""]
    lines += ["## Limits and error review", "",
              "- This predicts historical stored labels, not legal eligibility or an adjudicator's future decision. Source outcome extraction can be wrong.",
              "- Findings are written after adjudication; section extraction and sentence filtering reduce direct leakage but do not prove a prospective or leakage-free evaluation.",
              "- Review a manually labeled sample of section boundaries, source outcomes, and model errors before relying on the scores. No completed human audit is claimed.",
              "- Strict section rules exclude many cases. Reported scores apply only to the retained hearing subset; no appeal, multiclass, guideline-classification, or fairness performance is established.",
              "- Partitions are grouped random splits, not future-year validation. Unknown links between cases and near-duplicate documents may remain.",
              "- Probabilities are uncalibrated; do not treat them as validated confidence in clearance eligibility.",
              "- Inspect `top_features.json` for shortcuts and the local `validation_predictions.csv` for errors. Test predictions are for final reporting; avoid iterating on the test set.", "",
              "## Reproduce", "", "```bash", "python -m src.train_outcome", "```", "",
              "Model, manifests, row audit and per-case predictions are kept under ignored `data/processed/classifier/`. This report contains aggregate results only.", "",
              f"Split manifest SHA-256: `{manifest_hash}`", ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    print("Saved model, manifests, audit, predictions, metrics and report.", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "data/processed/classifier")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "outputs/classifier")
    args = parser.parse_args()
    train(args.artifact_dir, args.report_dir)


if __name__ == "__main__":
    main()
