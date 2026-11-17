"""Train and evaluate a multi-label HEARING guideline classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.data_loader import PROJECT_ROOT, load_cases
from src.guideline_data import LABEL_VERSION, LETTERS, prepare_guidelines, split_guidelines
from src.outcome_data import PREPROCESSING_VERSION

ARTIFACT_DIR = PROJECT_ROOT / "data/processed/guidelines"
REPORT_DIR = PROJECT_ROOT / "outputs/guidelines"
MIN_VALIDATION_POSITIVES = 10


def score_matrix(bundle, texts):
    features = bundle["vectorizer"].transform(texts)
    return np.column_stack([
        model.predict_proba(features)[:, 1] if model is not None else np.zeros(features.shape[0])
        for model in bundle["models"]
    ])


def validation_threshold(target, scores):
    """Choose from a fixed grid using validation only. Sparse labels use 0.50."""
    if min(target.sum(), len(target) - target.sum()) < MIN_VALIDATION_POSITIVES:
        return 0.5
    grid = np.linspace(0.10, 0.90, 17).round(2)
    return float(max(grid, key=lambda t: (f1_score(target, scores >= t, zero_division=0), -abs(t - 0.5))))


def metrics(target, predicted, scores):
    supported = target.sum(axis=0) > 0
    per_label = {}
    for i, letter in enumerate(LETTERS):
        actual, selected = target[:, i], predicted[:, i]
        per_label[letter] = {
            "support": int(actual.sum()),
            "predicted": int(selected.sum()),
            "precision": float(precision_score(actual, selected, zero_division=0)) if selected.any() else None,
            "recall": float(recall_score(actual, selected, zero_division=0)) if supported[i] else None,
            "f1": float(f1_score(actual, selected, zero_division=0)) if supported[i] else None,
            "average_precision": float(average_precision_score(actual, scores[:, i])) if supported[i] else None,
        }
    return {
        "micro_f1": float(f1_score(target, predicted, average="micro", zero_division=0)),
        "macro_f1_supported": float(f1_score(target[:, supported], predicted[:, supported], average="macro", zero_division=0)) if supported.any() else None,
        "exact_match": float(np.mean(np.all(target == predicted, axis=1))),
        "hamming_loss": float(hamming_loss(target, predicted)),
        "per_label": per_label,
    }


def train(artifact_dir, report_dir):
    columns = ["row_id", "case_number", "case_type", "full_text"]
    source = load_cases()[columns]
    cases, disposition = prepare_guidelines(source)
    del source
    cases = split_guidelines(cases)
    parts = {name: cases[cases.split.eq(name)] for name in ("train", "validation", "test")}
    encoder = MultiLabelBinarizer(classes=LETTERS).fit(parts["train"].labels)
    y = {name: encoder.transform(part.labels) for name, part in parts.items()}
    print("Guideline partitions:", {name: len(part) for name, part in parts.items()}, flush=True)
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    features = vectorizer.fit_transform(parts["train"].model_text)
    models = []
    for i in range(len(LETTERS)):
        target = y["train"][:, i]
        model = None
        if np.unique(target).size == 2:
            model = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42, max_iter=1000)
            model.fit(features, target)
            if model.n_iter_.max() >= model.max_iter:
                raise RuntimeError(f"Guideline {LETTERS[i]} classifier did not converge")
        models.append(model)
    bundle = {"vectorizer": vectorizer, "models": models, "letters": LETTERS,
              "label_version": LABEL_VERSION, "preprocessing_version": PREPROCESSING_VERSION}
    validation = score_matrix(bundle, parts["validation"].model_text)
    thresholds = np.array([validation_threshold(y["validation"][:, i], validation[:, i])
                           for i in range(len(LETTERS))])
    bundle["thresholds"] = thresholds
    majority = y["train"].mean(axis=0) >= 0.5
    manifest_columns = ["row_id", "case_number", "labels", "split_group", "document_hash", "model_text_hash", "split"]
    manifest = cases[manifest_columns].copy()
    manifest["labels"] = manifest.labels.map("".join)
    manifest_bytes = manifest.to_csv(index=False).encode()
    results = {
        "label_source": LABEL_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "split_seed": 42, "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "packages": {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "joblib")},
        "rows": {name: len(part) for name, part in parts.items()}, "row_disposition": disposition,
        "train_support": dict(zip(LETTERS, y["train"].sum(axis=0).tolist())),
        "thresholds": dict(zip(LETTERS, thresholds.tolist())),
        "untrained_labels": [letter for letter, model in zip(LETTERS, models) if model is None],
    }
    predictions = {}
    for name, scores in (("validation", validation), ("test", score_matrix(bundle, parts["test"].model_text))):
        selected = scores >= thresholds
        baseline = np.broadcast_to(majority, y[name].shape)
        results[name] = {"model": metrics(y[name], selected, scores), "majority": metrics(y[name], baseline, baseline)}
        predictions[name] = (scores, selected)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = artifact_dir / "guideline_model.tmp"
    joblib.dump(bundle, temporary_model)
    reloaded = joblib.load(temporary_model)
    test_scores = score_matrix(reloaded, parts["test"].model_text)
    np.testing.assert_array_equal(test_scores >= reloaded["thresholds"], predictions["test"][1])
    np.testing.assert_allclose(test_scores, predictions["test"][0], rtol=0, atol=0)
    temporary_model.replace(artifact_dir / "guideline_model.joblib")
    (artifact_dir / "split_manifest.csv").write_bytes(manifest_bytes)
    for name, (scores, selected) in predictions.items():
        output = parts[name][["row_id"]].copy()
        output["actual"] = parts[name].labels.map("".join)
        output["predicted"] = ["".join(letter for letter, keep in zip(LETTERS, row) if keep) for row in selected]
        for i, letter in enumerate(LETTERS):
            output[f"score_{letter}"] = scores[:, i]
        output.to_csv(artifact_dir / f"{name}_predictions.csv", index=False)
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(report_dir / "report.md", results)
    print(f"Reproduced all {len(parts['test']):,} test predictions after reload.")
    print(f"Report saved to {report_dir / 'report.md'}")


def write_report(path, results):
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    rows = results["rows"]
    lines = ["# Guideline classification evaluation", "",
             f"Training used {rows['train']:,} hearings, validation used {rows['validation']:,}, and testing used {rows['test']:,}.", "",
             "| Split | Model | Micro F1 | Macro F1 | Exact match |", "|---|---|---:|---:|---:|"]
    for name in ("validation", "test"):
        for model, result in results[name].items():
            lines.append(f"| {name} | {model} | {fmt(result['micro_f1'])} | {fmt(result['macro_f1_supported'])} | {fmt(result['exact_match'])} |")
    lines += ["", "Macro F1 averages letters with at least one positive in that split. The majority baseline predicts each letter only if it occurs on at least half of training cases.", "",
              "| Guideline | Train positives | Validation positives | Test positives | Precision | Recall | F1 | Average precision | Threshold |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for letter in LETTERS:
        m = results["test"]["model"]["per_label"][letter]
        v = results["validation"]["model"]["per_label"][letter]
        fields = " | ".join(fmt(m[k]) for k in ("precision", "recall", "f1", "average_precision"))
        lines.append(f"| {letter} | {results['train_support'][letter]:,} | {v['support']:,} | {m['support']:,} | {fields} | {results['thresholds'][letter]:.2f} |")
    lines += ["", "A dash indicates an undefined metric. Rare-label results have high sampling uncertainty. Scores are uncalibrated.", "",
              "Labels come from numbered guideline paragraphs in the Formal Findings of historical DOHA decisions. The model reads the earlier Findings of Fact. The targets represent recorded guideline mentions across the archive, with possible missing parser matches, historical guideline differences, and retrospective signals in the facts. They are not independently annotated applicability labels.", "",
              "Exact duplicate facts are deduplicated or excluded when labels conflict. Related case groups and exact text hashes do not cross partitions. TF-IDF and each Logistic Regression model are fitted on training data only. The fixed threshold grid is selected on validation data when the label has at least 10 positives and negatives there. Other thresholds remain 0.50. Held-out testing does not change the models or thresholds.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    train(args.artifact_dir, args.report_dir)


if __name__ == "__main__":
    main()
