"""Predict from a factual narrative or a full decision using a trusted local model."""
import argparse
import json
from pathlib import Path

import joblib

from src.data_loader import PROJECT_ROOT
from src.outcome_data import MIN_CHARS, PREPROCESSING_VERSION, START, clean_factual_text, extract_facts


def predict(path: Path, text: str, input_kind="facts"):
    bundle = joblib.load(path)  # Only load model artifacts you trust.
    if bundle["metadata"]["preprocessing_version"] != PREPROCESSING_VERSION:
        raise ValueError("Model and preprocessing versions do not match")
    if input_kind == "decision":
        text, reason, _, _ = extract_facts(text)
        if reason:
            raise ValueError(f"Cannot extract supported factual input: {reason}")
    else:
        if START.search(text):
            raise ValueError("For a full decision, use --input-kind decision")
        text = clean_factual_text(text)
        if len(text) < MIN_CHARS:
            raise ValueError(f"Provide at least {MIN_CHARS} characters of factual narrative")
    model = bundle["pipeline"]
    label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    return {"predicted_outcome": str(label),
            "uncalibrated_probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist())),
            "note": "Historical-label baseline; not a clearance determination. Factual-narrative use has not been prospectively validated."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 text file")
    parser.add_argument("--input-kind", choices=["facts", "decision"], default="facts")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "data/processed/classifier/outcome_model.joblib")
    args = parser.parse_args()
    try:
        result = predict(args.model, args.input.read_text(), args.input_kind)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
