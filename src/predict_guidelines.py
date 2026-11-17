"""Predict guideline letters from a UTF-8 narrative or parsed decision."""
import argparse
import json
from pathlib import Path

import joblib

from src.guideline_data import LABEL_VERSION, LETTERS
from src.outcome_data import MIN_CHARS, PREPROCESSING_VERSION, START, clean_factual_text, extract_facts
from src.train_guidelines import ARTIFACT_DIR, score_matrix


def predict(model_path, text, input_kind="facts"):
    if input_kind == "decision":
        text, reason, _, _ = extract_facts(text)
        if reason:
            raise ValueError(f"Cannot extract a supported factual section: {reason}")
    elif input_kind == "facts":
        if START.search(text):
            raise ValueError("For a full decision, use --input-kind decision")
        text = clean_factual_text(text)
        if len(text) < MIN_CHARS:
            raise ValueError(f"Provide at least {MIN_CHARS} characters of factual narrative")
    else:
        raise ValueError(f"Unsupported input kind: {input_kind}")
    bundle = joblib.load(model_path)
    if (bundle["label_version"], bundle["preprocessing_version"], tuple(bundle["letters"])) != (LABEL_VERSION, PREPROCESSING_VERSION, LETTERS):
        raise ValueError("Model and preprocessing versions do not match. Retrain the classifier.")
    scores = score_matrix(bundle, [text])[0]
    return {
        "predicted_guidelines": [letter for letter, score, threshold in zip(LETTERS, scores, bundle["thresholds"]) if score >= threshold],
        "uncalibrated_scores": {letter: float(score) if model is not None else None
                                for letter, score, model in zip(LETTERS, scores, bundle["models"])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-kind", choices=("facts", "decision"), default="facts")
    parser.add_argument("--model", type=Path, default=ARTIFACT_DIR / "guideline_model.joblib",
                        help="Only load model artifacts you trust")
    args = parser.parse_args()
    try:
        result = predict(args.model, args.input.read_text(encoding="utf-8"), args.input_kind)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
