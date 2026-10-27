"""Checks for leakage boundaries, transitive grouping and runnable artifacts."""
import joblib
import pandas as pd
import pytest

from src.outcome_data import (assert_split_integrity, extract_facts, prepare_cases,
                              split_cases)
from src.predict_outcome import predict
from src.train_outcome import train


def decision(facts):
    return "SYNOPSIS\nClearance is denied.\nFINDINGS OF FACT\n" + facts + "\nPOLICIES\nAll concerns apply.\nDECISION\nClearance is granted."


def narrative(topic, number=0):
    return (f"The applicant provided {topic} records for account {number}. "
            "The file describes employment, payments, documentation, and interviews. "
            "The applicant explained the circumstances and supplied supporting records. "
            "The hearing considered the factual timeline and the supporting evidence.")


def row(i, case, facts, label="DENIED"):
    return dict(row_id=i, case_number=case, case_type="hearing", outcome=label, full_text=decision(facts))


def corpus():
    return pd.DataFrame([
        row(i, f"22-{10000+i//2}", narrative("unpaid delinquent" if i % 4 < 2 else "paid resolved", i),
            "DENIED" if i % 4 < 2 else "GRANTED")
        for i in range(120)
    ])


def test_factual_boundaries_and_no_fallback():
    facts = narrative("financial") + " Clearance is denied."
    text, reason, start, end = extract_facts(decision(facts))
    assert reason == "" and 0 < start < end
    assert "denied" not in text.lower() and "granted" not in text.lower()
    assert "All concerns" not in text and "supporting records" in text
    assert extract_facts("FINDINGS OF FACT\n" + facts)[1] == "no_safe_section_end"
    assert extract_facts("APPEAL BOARD DECISION\n" + decision(facts))[1] == "appeal_document"


def test_transitive_group_survives_removal_of_duplicate_bridge():
    raw = pd.DataFrame([
        row(0, "22-10001", narrative("alpha")),
        row(1, "22-10001", narrative("beta")),
        row(2, "22-10002", narrative("beta")),
        row(3, "22-10002", narrative("gamma")),
    ])
    cases, audit = prepare_cases(raw)
    assert len(cases) == 3 and cases.split_group.nunique() == 1
    assert audit.loc[audit.row_id.eq(2), "reason"].iloc[0] == "duplicate_model_text"


def test_conflicting_duplicate_labels_are_excluded():
    raw = pd.DataFrame([row(0, "22-10001", narrative("debt")),
                        row(1, "22-10002", narrative("debt"), "GRANTED")])
    cases, audit = prepare_cases(raw)
    assert cases.empty
    assert set(audit.reason) == {"conflicting_duplicate_labels"}


def test_splits_are_reproducible_and_related_cases_stay_together():
    cases, _ = prepare_cases(corpus())
    a, b = split_cases(cases), split_cases(cases)
    assert a["split"].tolist() == b["split"].tolist()
    assert a.groupby("case_number")["split"].nunique().max() == 1
    assert set(a["split"]) == {"train", "validation", "test"}
    broken = a.copy()
    first = broken.index[0]
    same_group = broken.index[broken.split_group.eq(broken.loc[first, "split_group"])][-1]
    broken.loc[same_group, "split"] = "test" if broken.loc[first, "split"] != "test" else "train"
    with pytest.raises(ValueError, match="leakage"):
        assert_split_integrity(broken)


def test_train_save_reload_and_predict(tmp_path, monkeypatch):
    import src.train_outcome as module
    monkeypatch.setattr(module, "load_cases", corpus)
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    payload = train(artifacts, reports)
    model = artifacts / "outcome_model.joblib"
    result = predict(model, narrative("paid resolved"))
    assert result["predicted_outcome"] in {"DENIED", "GRANTED"}
    assert sum(result["uncalibrated_probabilities"].values()) == pytest.approx(1)
    assert predict(model, decision(narrative("paid resolved")), "decision")["predicted_outcome"] in {"DENIED", "GRANTED"}
    with pytest.raises(ValueError, match="at least"):
        predict(model, "too short")
    assert (artifacts / "split_manifest.csv").exists()
    assert (reports / "report.md").exists()
    assert set(payload["results"]) == {"validation", "test"}
    bundle = joblib.load(model)
    assert bundle["pipeline"]["tfidf"].get_feature_names_out().size > 0
