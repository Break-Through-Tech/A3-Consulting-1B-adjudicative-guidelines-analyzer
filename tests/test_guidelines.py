import numpy as np
import pandas as pd

from src.guideline_data import formal_guidelines, prepare_guidelines, split_guidelines
from src.train_guidelines import metrics, validation_threshold


FACTS = "The applicant submitted records documenting the circumstances and follow-up. " * 5


def document(facts, findings):
    return f"Findings of Fact\n{facts}\nFormal Findings\n{findings}\nDecision\nParagraph 3, Guideline M"


def test_multiple_labels_come_only_from_numbered_formal_findings():
    facts = FACTS + " The preliminary narrative mentions Guideline C."
    findings = "Paragraph 1, Guideline E: AGAINST APPLICANT\nParagraph 2, Guideline F FOR APPLICANT"
    assert formal_guidelines(document(facts, findings), len(facts)) == ("E", "F")
    assert formal_guidelines(document(facts, "Paragraph 2, Guideline F"), 0) == ()
    frame = pd.DataFrame([{"row_id": 1, "case_number": "12-12345", "case_type": "hearing",
                           "full_text": document(facts, findings)}])
    prepared, audit = prepare_guidelines(frame)
    assert prepared.iloc[0].labels == ("E", "F")
    assert "AGAINST APPLICANT" not in prepared.iloc[0].model_text
    assert audit["included"] == 1


def test_duplicate_fact_links_survive_deduplication_and_group_split():
    findings = "Paragraph 1, Guideline F FOR APPLICANT"
    rows = [(1, "12-12345", "first"), (2, "12-12345", "bridge"),
            (3, "12-98765", "bridge"), (4, "12-98765", "last")]
    rows.extend((i, f"22-{i:05d}", f"different narrative {i}") for i in range(5, 24))
    frame = pd.DataFrame([{"row_id": i, "case_number": case_id, "case_type": "hearing",
                           "full_text": document(FACTS + fact, findings)} for i, case_id, fact in rows])
    prepared, audit = prepare_guidelines(frame)
    assert audit["duplicate_facts"] == 1
    split = split_guidelines(prepared)
    assert set(split.split) == {"train", "validation", "test"}
    assert split[split.row_id.isin([1, 2, 4])].split.nunique() == 1


def test_rare_label_fallback_and_unsupported_test_metrics():
    assert validation_threshold(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9])) == 0.5
    target = np.zeros((4, 13), dtype=int)
    target[0, 5] = 1
    report = metrics(target, target, target)
    assert report["micro_f1"] == report["macro_f1_supported"] == 1
    assert report["per_label"]["A"]["recall"] is None
    assert report["per_label"]["F"]["recall"] == 1
