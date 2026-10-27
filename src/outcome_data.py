"""Conservative factual-section inputs and grouped splits for outcome prediction.

No fallback to full decisions: unknown section boundaries are exclusions. These
inputs reduce direct verdict leakage, but still require human review before any
claim about prospective clearance prediction.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PREPROCESSING_VERSION = "findings-only-v1"
LABELS = ["DENIED", "GRANTED"]
MIN_CHARS = 200
START = re.compile(r"(?im)^[ \t]*(?:\d+[.)][ \t]*)?findings[ \t]+of[ \t]+fact[ \t]*:?[ \t]*$")
END = re.compile(
    r"(?im)^[ \t]*(?:\d+[.)][ \t]*)?(?:policies|analysis|discussion|"
    r"conclusions?(?:[ \t]+of[ \t]+law)?|formal[ \t]+findings|decision|"
    r"whole[- ]person[ \t]+concept|administrative[ \t]+notice)[ \t]*:?[ \t]*$"
)
VERDICT = re.compile(
    r"\b(?:clearance|eligibility|access to classified information)\s+"
    r"(?:is|was|has been|should be|will be|must be)\s+(?:hereby\s+)?"
    r"(?:granted|denied|revoked)\b|"
    r"\b(?:grant(?:ed|ing)?|den(?:y|ied|ying)|revok(?:e|ed|ing))\s+"
    r"(?:(?:applicant(?:'s)?|his|her|a|the|security)\s+){0,4}clearance\b",
    re.I,
)


def normalized(text: str) -> str:
    return " ".join(str(text).lower().split())


def digest(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def clean_factual_text(text: str) -> str:
    """Apply the same normalization and direct-verdict filter at train/inference."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(s for s in sentences if not VERDICT.search(s))


def extract_facts(text: str) -> tuple[str, str, int, int]:
    """Return model text, exclusion reason, and original section offsets."""
    if not isinstance(text, str) or not text.strip():
        return "", "empty_text", -1, -1
    # A mislabeled appeal is still an appeal; do not train on its review of a verdict.
    if re.search(r"appeal\s+board\s+decision", text[:3000], re.I):
        return "", "appeal_document", -1, -1
    start = START.search(text)
    if not start:
        return "", "no_findings_heading", -1, -1
    end = END.search(text, start.end())
    if not end:
        return "", "no_safe_section_end", -1, -1
    facts = clean_factual_text(text[start.end():end.start()])
    if len(facts) < MIN_CHARS:
        return "", "facts_too_short", start.end(), end.start()
    return facts, "", start.end(), end.start()


def prepare_cases(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One audit record per source row, including explicit exclusion reasons."""
    if not raw.row_id.is_unique:
        raise ValueError("Source row_id must be unique")
    audit = raw[["row_id", "case_number", "case_type", "outcome"]].copy()
    audit["reason"] = ""
    records = []
    for row in raw.itertuples(index=False):
        if not isinstance(row.outcome, str) or row.outcome not in LABELS:
            reason, facts, start, end = "non_binary_outcome", "", -1, -1
        elif not isinstance(row.case_type, str) or row.case_type != "hearing":
            reason, facts, start, end = "non_hearing", "", -1, -1
        else:
            facts, reason, start, end = extract_facts(row.full_text)
        records.append((facts, reason, start, end, digest(row.full_text)))
    fields = pd.DataFrame(records, columns=["model_text", "reason", "facts_start", "facts_end", "document_hash"], index=audit.index)
    audit["reason"] = fields.reason
    candidates = pd.concat([audit.drop(columns="reason"), fields], axis=1)
    candidates = candidates[candidates.reason.eq("")].copy()
    candidates["model_text_hash"] = candidates.model_text.map(digest)
    # Identical facts with different source labels cannot supply an unambiguous target.
    for column in ["document_hash", "model_text_hash"]:
        conflicts = candidates.groupby(column).outcome.nunique()
        bad = candidates[column].isin(conflicts[conflicts > 1].index)
        audit.loc[candidates.index[bad], "reason"] = "conflicting_duplicate_labels"
        candidates = candidates[~bad].copy()
    # Connect case IDs AND duplicate texts transitively before removing copies.
    parent = {int(r): int(r) for r in candidates.row_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for row in candidates.itertuples(index=False):
        case = normalized(row.case_number)
        docket = re.search(r"\b\d{2}-\d{4,6}\b", case)
        case = docket.group() if docket else case
        keys = [("document", row.document_hash), ("facts", row.model_text_hash)]
        if case and case not in {"unknown", "none", "nan", ""}:
            keys.append(("case", case))
        for key in keys:
            if key in seen:
                a, b = find(int(row.row_id)), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = int(row.row_id)
    candidates["split_group"] = [find(int(r)) for r in candidates.row_id]
    dup = candidates.duplicated("model_text_hash", keep="first")
    audit.loc[candidates.index[dup], "reason"] = "duplicate_model_text"
    candidates = candidates[~dup].copy().reset_index(drop=True)
    audit.loc[audit.reason.eq(""), "reason"] = "included"
    return candidates, audit


def split_cases(cases: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Approximately 64/16/20 by rows; whole groups take priority over ratios."""
    out = cases.copy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = next(splitter.split(out, out.outcome, out.split_group))
    development = out.iloc[dev]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner.split(development, development.outcome, development.split_group))
    out["split"] = ""
    out.loc[out.index[test], "split"] = "test"
    out.loc[development.index[train_local], "split"] = "train"
    out.loc[development.index[val_local], "split"] = "validation"
    assert_split_integrity(out)
    return out


def assert_split_integrity(cases: pd.DataFrame) -> None:
    for column in ["split_group", "document_hash", "model_text_hash"]:
        if cases.groupby(column)["split"].nunique().max() > 1:
            raise ValueError(f"Cross-partition leakage in {column}")
    if cases.row_id.duplicated().any() or cases["split"].eq("").any():
        raise ValueError("Invalid row IDs or unassigned split")
    for name in ["train", "validation", "test"]:
        if set(cases.loc[cases["split"].eq(name), "outcome"]) != set(LABELS):
            raise ValueError(f"{name} does not contain both target classes")
