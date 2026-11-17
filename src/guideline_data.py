"""Build guideline targets from explicit DOHA formal findings."""
from __future__ import annotations

import re
import string

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.outcome_data import digest, extract_facts, related_case_groups

LETTERS = tuple(string.ascii_uppercase[:13])
LABEL_VERSION = "formal-findings-paragraphs-v1"
FORMAL = re.compile(r"^[ \t]*formal[ \t]+findings[ \t]*:?[ \t]*$", re.I | re.M)
FORMAL_END = re.compile(r"^[ \t]*(?:decision|conclusions?(?:[ \t]+of[ \t]+law)?)[ \t]*:?[ \t]*$", re.I | re.M)
PARAGRAPH = re.compile(
    r"^[ \t]*paragraph[ \t]+(?P<number>\d+)\s*[,.:–-]?\s*guideline[ \t]+(?P<letter>[A-M])\b",
    re.I | re.M,
)


def formal_guidelines(document: str, after: int) -> tuple[str, ...]:
    """Read the numbered guideline list, rejecting obvious gaps and duplicate IDs."""
    heading = FORMAL.search(document, after)
    if not heading:
        return ()
    end = FORMAL_END.search(document, heading.end())
    section = document[heading.end():end.start() if end else len(document)]
    entries = [(int(m['number']), m['letter'].upper()) for m in PARAGRAPH.finditer(section)]
    if [number for number, _ in entries] != list(range(1, len(entries) + 1)):
        return ()
    if len({letter for _, letter in entries}) != len(entries):
        return ()
    return tuple(sorted(letter for _, letter in entries))


def prepare_guidelines(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if not raw.row_id.is_unique:
        raise ValueError("Source row IDs must be unique")
    records = []
    disposition = {"non_hearing": 0, "unusable_facts": 0, "no_reliable_formal_labels": 0,
                   "conflicting_factual_duplicates": 0, "duplicate_facts": 0}
    for row in raw.itertuples(index=False):
        if not isinstance(row.case_type, str) or row.case_type != "hearing":
            disposition["non_hearing"] += 1
            continue
        facts, reason, _, end = extract_facts(row.full_text)
        if reason:
            disposition["unusable_facts"] += 1
            continue
        labels = formal_guidelines(row.full_text, end)
        records.append((row.row_id, row.case_number, facts, labels, digest(row.full_text), digest(facts)))
    frame = pd.DataFrame(records, columns=["row_id", "case_number", "model_text", "labels",
                                           "document_hash", "model_text_hash"])
    frame["split_group"] = related_case_groups(frame)
    labeled = frame.labels.map(bool)
    disposition["no_reliable_formal_labels"] = int((~labeled).sum())
    frame = frame[labeled].copy()
    conflicts = frame.groupby("model_text_hash").labels.transform("nunique").gt(1)
    disposition["conflicting_factual_duplicates"] = int(conflicts.sum())
    frame = frame[~conflicts]
    unique = frame.drop_duplicates("model_text_hash")
    disposition["duplicate_facts"] = len(frame) - len(unique)
    disposition["included"] = len(unique)
    return unique.reset_index(drop=True), disposition


def split_guidelines(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Hold out two sets of case groups before any feature fitting or scoring."""
    frame = frame.reset_index(drop=True).copy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(outer.split(frame, groups=frame.split_group))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    train, validation = next(inner.split(frame.iloc[development], groups=frame.iloc[development].split_group))
    frame.loc[development[train], "split"] = "train"
    frame.loc[development[validation], "split"] = "validation"
    frame.loc[test, "split"] = "test"
    for key in ("split_group", "document_hash", "model_text_hash"):
        if frame.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"Cross-partition overlap for {key}")
    return frame
