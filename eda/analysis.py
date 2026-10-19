"""Core exploratory data analysis functions for the DOHA analyzer."""

import os
import re
from collections import Counter
from typing import Any

import pandas as pd


TEXT_CANDIDATES = (
    "full_text", "text", "document_text", "content", "extracted_text"
)
OUTCOME_CANDIDATES = (
    "outcome", "decision", "result", "case_outcome", "disposition"
)
CASE_TYPE_CANDIDATES = (
    "case_type", "type", "proceeding_type", "document_type"
)
YEAR_CANDIDATES = ("year", "decision_year", "publication_year")


def find_column(df, candidates):
    """Find a matching column without depending on capitalization."""
    columns = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def load_dataset(path):
    """Load the processed Parquet dataset."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Check the path or use --input."
        )
    return pd.read_parquet(path)



def dataset_overview(df) -> dict[str, Any]:
    """Calculate general dataset statistics."""

    # Some columns contain arrays or other unhashable objects.
    # Count duplicate rows safely by converting those values to
    # stable representations before comparing rows.
    def normalize(value):
        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(k), repr(normalize(v)))
                    for k, v in value.items()
                )
            )

        if isinstance(value, (list, tuple)):
            return tuple(normalize(v) for v in value)

        if isinstance(value, set):
            return tuple(sorted(repr(normalize(v)) for v in value))

        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    duplicate_rows = int(
        df.apply(
            lambda column: column.map(normalize)
        ).duplicated().sum()
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": duplicate_rows,
        "memory_mb": round(
            float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 2
        ),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
    }



def missing_values_table(df):
    """Count missing values, blank strings, and distinct values per column."""

    def normalize_value(value):
        """Convert structured values into hashable representations."""
        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(k), repr(normalize_value(v)))
                    for k, v in value.items()
                )
            )

        if isinstance(value, (list, tuple)):
            return tuple(normalize_value(v) for v in value)

        if isinstance(value, set):
            return tuple(
                sorted(repr(normalize_value(v)) for v in value)
            )

        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    rows = []

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        blank = 0

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            blank = int(
                series.fillna("").astype(str).str.strip().eq("").sum()
            )

        # Avoid double-counting null values as blank strings.
        blank_only = max(0, blank - missing)
        missing_or_blank = min(len(df), missing + blank_only)

        # Normalize arrays and other unhashable objects before counting.
        normalized = series.map(normalize_value)
        unique_non_null = int(normalized.nunique(dropna=True))

        rows.append({
            "column": str(column),
            "dtype": str(series.dtype),
            "missing_values": missing,
            "blank_strings": blank_only,
            "missing_or_blank": missing_or_blank,
            "missing_percent": round(
                missing_or_blank / len(df) * 100, 2
            ) if len(df) else 0.0,
            "unique_non_null": unique_non_null,
        })

    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)



def frequency_table(df, column):
    """Count categories and calculate their share of the dataset."""
    values = df[column].fillna("[Missing]").astype(str).str.strip()
    values = values.replace("", "[Blank]")
    counts = values.value_counts(dropna=False)
    total = int(counts.sum())

    return pd.DataFrame({
        "category": counts.index.astype(str),
        "count": counts.values,
        "percent": [
            round(int(value) / total * 100, 2) if total else 0.0
            for value in counts.values
        ],
    })


def text_length_table(df, text_column):
    """Calculate character and word counts for each document."""
    text = df[text_column].fillna("").astype(str)

    return pd.DataFrame({
        "character_count": text.str.len(),
        "word_count": text.str.findall(r"\b[\w'-]+\b").str.len(),
    })


def common_terms(df, text_column, top_n=30):
    """Count common terms; these counts do not establish guideline applicability."""
    stopwords = set("""
        the and for that with this from were was are has have had not but
        his her their they you all can may shall will would could into than
        then also such been being its our out who what when where which while
        upon case decision applicant clearance security judge government
        department hearing appeal
    """.split())

    counter = Counter()

    for value in df[text_column].dropna().astype(str):
        terms = re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", value.lower())
        counter.update(term for term in terms if term not in stopwords)

    return pd.DataFrame(
        counter.most_common(top_n),
        columns=["term", "count"],
    )


def build_analysis(df):
    """Build all available analysis tables from a dataframe."""
    text_col = find_column(df, TEXT_CANDIDATES)
    outcome_col = find_column(df, OUTCOME_CANDIDATES)
    type_col = find_column(df, CASE_TYPE_CANDIDATES)
    year_col = find_column(df, YEAR_CANDIDATES)

    result = {
        "overview": dataset_overview(df),
        "missing_values": missing_values_table(df),
        "available_columns": {
            "text": text_col,
            "outcome": outcome_col,
            "case_type": type_col,
            "year": year_col,
        },
    }

    if outcome_col:
        result["outcomes"] = frequency_table(df, outcome_col)

    if type_col:
        result["case_types"] = frequency_table(df, type_col)

    if year_col:
        result["years"] = frequency_table(df, year_col)

    if text_col:
        lengths = text_length_table(df, text_col)
        result["text_lengths"] = lengths
        result["text_length_summary"] = (
            lengths[["character_count", "word_count"]]
            .describe()
            .round(2)
        )
        result["common_terms"] = common_terms(df, text_col)

    if outcome_col and type_col:
        clean = df[[type_col, outcome_col]].copy()

        for column in [type_col, outcome_col]:
            clean[column] = (
                clean[column].fillna("[Missing]").astype(str).str.strip()
            )
            clean[column] = clean[column].replace("", "[Blank]")

        result["outcomes_by_case_type"] = pd.crosstab(
            clean[type_col], clean[outcome_col]
        )

    return result