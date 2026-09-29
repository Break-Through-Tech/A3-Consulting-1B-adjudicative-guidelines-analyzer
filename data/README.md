# Data

## How to load the dataset

```python
from src.data_loader import load_cases

df = load_cases()   # first call builds data/processed/all_cases.parquet, later calls just read it
```

Or from the command line, to (re)build the cache without opening Python:

```bash
python -m src.data_loader          # build cache if missing/stale
python -m src.data_loader --rebuild  # force a rebuild
```

**Prerequisite:** the [`doha`](https://github.com/TeneikaAskew/doha) repo must be cloned as a
**sibling folder** next to this repo (see the root [`Challenge-Project-Overview.md`](../Challenge-Project-Overview.md)
"Getting Started" section, step 1):

```
A3-Consulting/
├── A3-Consulting-1B-adjudicative-guidelines-analyzer/   <- this repo
└── doha/                                                <- sibling clone, has the raw shards
    └── doha_parsed_cases/
        ├── all_cases_1.parquet
        └── all_cases_2.parquet
```

If you keep the `doha` repo somewhere else, set the `DOHA_RAW_DIR` environment variable to point
at its `doha_parsed_cases/` folder instead of moving anything.

The merged cache (`data/processed/all_cases.parquet` + `manifest.json`) is **git-ignored** — it's
~150-200MB and it's a derived artifact of data that's already version-controlled in the `doha`
repo, so there's no reason to duplicate it here. Every teammate builds/keeps their own local copy;
it auto-rebuilds if the raw shards change.

## Source

- **Origin:** DOHA (Defense Office of Hearings and Appeals) security-clearance decisions, parsed
  from public PDFs by the [`doha`](https://github.com/TeneikaAskew/doha) repo.
- **Raw files:** `doha_parsed_cases/all_cases_1.parquet` (~94MB) and `all_cases_2.parquet` (~77MB).
- **Combined size after loading:** 33,610 rows x 18 raw columns (+ `row_id`, `source_shard` added
  by the loader), ~725MB in memory as a pandas DataFrame.

## Schema (raw columns, unmodified)

| Column | Type | Notes |
|---|---|---|
| `case_number` | str | **Not a unique row key** — see quirks below |
| `date` | str | Free-text date, inconsistent formats (see quirks) |
| `outcome` | str | `DENIED` (21,084), `GRANTED` (9,433), `UNKNOWN` (2,734), `REMANDED` (344), `REVOKED` (15) |
| `guidelines` | array\<str\> | Subset of SEAD-4 guideline letters `A`–`M`. Clean — no stray values observed |
| `summary` | str | Short case summary |
| `full_text` | str | Full parsed decision text. Length ranges ~0–115K chars (mean ~18K) |
| `sor_allegations` | array | Statement-of-Reasons allegations, often empty |
| `mitigating_factors` | array | Often empty |
| `judge` | str | |
| `source_url` | str | Original DOHA document URL |
| `formal_findings` | dict (A–M keys) | Per-guideline formal finding, often all-null |
| `case_type` | str | `hearing` (27,973) or `appeal` (5,637) |
| `appeal_board_members` | array | Empty for `hearing` rows |
| `who_appealed` | str | `APPLICANT`, `GOVERNMENT`, `BOTH`, `UNKNOWN`, or empty (hearings) |
| `judges_findings_of_fact` | str | Sectioned text extract |
| `judges_analysis` | str | Sectioned text extract, often empty |
| `discussion` | str | Sectioned text extract, often empty |
| `order` | str | Sectioned text extract |

**Columns added by the loader (not in the raw shards):**

| Column | Type | Notes |
|---|---|---|
| `row_id` | int | Synthetic, stable position after concatenation. Not derived from case content — just a handle for reproducible splits/joins |
| `source_shard` | str | `all_cases_1.parquet` or `all_cases_2.parquet` |

## Known data-quality quirks (deliberately **not** cleaned by the loader)

The loader concatenates the raw shards as-is. These are worth handling explicitly during
EDA/preprocessing rather than assuming the data is clean:

1. **No native unique row key.** `case_number` repeats legitimately — a hearing and its later
   appeal share the same number, and remanded cases get re-heard (and re-appear) under the same
   number. Use `row_id` (added by the loader) if you need a stable per-row key.
2. **857 pairs (1,714 rows) are exact duplicates** on `(case_number, case_type, full_text)` — all
   located inside `all_cases_2.parquet`, likely a parsing artifact rather than genuine re-decisions.
   Decide whether/how to de-duplicate before modeling (e.g.
   `df.drop_duplicates(subset=["case_number", "case_type", "full_text"])`).
3. **221 `(case_number, case_type)` groups have >1 *distinct* `full_text`.** These look like
   genuine re-decisions (e.g. remand -> re-hearing) rather than duplicates — don't drop these
   without inspecting them first.
4. **`date` is free text, not parsed.** ~31% (10,471/33,610) of values fail a plain
   `pd.to_datetime(df["date"])`. Formats mix `"August 2, 2007"` and `"07/10/2007"` style strings,
   and a few values are obviously wrong (e.g. a year of 3009). Plan for lenient parsing +
   sanity-check the resulting year range.
5. **Mojibake in `full_text`/related text fields** — stray `�` characters from PDF text-extraction
   encoding issues. Consider whether this needs cleanup for your NLP pipeline (e.g. for TF-IDF it's
   usually harmless noise; for LLM prompts you may want to strip/normalize it).
6. **`year` mentioned in the Challenge Project Overview does not exist as a column** — nor does
   `case_id`. Use `case_number` + a parsed `date` if you need a year.

## Suggested next steps for EDA (per the September milestone)

- Outcome distribution (`outcome`, split by `case_type`)
- Guideline frequency (`guidelines`, multi-label — will need exploding/one-hot encoding)
- Text length distributions (`full_text`)
- Decide on a dedup/date-cleaning strategy (see quirks above) before building the reproducible
  train/val/test split.

## Heads up: existing teammate EDA branch

There's a remote branch `labahadi-eda-doha` (not yet merged into `main`) with an
`notebooks/eda_doha.ipynb`. Coordinate with that teammate so this loader and their EDA notebook
don't diverge on how the data gets read in.
