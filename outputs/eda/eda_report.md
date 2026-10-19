# DOHA Exploratory Data Analysis Report

Generated (UTC): 2026-10-19T23:43:07.363065+00:00
Input dataset: `data/processed/all_cases.parquet`

## 1. Dataset overview

- **Rows:** 33,610
- **Columns:** 20
- **Exact duplicate rows:** 0
- **Approximate in-memory size:** 691.07 MB

### Column types

| Column | Data type |
|---|---|
| `row_id` | `int64` |
| `case_number` | `str` |
| `date` | `str` |
| `outcome` | `str` |
| `guidelines` | `object` |
| `summary` | `str` |
| `full_text` | `str` |
| `sor_allegations` | `object` |
| `mitigating_factors` | `object` |
| `judge` | `str` |
| `source_url` | `str` |
| `formal_findings` | `object` |
| `case_type` | `str` |
| `appeal_board_members` | `object` |
| `who_appealed` | `str` |
| `judges_findings_of_fact` | `str` |
| `judges_analysis` | `str` |
| `discussion` | `str` |
| `order` | `str` |
| `source_shard` | `str` |

## 2. Data quality

The following table lists columns with the greatest proportion of missing or blank values.

| Column | Missing or blank | Percentage | Unique non-null |
|---|---:|---:|---:|
| `judges_analysis` | 31,303 | 93.14% | 1,365 |
| `discussion` | 30,447 | 90.59% | 1,830 |
| `judges_findings_of_fact` | 28,501 | 84.80% | 2,277 |
| `order` | 27,800 | 82.71% | 556 |
| `who_appealed` | 25,701 | 76.47% | 5 |
| `full_text` | 6 | 0.02% | 30,506 |
| `summary` | 6 | 0.02% | 29,719 |
| `appeal_board_members` | 0 | 0.00% | 325 |
| `case_number` | 0 | 0.00% | 28,163 |
| `case_type` | 0 | 0.00% | 2 |
| `date` | 0 | 0.00% | 8,464 |
| `formal_findings` | 0 | 0.00% | 3,853 |
| `guidelines` | 0 | 0.00% | 788 |
| `judge` | 0 | 0.00% | 357 |
| `mitigating_factors` | 0 | 0.00% | 20,978 |
| `outcome` | 0 | 0.00% | 5 |
| `row_id` | 0 | 0.00% | 33,610 |
| `sor_allegations` | 0 | 0.00% | 1,181 |
| `source_shard` | 0 | 0.00% | 2 |
| `source_url` | 0 | 0.00% | 32,768 |

## 3. Available analysis dimensions

- **Text:** `full_text`
- **Outcome:** `outcome`
- **Case Type:** `case_type`
- **Year:** No matching column detected

## Outcome distribution

| category   |   count |   percent |
|:-----------|--------:|----------:|
| DENIED     |   21084 |     62.73 |
| GRANTED    |    9433 |     28.07 |
| UNKNOWN    |    2734 |      8.13 |
| REMANDED   |     344 |      1.02 |
| REVOKED    |      15 |      0.04 |

## Case type distribution

| category   |   count |   percent |
|:-----------|--------:|----------:|
| hearing    |   27973 |     83.23 |
| appeal     |    5637 |     16.77 |

## Most common document terms

| term           |   count |
|:---------------|--------:|
| information    |  398999 |
| sor            |  396923 |
| under          |  353231 |
| evidence       |  348162 |
| she            |  328441 |
| financial      |  268748 |
| conduct        |  268174 |
| administrative |  257788 |
| guideline      |  255474 |
| debts          |  254006 |
| classified     |  247509 |
| directive      |  218573 |
| about          |  206955 |
| defense        |  195747 |
| any            |  195638 |

## 4. Document length statistics

|       |   character_count |   word_count |
|:------|------------------:|-------------:|
| count |          33610    |     33610    |
| mean  |          18285    |      2791.04 |
| std   |           9688.46 |      1465.66 |
| min   |              0    |         0    |
| 25%   |          12160.2  |      1867    |
| 50%   |          17327.5  |      2640    |
| 75%   |          22976.8  |      3505    |
| max   |         115347    |     17940    |

## 5. Outcomes by case type

| case_type   |   DENIED |   GRANTED |   REMANDED |   REVOKED |   UNKNOWN |
|:------------|---------:|----------:|-----------:|----------:|----------:|
| appeal      |     3269 |        70 |        230 |         0 |      2068 |
| hearing     |    17815 |      9363 |        114 |        15 |       666 |

## Generated charts

- `outcomes.png` — outcomes
- `case_types.png` — case types
- `common_terms.png` — common terms
- `word_count_distribution.png` — word count
- `character_count_distribution.png` — character count
- `outcomes_by_case_type.png` — outcomes by case type

## Interpretation notes

- These results describe the dataset; they do not establish causation.
- Term frequency does not establish that a particular guideline applies.
- Outcome comparisons may reflect differences in case mix, time period, or selection into hearings and appeals.
- Verify column meanings and outcome labels before using results for machine-learning decisions.
