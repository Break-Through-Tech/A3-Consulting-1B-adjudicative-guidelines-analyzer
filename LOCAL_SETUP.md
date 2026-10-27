# Local development

Run commands from the 1B repository root. Python 3.13 is supported by the
environment verified on this Mac.

```bash
source .venv/bin/activate
python -m pip check
python -m jupyterlab
```

Choose **A3 Consulting 1B (.venv)** as the notebook kernel. If an editor asks for
a Python interpreter, select this repository's `.venv/bin/python`.

## Recreate the environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ipykernel install --sys-prefix --name a3-consulting-1b --display-name "A3 Consulting 1B (.venv)"
```

`requirements.txt` contains the current project dependencies.
`requirements-dev.txt` also supplies testing, Streamlit, embeddings, FAISS, and
the Anthropic/Gemini clients used by the separate DOHA reference implementation.
Installing those tools does not implement the team's unfinished features.
PyMuPDF stays on the 1.26 series used by the reference code.
`requirements-lock.txt` records the exact installed versions from this Mac;
use `python -m pip install -r requirements-lock.txt` to reproduce that package
set on a compatible Python/platform instead of resolving newer versions.

## Data and existing workflows

Keep the DOHA repository as a sibling of this repository:

```text
<parent>/
  A3-Consulting-1B-adjudicative-guidelines-analyzer/
  doha/doha_parsed_cases/all_cases_1.parquet
  doha/doha_parsed_cases/all_cases_2.parquet
```

The loader also accepts `DOHA_RAW_DIR` for a different data location.
No scraping or downloading is required for the supplied Parquet shards.

```bash
# Build/reuse the ignored merged cache.
python -m src.data_loader

# Run EDA into ignored local output, preserving the team's committed report.
python eda/run_eda.py --output-dir data/processed/eda-local

# Extract the supplied PDF sample into ignored local output.
python -m src.pdf_extractor
```

Notebook order:

1. `notebooks/00_load_data.ipynb` loads and inspects both source shards.
2. `data_split.ipynb` creates exploratory train/validation/test splits under
   `data/processed/splits/` (64/16/20, seed 42).
3. `notebooks/01_baseline.ipynb` presents the outcome classifier's grouped splits,
   trained TF-IDF/Logistic Regression pipeline, and held-out evaluation. See
   `CLASSIFIER.md` for `python -m src.train_outcome` and prediction instructions.
4. `python feature_extraction.py` demonstrates keyword tags and prompt formatting
   using the generated training split. It prints case text, so run locally.

The classifier uses its own Findings-of-Fact inputs and grouped partitions under
`data/processed/classifier/`. The older row-stratified split notebook remains an
exploratory example; it does not group related cases or duplicates and is not the
classifier evaluation split. Section filtering reduces direct verdict leakage;
manual label and section audits and prospective validation remain research work.

## Optional reference demo

From this repository, with this environment active:

```bash
python -m streamlit run ../doha/sead4_llm/demo_ui.py --server.address 127.0.0.1
```

This launches the **reference DOHA demo**, not a team-built 1B application.
Basic native analysis does not require an API key. Enhanced embedding analysis
may download a model on first use. Live Gemini requires `GEMINI_API_KEY` (the
reference code's actual variable); Claude requires `ANTHROPIC_API_KEY`.
Supply keys privately in the shell; do not commit them. No live API calls are
part of environment verification.

Do not install DOHA's root and LLM requirements together unchanged: they require
incompatible pandas versions. This environment retains the 1B pandas 3.0 series.
The reference index contains only three example cases, not the full corpus.
