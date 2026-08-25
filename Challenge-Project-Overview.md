# Explainable AI for Security Clearance Decision

**Company / Org:** A3 Consulting LLC  
**Challenge Advisor:** Teneika Askew, teneika.askew@a3consultingllc.com  
**AI Studio Coach:** Shaun Figueiro, shaun.figueiro@breakthroughtech.org                 
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About A3 Consulting LLC

A3 Consulting works at the intersection of data science and national security policy, building tools that help reviewers and adjudicators reason over public precedent. This project adapts their SEAD‑4 expertise into a reproducible educational pipeline that demonstrates explainable automated analysis of security‑clearance decisions.

---

## 🎯 The Challenge

### Project Summary
Build an explainable NLP + LLM pipeline that ingests ~36,700 public DOHA security‑clearance decision documents to:

- Predict the likely adjudication outcome for a case,
- Identify which SEAD‑4 guideline letters (A–M) are relevant (multi‑label classification),
- Extract specific disqualifiers and mitigating factors with supporting evidence (text spans and AG paragraph references),
- Retrieve and surface relevant precedent cases (RAG) with relevance scores and short similarity notes, so a human reviewer can quickly verify the model's reasoning and recommendations.

### Success Criteria
- Quantitative:
  - Strong baseline-to‑improved outcome prediction on held‑out cases (report Accuracy, Macro/Micro F1; per‑label metrics for A–M).
  - Reliable multi‑label guideline classification (per‑label precision/recall/F1).

- Qualitative:
  - Extracted disqualifiers/mitigators are traceable to source text spans or explicit SEAD‑4 paragraphs; show provenance for LLM outputs.
  - Precedent retrieval returns semantically relevant cases with interpretable similarity indicators and no obvious hallucination.

- Deliverable:
  - A working end‑to‑end demo (Streamlit or similar) that accepts a case (PDF/text) and outputs: recommendation + confidence, per‑guideline assessments, cited evidence (text spans + AG references), and similar precedents.

### Stretch Goals
- Calibration analysis of confidence scores and per‑label calibration.
- Fairness / consistency audits (do similar fact patterns across years produce consistent outputs?).
- Fine‑tune a compact open LLM on the parsed DOHA corpus for improved extraction quality.
- Automated evaluation of explanation traceability (matching extracted spans against ground truth).
  
## Project Milestones

Use these milestones as your high‑level roadmap; teams should create a GitHub Projects board to track weekly tasks.

| Month     | Milestone             | Key Activities                                                      |
|-----------|-----------------------|---------------------------------------------------------------------|
| **September** | Data & Baseline | Onboard team; load committed Parquet shards; exploratory data analysis (outcome distribution, guideline frequency, text lengths); implement a TF‑IDF + Logistic Regression baseline for outcome prediction; produce a short EDA report and a reproducible train/val/test split. |
| **October** | Guideline Classification & Extraction | Build and validate a multi‑label classifier for SEAD‑4 guidelines (A–M); design LLM prompt templates + small rule‑based heuristics for extracting disqualifiers and mitigators; start building a RAG index (embeddings + vector store) from parsed cases. |
| **November** | End‑to‑End Analyzer & Demo | Integrate classifier, extraction, and RAG retrieval into a single pipeline that outputs structured SEAD‑4 JSON with provenance; build a Streamlit demo UI; run quantitative + manual qualitative evaluation; finalize README and final presentation materials. |

> Note for the team: create a GitHub Project board in this repository to break these milestones into weekly tasks (Projects → New project → Board).

---

## 📊 Dataset

**Name and Source:** DOHA Security Clearance Decision Documents  
**Format:** Parquet (committed shards), JSON (larger, local), PDF (original documents)  
**Size:** Parquet shards committed (~100–200 MB total); full PDF+JSON archive ~5–10 GB if re-created locally  
**Location:** https://github.com/TeneikaAskew/doha

### Key Details
- Approximately ~36,700 publicly available DOHA decisions:
  - ~28,650 hearing decisions (initial adjudications)
  - ~8,050 appeal decisions
- The referenced DOHA repo contains parsing & scraping tools plus two Parquet shards (`doha_parsed_cases/all_cases_1.parquet` and `all_cases_2.parquet`) and `failed_cases.json` listing parse problems.
- The Parquet shards contain fields such as `case_id`/`case_number`, `case_type` (hearing/appeal), `year`, `outcome`, parsed `full_text` and sectioned text, and extracted metadata (docket numbers, headings). Use these shards as the canonical starting dataset.
- If you need original PDFs or wish to rebuild parsing, follow the doha repo README — note Playwright is required for automated downloads and the full run can take hours.

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP — multi‑label classification + information extraction + retrieval-augmented generation (RAG)

**Recommended Libraries / Tools**
- Data + ETL: `pandas`, `pyarrow`, `PyMuPDF` (fitz)
- Modeling: `scikit-learn` (baselines), Hugging Face `transformers`, `PyTorch` or `TensorFlow`
- Embeddings & retrieval: `sentence-transformers`, `faiss` (or a managed vector DB)
- LLMs: Google Gemini (if available) or Anthropic Claude; for reproducibility consider small open models (Llama-family)
- Demo: `streamlit`
- Dev tooling: `pytest` for tests, pre-commit hooks, GitHub Actions for basic CI

**Evaluation Metrics**
- Outcome prediction: Accuracy, Macro / Micro F1, per-class Precision/Recall, AUC as applicable
- Guideline classification: per-label Precision/Recall/F1, average precision
- Precedent retrieval: precision@k, MRR on a manually labeled sample
- Explainability: manual traceability pass rate on a random sample (e.g., does the extracted disqualifier link to a correct text span and AG paragraph?)

---

## 📚 Resources to Get Started

**Repo & Docs**
- Primary dataset & tooling: https://github.com/TeneikaAskew/doha (README contains step-by-step scrape, parse, and index instructions)  
- Project codebase: see `sead4_llm/` for `analyze.py`, `build_index.py`, and prompt templates.

**Background Reading**
- SEAD‑4 Adjudicative Guidelines: https://www.dni.gov/files/NCSC/documents/Regulations/SEAD-4-Adjudicative-Guidelines-U.pdf  
- Intro to RAG and retrieval-augmented generation: search for "RAG retrieval augmented generation tutorial"

**Technical Tutorials**
- Pandas & Parquet quick start: pandas docs on `read_parquet`  
- sentence-transformers quickstart: https://www.sbert.net/  
- Streamlit quickstart: https://docs.streamlit.io/

**Code Examples**
- The DOHA repo contains end-to-end scripts and a demo UI — use `sead4_llm/demo_ui.py`, `sead4_llm/analyze.py`, and `sead4_llm/build_index.py` as working references.



## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 60-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Email; please copy your teammates and AI Studio Coach
* Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.


**Recommended free coding / collaboration tools**
- GitHub Projects board for task tracking
- Issues for bugs and blockers
- Pull Requests for code reviews
- A shared Google Drive or repo folder for slide decks and final reports

---

## 🚀 Getting Started

Follow these steps to get a working development environment and a quick baseline running. Use the committed Parquet shards for the fastest onboarding; only re-run scraping/downloading if you need original PDFs.

### 1) Clone the repos

Clone the project repo (where this overview lives) and the DOHA data/tools repo:

```bash
git clone https://github.com/Break-Through-Tech/A3-Consulting-1C-adjudicative-guidelines-analyzer.git
git clone https://github.com/TeneikaAskew/doha.git
cd doha
```

### 2) Create a Python environment & install dependencies

Recommended: create a venv and install requirements from the DOHA repo (and the project repo if it has its own requirements).

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows (PowerShell/CMD)
pip install --upgrade pip
pip install -r requirements.txt
# If the project repo has its own requirements file, run that too from its root:
# pip install -r ../A3-Consulting-1C-adjudicative-guidelines-analyzer/requirements.txt
```

### 3) Optional: Install Playwright (only if you plan to re-run PDF downloads)

Playwright and a browser are required for the automated downloader/scraper. Skip this if you only use the committed Parquet shards.

```bash
python -m playwright install
```

### 4) Environment variables for LLM providers (optional for LLM functionality)

If you plan to run LLM-based extraction or the demo with live LLMs, set an API key for your chosen provider:

```bash
# Example (Linux/macOS)
export GOOGLE_API_KEY="your_gemini_api_key_here"
export ANTHROPIC_API_KEY="your_claude_api_key_here"

# Windows (PowerShell)
setx GOOGLE_API_KEY "your_gemini_api_key_here"
setx ANTHROPIC_API_KEY "your_claude_api_key_here"
```

If you don't have keys, use cached LLM responses in the repo or small local open models for experimentation.

### 5) Quick local checks (inspect the Parquet shards)

List the Parquet files and sizes:

```bash
ls -lh doha/doha_parsed_cases/
```

Peek columns and a few rows with a short Python one‑liner (or run inside a notebook):

```bash
python - <<'PY'
import glob, pandas as pd
files = sorted(glob.glob("doha/doha_parsed_cases/*.parquet"))
print("Found", len(files), "parquet files:", files)
df = pd.read_parquet(files[0])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(2).to_dict(orient='records'))
PY
```

### 6) Run a quick baseline (TF‑IDF + Logistic Regression)

Create a very small script or run a notebook cell to train a quick baseline on `full_text` → `outcome`. Save it as `notebooks/01_baseline.ipynb` or run inline:

```bash
python - <<'PY'
import glob, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

files = sorted(glob.glob("doha/doha_parsed_cases/*.parquet"))
df = pd.read_parquet(files[0]).dropna(subset=['full_text','outcome']).sample(frac=1, random_state=42)
df = df.head(2000)  # small sample for quick run
X_train, X_val, y_train, y_val = train_test_split(df['full_text'], df['outcome'], test_size=0.2, random_state=42, stratify=df['outcome'])
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
Xtr = vec.fit_transform(X_train)
Xv = vec.transform(X_val)
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(Xtr, y_train)
preds = clf.predict(Xv)
print(classification_report(y_val, preds))
PY
```

This gives you a quick baseline and confirms the data pipeline works.

### 7) Run the demo / analysis scripts

The DOHA repo includes an analysis entry point and a Streamlit demo. Example commands (from the doha repo root or `sead4_llm/` as noted):

```bash
# From project root (or adjust path to sead4_llm)
cd sead4_llm
# Analyze a single PDF (uses Gemini by default if API key set)
python analyze.py --input ../some_report.pdf --output result.json

# Batch mode (use Parquet and skip re-downloading)
python analyze.py --input-dir ../sample_reports --output-dir ./results

# Run Streamlit demo (if demo_ui.py exists)
python -m streamlit run demo_ui.py
```

If the demo uses cached LLM outputs, it will run without API keys. Otherwise provide API keys as environment variables.

### 8) If you need the original PDFs / want to rebuild parsing or index

Collect links (`run_full_scrape.py`) and download PDFs (`download_pdfs.py`). These steps can take hours for the full corpus and require Playwright and sufficient disk space.

```bash
# Collect links (~11 minutes)
python run_full_scrape.py

# Download and parse PDFs (test with --max-cases first)
python download_pdfs.py --max-cases 10
# Full run (example with 4 workers)
python download_pdfs.py --workers 4
```

### 9) Create the GitHub Project board & add first-week tasks

Create a Project board in the Break-Through-Tech repo and add tasks:

- **Data lead:** inspect Parquet, produce data dictionary, create train/val/test split
- **ML lead:** quick baseline + evaluation report
- **RAG lead:** run `build_index.py` on Parquet or test vector searches
- **UI lead:** run Streamlit demo and wire baseline output
- **Admin:** set up communication channel and schedule the first check‑in

### 10) Troubleshooting & notes

- Parquet shards are the fastest way to get started and are already in the DOHA repo — you do not need to re-scrape unless you require PDFs or different parsing.
- If you encounter memory issues when loading entire shards, operate on a sample or read in chunks using pyarrow/fastparquet.
- Respect the repo's ethics note: do not publish PII or raw PDFs with sensitive content in public repositories.


I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session C).
