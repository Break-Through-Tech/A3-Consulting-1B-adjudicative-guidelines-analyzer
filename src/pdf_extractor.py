"""Extract, clean, and summarize text from DOHA PDF decisions.

Run from the project root:
    python3 -m src.pdf_extractor ./data/raw_pdfs

Outputs:
    data/processed/pdf_extractions.parquet
    data/processed/extraction_summary.csv
    data/processed/extracted_text/<pdf-name>.txt
"""

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import pymupdf
    PDF_LIBRARY = pymupdf
except ImportError:
    try:
        import fitz as PDF_LIBRARY
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required. Install it with: python3 -m pip install PyMuPDF"
        ) from exc

from src.extraction_summary import print_extraction_summary


DEFAULT_OUTPUT_PATH = Path("data/processed/pdf_extractions.parquet")
DEFAULT_TEXT_DIR = Path("data/processed/extracted_text")


KNOWN_HEADINGS = {
    "statement of the case",
    "findings of fact",
    "conclusions of law",
    "formal findings",
    "decision",
    "personal background",
    "financial considerations",
    "foreign influence",
    "foreign preference",
    "drug involvement",
    "guideline b",
    "guideline h",
    "burden of proof",
    "whole-person concept",
    "whole person concept",
    "summary",
    "appearances",
}


def is_heading(line):
    """Return True for likely section headings, without treating every line as one."""
    cleaned = line.strip().strip(":")
    if not cleaned:
        return False

    if cleaned.lower() in KNOWN_HEADINGS:
        return True

    # Likely numbered legal sections, such as "1. Findings of Fact".
    if re.match(r"^\d{1,2}\.\s+[A-Z][A-Za-z ]{2,60}$", cleaned):
        return True

    # All-caps headings are common in these decisions. Avoid long body lines.
    words = cleaned.split()
    if (
        1 <= len(words) <= 10
        and len(cleaned) <= 90
        and cleaned.upper() == cleaned
        and re.search(r"[A-Z]", cleaned)
        and not re.fullmatch(r"[\W\d_]+", cleaned)
    ):
        return True

    return False


def clean_extracted_text(text):
    """Normalize PDF extraction artifacts and rejoin lines into readable paragraphs."""
    if not isinstance(text, str):
        return ""

    text = (
        text.replace("\x0c", "\n")
        .replace("\u00a0", " ")
        .replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    output_lines = []
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            output_lines.append(" ".join(current_paragraph).strip())
            current_paragraph.clear()

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        # Skip empty lines, repeated underscores, and common page decoration.
        if not line:
            flush_paragraph()
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue

        if re.fullmatch(r"[_=\-.\s]{3,}", line):
            continue

        # Remove isolated layout artifacts, but retain parentheses in real text.
        if line in {"(", ")", "[", "]", "|"}:
            continue

        if is_heading(line):
            flush_paragraph()
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(line)
            output_lines.append("")
            continue

        # Preserve a new paragraph when the previous line appears to end a sentence.
        if current_paragraph and re.search(r'[.!?]["\')\]]?$', current_paragraph[-1]):
            flush_paragraph()

        current_paragraph.append(line)

    flush_paragraph()

    # Collapse repeated blank lines and trim the final result.
    cleaned_lines = []
    for line in output_lines:
        if line == "" and (not cleaned_lines or cleaned_lines[-1] == ""):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def extract_metadata(text):
    """Extract a few useful fields from the decision text using simple heuristics."""
    if not isinstance(text, str):
        text = ""

    case_match = re.search(
        r"\b(?:ISCR|DD|CAF|DOD)\s+Case\s+No\.?\s*[:#]?\s*([A-Z0-9-]+)",
        text,
        flags=re.IGNORECASE,
    )

    date_match = re.search(
        r"(?im)^\s*(\d{1,2}/\d{1,2}/\d{4}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4})\s*$",
        text,
    )

    judge_match = re.search(
        r"(?im)^\s*(?:Administrative Judge|Judge)\s*[:,-]?\s*"
        r"([A-Z][A-Za-z.' -]{1,80})\s*[:,]?\s*$",
        text,
    )
    if not judge_match:
        judge_match = re.search(
            r"(?im)^\s*([A-Z][A-Za-z.' -]{2,60}),\s*"
            r"Administrative Judge\s*:",
            text,
        )

    guideline_matches = re.findall(
        r"\bGuideline\s+([A-Z])\b",
        text,
        flags=re.IGNORECASE,
    )
    guidelines = list(dict.fromkeys(letter.upper() for letter in guideline_matches))

    decision_match = re.search(
        r"(?is)\b(?:Security concerns? under|The security concerns? under)\s+"
        r".{0,500}?\b(?:is|are)\s+(?:mitigated|not mitigated|mitigated in part)"
        r".{0,250}?(?:granted|denied|continued|revoked|eligible|ineligible)\b"
        r"[^.\n]*\.?",
        text,
    )

    decision_summary = decision_match.group(0).strip() if decision_match else None

    return {
        "case_number": case_match.group(1).strip() if case_match else None,
        "decision_date": date_match.group(1).strip() if date_match else None,
        "judge": judge_match.group(1).strip(" ,:") if judge_match else None,
        "guidelines": guidelines,
        "decision_summary": decision_summary,
    }


def extract_pdf(pdf_path):
    """Extract raw and cleaned text, plus per-page details, from one PDF."""
    pdf_path = Path(pdf_path)
    pages = []
    raw_page_texts = []
    cleaned_page_texts = []
    errors = []

    try:
        document = PDF_LIBRARY.open(str(pdf_path))
    except Exception as exc:
        return {
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": 0,
            "extracted_pages": 0,
            "character_count": 0,
            "full_text": "",
            "cleaned_text": "",
            "cleaned_character_count": 0,
            "pages": [],
            "case_number": None,
            "decision_date": None,
            "judge": None,
            "guidelines": [],
            "decision_summary": None,
            "extraction_status": "error",
            "error": str(exc),
        }

    with document:
        page_count = len(document)

        for page_index, page in enumerate(document, start=1):
            try:
                page_text = page.get_text("text") or ""
                cleaned_page_text = clean_extracted_text(page_text)
                status = "success" if page_text.strip() else "no_text"
                page_error = None
            except Exception as exc:
                page_text = ""
                cleaned_page_text = ""
                status = "error"
                page_error = str(exc)
                errors.append(f"Page {page_index}: {exc}")

            raw_page_texts.append(page_text)
            cleaned_page_texts.append(cleaned_page_text)
            pages.append(
                {
                    "page_number": page_index,
                    "text": page_text,
                    "character_count": len(page_text),
                    "cleaned_text": cleaned_page_text,
                    "cleaned_character_count": len(cleaned_page_text),
                    "extraction_status": status,
                    "error": page_error,
                }
            )

    full_text = "\n\n".join(raw_page_texts).strip()
    cleaned_text = clean_extracted_text("\n\n".join(cleaned_page_texts))
    metadata = extract_metadata(cleaned_text)

    extracted_pages = sum(bool(text.strip()) for text in raw_page_texts)
    if errors:
        extraction_status = "partial_error" if extracted_pages else "error"
    elif extracted_pages == 0:
        extraction_status = "no_text"
    else:
        extraction_status = "success"

    return {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "page_count": page_count,
        "extracted_pages": extracted_pages,
        "character_count": len(full_text),
        "full_text": full_text,
        "cleaned_text": cleaned_text,
        "cleaned_character_count": len(cleaned_text),
        "pages": pages,
        **metadata,
        "extraction_status": extraction_status,
        "error": "; ".join(errors) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean text from DOHA PDF decisions."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/raw_pdfs",
        help="Folder containing PDF files (default: data/raw_pdfs)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    text_dir = output_path.parent / "extracted_text"
    summary_csv_path = output_path.parent / "extraction_summary.csv"

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    pdf_files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    records = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] Extracting {pdf_path.name}")
        record = extract_pdf(pdf_path)
        records.append(record)

        # Write one cleaned text file per PDF.
        text_output_path = text_dir / f"{pdf_path.stem}.txt"
        text_output_path.write_text(
            record.get("cleaned_text") or "",
            encoding="utf-8",
        )

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)

    summary_columns = [
        "source_file",
        "case_number",
        "decision_date",
        "judge",
        "guidelines",
        "decision_summary",
        "page_count",
        "extracted_pages",
        "character_count",
        "cleaned_character_count",
        "extraction_status",
        "error",
    ]
    available_summary_columns = [
        column for column in summary_columns if column in df.columns
    ]
    df[available_summary_columns].to_csv(summary_csv_path, index=False)

    print("\nExtraction files saved:")
    print(f"  Parquet: {output_path}")
    print(f"  CSV summary: {summary_csv_path}")
    print(f"  Cleaned text files: {text_dir}")

    # Automatically print the same preview/metadata report after extraction.
    print_extraction_summary(output_path)


if __name__ == "__main__":
    main()