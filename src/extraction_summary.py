"""Print a compact summary of the PDF extraction output.

This can be run directly:
    python3 -m src.extraction_summary

It can also be called from src/pdf_extractor.py after extraction finishes:
    from src.extraction_summary import print_extraction_summary
    print_extraction_summary(output_path)
"""

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_PARQUET_PATH = Path("data/processed/pdf_extractions.parquet")
PREVIEW_CHARACTERS = 1800


def print_extraction_summary(parquet_path=DEFAULT_PARQUET_PATH, preview_characters=PREVIEW_CHARACTERS):
    """Print metadata and original/cleaned text previews from an extraction Parquet file."""
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        print(f"Extraction output not found: {parquet_path}")
        print("Run the PDF extractor first, for example:")
        print("  python3 -m src.pdf_extractor ./data/raw_pdfs")
        return

    df = pd.read_parquet(parquet_path)

    if df.empty:
        print(f"No extracted PDFs were found in {parquet_path}.")
        return

    preferred_columns = [
        "source_file",
        "case_number",
        "decision_date",
        "judge",
        "guidelines",
        "extraction_status",
    ]
    available_columns = [column for column in preferred_columns if column in df.columns]

    print("\n" + "=" * 72)
    print("PDF EXTRACTION SUMMARY")
    print("=" * 72)
    print(f"Output file: {parquet_path}")
    print(f"PDFs processed: {len(df)}")

    if "extraction_status" in df.columns:
        print("\nExtraction status counts:")
        print(df["extraction_status"].fillna("unknown").value_counts().to_string())

    if available_columns:
        print("\nCase metadata:")
        print(df[available_columns].to_string(index=False, max_colwidth=80))

    row = df.iloc[0]

    print("\n" + "-" * 72)
    print(f"TEXT PREVIEW: {row.get('source_file', 'First extracted PDF')}")
    print("-" * 72)

    original_text = row.get("full_text", "")
    cleaned_text = row.get("cleaned_text", "")

    if isinstance(original_text, str) and original_text.strip():
        print("\n--- ORIGINAL TEXT ---\n")
        print(original_text[:preview_characters])
    else:
        print("\nNo original full_text preview is available.")

    if isinstance(cleaned_text, str) and cleaned_text.strip():
        print("\n--- CLEANED TEXT ---\n")
        print(cleaned_text[:preview_characters])
    else:
        print("\nNo cleaned_text field is available. Check that the updated PDF extractor ran.")

    print("\n" + "=" * 72)
    print("Summary complete. The previews above show the first extracted PDF.")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize PDF extraction results.")
    parser.add_argument(
        "parquet_path",
        nargs="?",
        default=str(DEFAULT_PARQUET_PATH),
        help=f"Path to extraction Parquet file (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--preview-characters",
        type=int,
        default=PREVIEW_CHARACTERS,
        help=f"Number of characters to show per text preview (default: {PREVIEW_CHARACTERS})",
    )
    args = parser.parse_args()
    print_extraction_summary(args.parquet_path, args.preview_characters)


if __name__ == "__main__":
    main()
