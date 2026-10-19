"""Run the complete DOHA exploratory data analysis pipeline."""

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from analysis import build_analysis, load_dataset
from visualizations import generate_charts


def export_tables(results, output_dir):
    """Export analysis tables as CSV files."""
    exported = {}

    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            path = os.path.join(output_dir, f"{name}.csv")
            value.to_csv(path, index=True if name == "outcomes_by_case_type" else False)
            exported[name] = path

    return exported


def write_report(results, charts, output_dir, input_path):
    """Create a readable Markdown report."""
    overview = results["overview"]
    columns = results["available_columns"]

    report_path = os.path.join(output_dir, "eda_report.md")

    lines = [
        "# DOHA Exploratory Data Analysis Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Input dataset: `{input_path}`",
        "",
        "## 1. Dataset overview",
        "",
        f"- **Rows:** {overview['rows']:,}",
        f"- **Columns:** {overview['columns']:,}",
        f"- **Exact duplicate rows:** {overview['duplicate_rows']:,}",
        f"- **Approximate in-memory size:** {overview['memory_mb']:,} MB",
        "",
        "### Column types",
        "",
        "| Column | Data type |",
        "|---|---|",
    ]

    for column, dtype in overview["dtypes"].items():
        lines.append(f"| `{column}` | `{dtype}` |")

    lines.extend([
        "",
        "## 2. Data quality",
        "",
        "The following table lists columns with the greatest proportion "
        "of missing or blank values.",
        "",
        "| Column | Missing or blank | Percentage | Unique non-null |",
        "|---|---:|---:|---:|",
    ])

    for _, row in results["missing_values"].head(20).iterrows():
        lines.append(
            f"| `{row['column']}` | {int(row['missing_or_blank']):,} "
            f"| {row['missing_percent']:.2f}% "
            f"| {int(row['unique_non_null']):,} |"
        )

    lines.extend(["", "## 3. Available analysis dimensions", ""])

    for dimension, column in columns.items():
        lines.append(
            f"- **{dimension.replace('_', ' ').title()}:** "
            + (f"`{column}`" if column else "No matching column detected")
        )

    for key, title in [
        ("outcomes", "Outcome distribution"),
        ("case_types", "Case type distribution"),
        ("years", "Year distribution"),
        ("common_terms", "Most common document terms"),
    ]:
        table = results.get(key)

        if table is not None and not table.empty:
            lines.extend([
                "",
                f"## {title}",
                "",
                table.head(15).to_markdown(index=False),
            ])

    summary = results.get("text_length_summary")

    if summary is not None:
        lines.extend([
            "",
            "## 4. Document length statistics",
            "",
            summary.to_markdown(),
        ])

    cross_tab = results.get("outcomes_by_case_type")

    if cross_tab is not None and not cross_tab.empty:
        lines.extend([
            "",
            "## 5. Outcomes by case type",
            "",
            cross_tab.to_markdown(),
        ])

    lines.extend([
        "",
        "## Generated charts",
        "",
    ])

    if charts:
        for name, path in charts.items():
            lines.append(f"- `{os.path.basename(path)}` — {name.replace('_', ' ')}")
    else:
        lines.append("No charts could be generated from the available columns.")

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- These results describe the dataset; they do not establish causation.",
        "- Term frequency does not establish that a particular guideline applies.",
        "- Outcome comparisons may reflect differences in case mix, time period, "
        "or selection into hearings and appeals.",
        "- Verify column meanings and outcome labels before using results for "
        "machine-learning decisions.",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate EDA outputs for the DOHA analyzer dataset."
    )
    parser.add_argument(
        "--input",
        default="data/processed/all_cases.parquet",
        help="Path to the input Parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eda",
        help="Directory for generated reports, tables, and charts.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    print(f"Loading dataset: {args.input}")
    df = load_dataset(args.input)

    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
    print("Running EDA analysis...")

    results = build_analysis(df)

    print("Generating charts...")
    charts = generate_charts(results, charts_dir)

    print("Exporting tables...")
    tables = export_tables(results, args.output_dir)

    report_path = write_report(
        results,
        charts,
        args.output_dir,
        args.input,
    )

    overview_path = os.path.join(
        args.output_dir, "dataset_overview.json"
    )

    with open(overview_path, "w", encoding="utf-8") as file:
        json.dump(results["overview"], file, indent=2, default=str)

    print("\nEDA complete!")
    print(f"Report: {report_path}")
    print(f"Charts: {charts_dir}")
    print(f"CSV tables: {len(tables)}")
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()