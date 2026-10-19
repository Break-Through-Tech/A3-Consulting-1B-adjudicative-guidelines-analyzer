"""Chart generation for the DOHA EDA system."""

import os
import re

import matplotlib
matplotlib.use("Agg")  # Supports running on servers and in terminals.

import matplotlib.pyplot as plt
import pandas as pd


def safe_filename(value):
    """Convert a column/category name into a safe filename."""
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))
    return name.strip("_").lower() or "chart"


def save_bar_chart(
    table,
    label_column,
    value_column,
    title,
    output_path,
    top_n=25,
    horizontal=True,
):
    """Save a bar chart from a table containing labels and values."""
    if table is None or table.empty:
        return None

    plot_data = table[[label_column, value_column]].head(top_n).copy()

    if plot_data.empty:
        return None

    plot_data[label_column] = plot_data[label_column].astype(str)
    plot_data = plot_data.iloc[::-1] if horizontal else plot_data

    height = max(4, min(12, len(plot_data) * 0.35))
    fig, ax = plt.subplots(figsize=(10, height))

    if horizontal:
        ax.barh(
            plot_data[label_column],
            plot_data[value_column],
        )
        ax.set_xlabel(value_column.replace("_", " ").title())
        ax.set_ylabel("")
    else:
        ax.bar(
            plot_data[label_column],
            plot_data[value_column],
        )
        ax.set_ylabel(value_column.replace("_", " ").title())
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")

    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_histogram(series, title, xlabel, output_path, bins=40):
    """Save a histogram for a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of documents")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_outcome_case_type_chart(table, output_path):
    """Save a grouped bar chart comparing outcomes by case type."""
    if table is None or table.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax)
    ax.set_title("Outcome Counts by Case Type")
    ax.set_xlabel("Case Type")
    ax.set_ylabel("Number of Cases")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_charts(results, output_dir):
    """Generate all applicable charts and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    chart_specs = [
        ("outcomes", "category", "count", "Case Outcome Distribution"),
        ("case_types", "category", "count", "Cases by Type"),
        ("years", "category", "count", "Cases by Year"),
        ("common_terms", "term", "count", "Most Common Document Terms"),
    ]

    for result_key, label_col, value_col, title in chart_specs:
        table = results.get(result_key)

        if table is not None and not table.empty:
            path = os.path.join(
                output_dir, f"{safe_filename(result_key)}.png"
            )
            saved = save_bar_chart(
                table,
                label_col,
                value_col,
                title,
                path,
            )
            if saved:
                generated[result_key] = saved

    lengths = results.get("text_lengths")

    if lengths is not None:
        for column in ("word_count", "character_count"):
            path = os.path.join(
                output_dir, f"{column}_distribution.png"
            )
            saved = save_histogram(
                lengths[column],
                f"Document {column.replace('_', ' ').title()} Distribution",
                column.replace("_", " ").title(),
                path,
            )
            if saved:
                generated[column] = saved

    cross_tab = results.get("outcomes_by_case_type")

    if cross_tab is not None and not cross_tab.empty:
        path = os.path.join(output_dir, "outcomes_by_case_type.png")
        saved = save_outcome_case_type_chart(cross_tab, path)
        if saved:
            generated["outcomes_by_case_type"] = saved

    return generated