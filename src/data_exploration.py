"""Reusable data-quality profiling utilities for Project 1."""

import csv
from pathlib import Path

import numpy as np


PROFILE_FIELDS = (
    "feature",
    "index",
    "train_missing_count",
    "train_missing_rate",
    "test_missing_count",
    "test_missing_rate",
    "missing_rate_gap",
    "train_unique_count",
    "train_min",
    "train_q25",
    "train_median",
    "train_q75",
    "train_max",
    "flags",
)


def profile_features(x, feature_names):
    """Return one data-quality profile dictionary per feature column."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional feature matrix")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of columns in x")

    profiles = []
    num_rows = x.shape[0]
    for index, feature_name in enumerate(feature_names):
        column = x[:, index]
        finite_values = column[np.isfinite(column)]
        missing_count = num_rows - finite_values.size
        missing_rate = missing_count / num_rows if num_rows else np.nan

        if finite_values.size:
            unique_count = np.unique(finite_values).size
            minimum, q25, median, q75, maximum = np.quantile(
                finite_values, [0.0, 0.25, 0.5, 0.75, 1.0]
            )
        else:
            unique_count = 0
            minimum = q25 = median = q75 = maximum = np.nan

        flags = []
        if missing_count == num_rows:
            flags.append("all_missing")
        elif missing_rate >= 0.90:
            flags.append("missing_ge_90pct")
        elif missing_rate >= 0.50:
            flags.append("missing_ge_50pct")
        if unique_count == 1:
            flags.append("constant")

        profiles.append(
            {
                "feature": str(feature_name),
                "index": index,
                "missing_count": int(missing_count),
                "missing_rate": float(missing_rate),
                "unique_count": int(unique_count),
                "min": float(minimum),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "max": float(maximum),
                "flags": flags,
            }
        )
    return profiles


def combine_profiles(train_profiles, test_profiles):
    """Combine aligned train and test profiles into rows for CSV output."""
    if len(train_profiles) != len(test_profiles):
        raise ValueError("Train and test profiles have different lengths")

    rows = []
    for train, test in zip(train_profiles, test_profiles):
        if train["feature"] != test["feature"]:
            raise ValueError("Train and test feature names are not aligned")

        flags = list(train["flags"])
        flags.extend(flag for flag in test["flags"] if flag not in flags)
        missing_rate_gap = abs(train["missing_rate"] - test["missing_rate"])
        if missing_rate_gap >= 0.10:
            flags.append("missing_gap_ge_10pct")

        rows.append(
            {
                "feature": train["feature"],
                "index": train["index"],
                "train_missing_count": train["missing_count"],
                "train_missing_rate": train["missing_rate"],
                "test_missing_count": test["missing_count"],
                "test_missing_rate": test["missing_rate"],
                "missing_rate_gap": missing_rate_gap,
                "train_unique_count": train["unique_count"],
                "train_min": train["min"],
                "train_q25": train["q25"],
                "train_median": train["median"],
                "train_q75": train["q75"],
                "train_max": train["max"],
                "flags": ";".join(flags),
            }
        )
    return rows


def write_profile_csv(rows, output_path):
    """Write combined feature profiles to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_path,
    rows,
    train_shape,
    test_shape,
    label_counts,
    max_rows=None,
):
    """Write a concise Markdown summary of the profiling results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suspicious = [row for row in rows if row["flags"]]
    high_missing = sorted(
        rows, key=lambda row: row["train_missing_rate"], reverse=True
    )[:15]
    largest_gaps = sorted(
        rows, key=lambda row: row["missing_rate_gap"], reverse=True
    )[:15]
    total_labels = sum(label_counts.values())

    lines = [
        "# Project 1 Data Exploration Summary",
        "",
        f"- Scope: {'first ' + str(max_rows) + ' rows' if max_rows else 'complete dataset'}",
        f"- Training shape: `{train_shape}`",
        f"- Test shape: `{test_shape}`",
        f"- Number of flagged features: `{len(suspicious)}`",
        "",
        "## Label distribution",
        "",
        "| Label | Count | Rate |",
        "|---:|---:|---:|",
    ]
    for label, count in sorted(label_counts.items()):
        rate = count / total_labels if total_labels else np.nan
        lines.append(f"| {label:g} | {count} | {rate:.2%} |")

    lines.extend(
        [
            "",
            "## Highest training missing rates",
            "",
            "| Feature | Train missing | Test missing | Flags |",
            "|---|---:|---:|---|",
        ]
    )
    for row in high_missing:
        lines.append(
            f"| `{row['feature']}` | {row['train_missing_rate']:.2%} | "
            f"{row['test_missing_rate']:.2%} | {row['flags'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Largest train/test missing-rate gaps",
            "",
            "| Feature | Absolute gap | Train missing | Test missing |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in largest_gaps:
        lines.append(
            f"| `{row['feature']}` | {row['missing_rate_gap']:.2%} | "
            f"{row['train_missing_rate']:.2%} | {row['test_missing_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Missing fields remain `NaN`; this report does not impute them.",
            "- Numeric survey codes may represent categories or sentinel values such as",
            "  'don't know' and 'refused'. Consult the BRFSS codebook before treating",
            "  every column as continuous.",
            "- A feature is flagged when it is constant, at least 50% missing, or has",
            "  a train/test missing-rate gap of at least 10 percentage points.",
            "- Generated files are diagnostic artifacts and are excluded from Git via",
            "  the `outputs/` rule.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
