"""Generate reproducible data-quality reports for Project 1."""

import argparse
from pathlib import Path

import numpy as np

from src.data_exploration import (
    combine_profiles,
    profile_features,
    write_profile_csv,
    write_summary,
)
from src.data_io import load_test_data, load_training_data


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
        help="Directory containing the Project 1 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory for the generated profile and Markdown summary.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally profile only the first N rows for quick iteration.",
    )
    return parser.parse_args()


def main():
    """Load data, build feature profiles and save inspectable reports."""
    args = parse_args()
    y, x_train, _, train_names = load_training_data(
        args.data_dir, max_rows=args.max_rows
    )
    x_test, _, test_names = load_test_data(
        args.data_dir, max_rows=args.max_rows
    )
    if train_names != test_names:
        raise ValueError("Training and test feature headers are not aligned")

    train_profiles = profile_features(x_train, train_names)
    test_profiles = profile_features(x_test, test_names)
    rows = combine_profiles(train_profiles, test_profiles)
    labels, counts = np.unique(y, return_counts=True)
    label_counts = {
        float(label): int(count) for label, count in zip(labels, counts)
    }

    profile_path = args.output_dir / "feature_profile.csv"
    summary_path = args.output_dir / "summary.md"
    write_profile_csv(rows, profile_path)
    write_summary(
        summary_path,
        rows,
        x_train.shape,
        x_test.shape,
        label_counts,
        max_rows=args.max_rows,
    )

    print(f"Feature profile written to: {profile_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
