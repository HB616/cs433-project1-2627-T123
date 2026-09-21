"""Entry point for reproducing the final Project 1 prediction file."""

import argparse
from pathlib import Path


REQUIRED_DATA_FILES = (
    "x_train.csv",
    "y_train.csv",
    "x_test.csv",
    "sample_submission.csv",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
        help="Directory containing the four Project 1 CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.csv"),
        help="Path for the generated AIcrowd submission.",
    )
    return parser.parse_args()


def validate_data_directory(data_dir):
    """Check that all required data files are available locally."""
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing data files in {data_dir}: {missing_text}")


def main():
    """Validate the local setup before the modeling pipeline is implemented."""
    args = parse_args()
    validate_data_directory(args.data_dir)
    print(f"Dataset found: {args.data_dir}")
    print("Modeling pipeline has not been implemented yet.")


if __name__ == "__main__":
    main()
