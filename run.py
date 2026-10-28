"""Train the selected Project 1 model and create an AIcrowd submission."""

import argparse
from pathlib import Path

import numpy as np

from evaluate_svm import add_intercept
from implementations import linear_svm_sgd
from src.data_io import load_test_data, load_training_data
from src.preprocessing import BRFSSPreprocessor
from src.submission import create_csv_submission


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_DECISION_THRESHOLD = 0.5008938105844928


def parse_args():
    """Parse reproducible final-training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/submission.csv")
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.0001)
    parser.add_argument("--max-iters", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Predict +1 when the SVM decision score is at least this value.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=0.0,
        help=(
            "Positive hinge-loss weight. The default 0 uses the class ratio "
            "computed from all training labels."
        ),
    )
    parser.add_argument("--max-missing-rate", type=float, default=1.0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Development only: train on the first N rows; 0 uses all rows.",
    )
    return parser.parse_args()


def train_and_predict(
    y,
    x_train_raw,
    x_test_raw,
    feature_names,
    lambda_=0.01,
    gamma=0.0001,
    max_iters=200_000,
    seed=42,
    positive_class_weight=None,
    max_missing_rate=1.0,
    decision_threshold=DEFAULT_DECISION_THRESHOLD,
):
    """Fit preprocessing and the selected SVM, then predict test labels."""
    if positive_class_weight is None:
        num_positive = int(np.sum(y == 1))
        if num_positive == 0:
            raise ValueError("Training labels contain no positive samples")
        positive_class_weight = float(np.sum(y == -1) / num_positive)

    preprocessor = BRFSSPreprocessor(max_missing_rate=max_missing_rate)
    x_train = preprocessor.fit_transform(x_train_raw, feature_names)
    x_train = add_intercept(x_train)
    initial_w = np.zeros(x_train.shape[1], dtype=np.float64)
    weights, objective = linear_svm_sgd(
        y,
        x_train,
        lambda_=lambda_,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        seed=seed,
        positive_class_weight=positive_class_weight,
    )

    # Release the large training representation before allocating test features.
    del x_train
    x_test = add_intercept(preprocessor.transform(x_test_raw))
    if not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    scores = x_test @ weights
    predictions = np.where(scores >= decision_threshold, 1, -1)
    return predictions, {
        "objective": float(objective),
        "positive_class_weight": float(positive_class_weight),
        "num_processed_features": int(weights.size),
        "num_dropped_features": len(preprocessor.dropped_features_),
    }


def main():
    """Train on labeled rows and write a validated submission CSV."""
    args = parse_args()
    if args.max_train_rows < 0:
        raise ValueError("max-train-rows must be non-negative")
    if args.positive_class_weight < 0:
        raise ValueError("positive-class-weight must be non-negative")
    max_rows = None if args.max_train_rows == 0 else args.max_train_rows
    class_weight = (
        None if args.positive_class_weight == 0 else args.positive_class_weight
    )

    y, x_train, _, feature_names = load_training_data(
        args.data_dir, dtype=np.float32, max_rows=max_rows
    )
    x_test, test_ids, test_feature_names = load_test_data(
        args.data_dir, dtype=np.float32
    )
    if feature_names != test_feature_names:
        raise ValueError("Training and test feature columns do not match")

    predictions, details = train_and_predict(
        y,
        x_train,
        x_test,
        feature_names,
        lambda_=args.lambda_,
        gamma=args.gamma,
        max_iters=args.max_iters,
        seed=args.seed,
        positive_class_weight=class_weight,
        max_missing_rate=args.max_missing_rate,
        decision_threshold=args.decision_threshold,
    )
    output_path = create_csv_submission(test_ids, predictions, args.output)

    print("Final Linear SVM training complete")
    print(f"  training rows: {y.size}")
    print(f"  test rows: {test_ids.size}")
    print(f"  processed features: {details['num_processed_features']}")
    print(f"  dropped raw features: {details['num_dropped_features']}")
    print(f"  positive class weight: {details['positive_class_weight']:.4f}")
    print(f"  final objective: {details['objective']:.6f}")
    print(f"  decision threshold: {args.decision_threshold:.10f}")
    print(f"  positive predictions: {int(np.sum(predictions == 1))}")
    print(f"  submission: {output_path.resolve()}")


if __name__ == "__main__":
    main()
