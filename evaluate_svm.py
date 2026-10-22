"""Train and evaluate the NumPy-only linear SVM on a local validation split."""

import argparse
from pathlib import Path

import numpy as np

from implementations import linear_svm_sgd, predict_linear_svm
from src.data_io import load_training_data
from src.metrics import classification_metrics
from src.preprocessing import (
    BRFSSPreprocessor,
    BasicPreprocessor,
    stratified_split_indices,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"


def parse_args():
    """Parse experiment settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing x_train.csv and y_train.csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Rows to load for development; use 0 to load the full dataset.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.0003)
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocessing",
        choices=("basic", "brfss"),
        default="basic",
        help="Use the baseline or codebook-aware BRFSS feature transformation.",
    )
    parser.add_argument(
        "--max-missing-rate",
        type=float,
        default=1.0,
        help=(
            "Drop BRFSS features at or above this training missing rate; "
            "1 keeps every non-empty feature."
        ),
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=0.0,
        help=(
            "Positive-class hinge-loss weight; the default 0 automatically "
            "uses num_negative / num_positive. Use 1 for no class weighting."
        ),
    )
    return parser.parse_args()


def add_intercept(x):
    """Prepend a column of ones to a two-dimensional feature matrix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional feature matrix")
    return np.column_stack((np.ones(x.shape[0]), x))


def run_svm_experiment(
    y,
    x,
    feature_names,
    validation_ratio=0.2,
    lambda_=0.03,
    gamma=0.0003,
    max_iters=100_000,
    seed=42,
    positive_class_weight=None,
    preprocessing="basic",
    max_missing_rate=1.0,
):
    """Fit preprocessing and an SVM, then return validation results.

    Splitting happens before preprocessing. Consequently, imputation, column
    filtering and standardization are learned exclusively from the training
    partition and then reused unchanged for validation data.
    """
    train_indices, validation_indices = stratified_split_indices(
        y, validation_ratio=validation_ratio, seed=seed
    )
    y_train = y[train_indices]
    y_validation = y[validation_indices]

    if positive_class_weight is None:
        num_negative = np.sum(y_train == -1)
        num_positive = np.sum(y_train == 1)
        if num_positive == 0:
            raise ValueError("The training partition contains no positive samples")
        positive_class_weight = num_negative / num_positive

    if preprocessing == "basic":
        preprocessor = BasicPreprocessor()
    elif preprocessing == "brfss":
        preprocessor = BRFSSPreprocessor(max_missing_rate=max_missing_rate)
    else:
        raise ValueError("preprocessing must be 'basic' or 'brfss'")
    x_train = preprocessor.fit_transform(
        x[train_indices], feature_names=feature_names
    )
    x_validation = preprocessor.transform(x[validation_indices])

    x_train = add_intercept(x_train)
    x_validation = add_intercept(x_validation)
    initial_w = np.zeros(x_train.shape[1], dtype=np.float64)

    w, objective = linear_svm_sgd(
        y_train,
        x_train,
        lambda_=lambda_,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        seed=seed,
        positive_class_weight=positive_class_weight,
    )
    predictions = predict_linear_svm(x_validation, w)
    svm_metrics = classification_metrics(y_validation, predictions)

    dummy_predictions = np.full(y_validation.shape, -1.0)
    dummy_metrics = classification_metrics(y_validation, dummy_predictions)

    return {
        "weights": w,
        "objective": float(objective),
        "positive_class_weight": float(positive_class_weight),
        "svm_metrics": svm_metrics,
        "dummy_metrics": dummy_metrics,
        "num_training_rows": int(train_indices.size),
        "num_validation_rows": int(validation_indices.size),
        "num_input_features": int(x.shape[1]),
        "num_processed_features": int(x_train.shape[1]),
        "dropped_features": preprocessor.dropped_features_,
        "preprocessing": preprocessing,
    }


def print_metrics(name, metrics):
    """Print classification metrics in a compact, comparable format."""
    print(f"\n{name}")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
    ):
        print(f"  {key:>17}: {metrics[key]:.4f}")
    print(
        "  confusion matrix: "
        f"TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, "
        f"FN={metrics['false_negative']}, "
        f"TP={metrics['true_positive']}"
    )


def main():
    """Load real data, run the SVM experiment and report its performance."""
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    if args.positive_class_weight < 0:
        raise ValueError("positive_class_weight must be non-negative")
    max_rows = None if args.max_rows == 0 else args.max_rows
    positive_class_weight = (
        None if args.positive_class_weight == 0 else args.positive_class_weight
    )

    y, x, _, feature_names = load_training_data(
        args.data_dir, max_rows=max_rows
    )
    results = run_svm_experiment(
        y,
        x,
        feature_names,
        validation_ratio=args.validation_ratio,
        lambda_=args.lambda_,
        gamma=args.gamma,
        max_iters=args.max_iters,
        seed=args.seed,
        positive_class_weight=positive_class_weight,
        preprocessing=args.preprocessing,
        max_missing_rate=args.max_missing_rate,
    )

    print("Linear SVM validation experiment")
    print(f"  data directory: {args.data_dir}")
    print(f"  loaded rows: {y.size}")
    print(f"  training rows: {results['num_training_rows']}")
    print(f"  validation rows: {results['num_validation_rows']}")
    print(
        "  features: "
        f"{results['num_input_features']} raw -> "
        f"{results['num_processed_features']} including intercept"
    )
    print(f"  dropped features: {len(results['dropped_features'])}")
    print(f"  positive class weight: {results['positive_class_weight']:.4f}")
    print(f"  preprocessing: {results['preprocessing']}")
    print(f"  final SVM objective: {results['objective']:.6f}")
    print_metrics("All-negative dummy baseline", results["dummy_metrics"])
    print_metrics("Linear SVM", results["svm_metrics"])


if __name__ == "__main__":
    main()
