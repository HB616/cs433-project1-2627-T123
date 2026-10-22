"""Run a reproducible grid search for the NumPy-only linear SVM."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from evaluate_svm import DEFAULT_DATA_DIR, add_intercept
from implementations import linear_svm_sgd, predict_linear_svm
from src.data_io import load_training_data
from src.metrics import classification_metrics
from src.preprocessing import (
    BRFSSPreprocessor,
    BasicPreprocessor,
    stratified_split_indices,
)


def comma_separated_floats(value):
    """Parse a comma-separated command-line value into floats."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def comma_separated_ints(value):
    """Parse a comma-separated command-line value into integers."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_args():
    """Parse grid-search settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--lambdas",
        type=comma_separated_floats,
        default=(0.001, 0.01, 0.1),
    )
    parser.add_argument(
        "--class-weight-multipliers",
        type=comma_separated_floats,
        default=(0.5, 0.75, 1.0, 1.25),
        help="Multipliers applied to the automatic negative/positive ratio.",
    )
    parser.add_argument(
        "--gammas",
        type=comma_separated_floats,
        default=(0.001,),
    )
    parser.add_argument(
        "--iteration-values",
        type=comma_separated_ints,
        default=(100_000,),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocessing", choices=("basic", "brfss"), default="basic"
    )
    parser.add_argument("--max-missing-rate", type=float, default=1.0)
    return parser.parse_args()


def prepare_validation_data(
    y,
    x,
    feature_names,
    validation_ratio,
    seed,
    preprocessing="basic",
    max_missing_rate=1.0,
):
    """Create one leakage-safe split and transformation for every grid point."""
    train_indices, validation_indices = stratified_split_indices(
        y, validation_ratio=validation_ratio, seed=seed
    )
    y_train = y[train_indices]
    y_validation = y[validation_indices]

    if preprocessing == "basic":
        preprocessor = BasicPreprocessor()
    elif preprocessing == "brfss":
        preprocessor = BRFSSPreprocessor(max_missing_rate=max_missing_rate)
    else:
        raise ValueError("preprocessing must be 'basic' or 'brfss'")
    x_train = add_intercept(
        preprocessor.fit_transform(x[train_indices], feature_names=feature_names)
    )
    x_validation = add_intercept(preprocessor.transform(x[validation_indices]))
    return x_train, y_train, x_validation, y_validation, preprocessor


def run_grid_search(
    x_train,
    y_train,
    x_validation,
    y_validation,
    lambdas,
    class_weight_multipliers,
    gammas,
    iteration_values,
    seed,
):
    """Train every hyperparameter combination and return ranked results."""
    num_negative = np.sum(y_train == -1)
    num_positive = np.sum(y_train == 1)
    if num_positive == 0:
        raise ValueError("The training partition contains no positive samples")
    automatic_class_weight = num_negative / num_positive
    initial_w = np.zeros(x_train.shape[1], dtype=np.float64)
    results = []

    combinations = itertools.product(
        lambdas, class_weight_multipliers, gammas, iteration_values
    )
    for lambda_, multiplier, gamma, max_iters in combinations:
        positive_class_weight = automatic_class_weight * multiplier
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
        metrics = classification_metrics(y_validation, predictions)
        results.append(
            {
                "lambda": lambda_,
                "class_weight_multiplier": multiplier,
                "positive_class_weight": positive_class_weight,
                "gamma": gamma,
                "max_iters": max_iters,
                "objective": float(objective),
                **metrics,
            }
        )

    return sorted(
        results,
        key=lambda result: (
            result["balanced_accuracy"],
            result["f1_score"],
            result["recall"],
        ),
        reverse=True,
    )


def print_results(results):
    """Print ranked validation results as a compact table."""
    header = (
        "rank  lambda    cw_mult  class_w  gamma     iters    "
        "bal_acc  f1      recall  precision  accuracy"
    )
    print(header)
    print("-" * len(header))
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>4}  "
            f"{result['lambda']:<9.4g} "
            f"{result['class_weight_multiplier']:<8.3g} "
            f"{result['positive_class_weight']:<8.3f} "
            f"{result['gamma']:<9.4g} "
            f"{result['max_iters']:<8d} "
            f"{result['balanced_accuracy']:<8.4f} "
            f"{result['f1_score']:<7.4f} "
            f"{result['recall']:<7.4f} "
            f"{result['precision']:<10.4f} "
            f"{result['accuracy']:.4f}"
        )


def main():
    """Load the dataset once and run all requested SVM experiments."""
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    max_rows = None if args.max_rows == 0 else args.max_rows

    y, x, _, feature_names = load_training_data(
        args.data_dir, max_rows=max_rows
    )
    prepared = prepare_validation_data(
        y,
        x,
        feature_names,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        preprocessing=args.preprocessing,
        max_missing_rate=args.max_missing_rate,
    )
    x_train, y_train, x_validation, y_validation, preprocessor = prepared
    results = run_grid_search(
        x_train,
        y_train,
        x_validation,
        y_validation,
        lambdas=args.lambdas,
        class_weight_multipliers=args.class_weight_multipliers,
        gammas=args.gammas,
        iteration_values=args.iteration_values,
        seed=args.seed,
    )

    print("Linear SVM hyperparameter search")
    print(f"  loaded rows: {y.size}")
    print(f"  training / validation: {y_train.size} / {y_validation.size}")
    print(f"  processed features including intercept: {x_train.shape[1]}")
    print(f"  dropped features: {len(preprocessor.dropped_features_)}")
    print(f"  preprocessing: {args.preprocessing}")
    print(f"  combinations: {len(results)}\n")
    print_results(results)


if __name__ == "__main__":
    main()
