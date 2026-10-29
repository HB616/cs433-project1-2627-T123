import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data_io import load_test_data, load_training_data
from src.metrics import classification_metrics
from src.modeling import (
    fit_balanced_logistic_regression,
    predict_probabilities,
    select_accuracy_threshold,
    select_balanced_accuracy_threshold,
    select_f1_threshold,
)
from src.preprocessing import BasicPreprocessor, stratified_split_indices


def parse_args():
    "Parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Directory containing x_train.csv, y_train.csv and x_test.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission.csv"),
        help="Path for the generated AIcrowd-format prediction file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--threshold-objective",
        choices=("accuracy", "balanced_accuracy", "f1"),
        default="accuracy",
        help="Validation metric used to select the test prediction threshold.",
    )
    return parser.parse_args()


def _print_metrics(title, metrics):
    "Print the metrics that matter for an imbalanced target
    print(
        f"{title}: accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"specificity={metrics['specificity']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )


def _write_predictions(output_path, ids, predictions):
    "Write predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Id", "Prediction"))
        writer.writerows(zip(ids.tolist(), predictions.tolist()))


def main():
    "Run a reproducible stratified validation and test
    args = parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")

    y, x, _, feature_names = load_training_data(
        args.data_dir, dtype=np.float32, logistic_labels=False
    )
    train_idx, validation_idx = stratified_split_indices(
        y, validation_ratio=args.validation_ratio, seed=args.seed
    )
    y_train_binary = (y[train_idx] == 1.0).astype(np.float64)
    y_validation = (y[validation_idx] == 1.0).astype(np.int64)

    preprocessor = BasicPreprocessor(dtype=np.float32)
    preprocessor.fit(x[train_idx], feature_names=feature_names)
    x_fit = preprocessor.transform(x[train_idx])
    x_validation = preprocessor.transform(x[validation_idx])
    del x

    weights, intercept, balanced_counts = fit_balanced_logistic_regression(
        y_train_binary,
        x_fit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        seed=args.seed,
    )
    validation_probabilities = predict_probabilities(
        x_validation, weights, intercept
    )
    accuracy_threshold, accuracy_threshold_metrics = select_accuracy_threshold(
        y_validation, validation_probabilities
    )
    balanced_threshold, balanced_threshold_metrics = (
        select_balanced_accuracy_threshold(y_validation, validation_probabilities)
    )
    f1_threshold, f1_threshold_metrics = select_f1_threshold(
        y_validation, validation_probabilities
    )
    if args.threshold_objective == "accuracy":
        threshold = accuracy_threshold
        threshold_metrics = accuracy_threshold_metrics
    elif args.threshold_objective == "balanced_accuracy":
        threshold = balanced_threshold
        threshold_metrics = balanced_threshold_metrics
    else:
        threshold = f1_threshold
        threshold_metrics = f1_threshold_metrics
    validation_predictions = (validation_probabilities >= threshold).astype(np.int64)
    validation_metrics = classification_metrics(
        y_validation,
        validation_predictions,
        negative_label=0,
        positive_label=1,
    )
    majority_baseline = classification_metrics(
        y_validation,
        np.zeros_like(y_validation),
        negative_label=0,
        positive_label=1,
    )

    class_values, class_counts = np.unique(y_train_binary, return_counts=True)
    print(
        f"Training rows: {y_train_binary.size}; validation rows: "
        f"{y_validation.size}; features retained: {x_fit.shape[1]}"
    )
    print(
        "Training class counts: "
        + ", ".join(
            f"{int(value)}={int(count)}" for value, count in zip(class_values, class_counts)
        )
    )
    print(
        f"Balanced training sample: negative={balanced_counts[0]}, "
        f"positive={balanced_counts[1]}"
    )
    print(
        f"Submission threshold selected for {args.threshold_objective}: "
        f"{threshold:.3f}"
    )
    print(
        "Balanced-accuracy reference threshold: "
        f"{balanced_threshold:.3f} "
        f"(balanced accuracy={balanced_threshold_metrics['balanced_accuracy']:.4f}, "
        f"recall={balanced_threshold_metrics['recall']:.4f})"
    )
    print(
        f"F1 reference threshold: {f1_threshold:.6f} "
        f"(F1={f1_threshold_metrics['f1_score']:.4f}, "
        f"precision={f1_threshold_metrics['precision']:.4f}, "
        f"recall={f1_threshold_metrics['recall']:.4f})"
    )
    _print_metrics("Majority-class baseline", majority_baseline)
    _print_metrics("Model at submission threshold", validation_metrics)

    x_test, test_ids, test_feature_names = load_test_data(
        args.data_dir, dtype=np.float32
    )
    if test_feature_names != feature_names:
        raise ValueError("Training and test feature columns are not aligned")
    x_test_processed = preprocessor.transform(x_test)
    test_probabilities = predict_probabilities(x_test_processed, weights, intercept)
    predictions = np.where(test_probabilities >= threshold, 1, -1).astype(np.int8)
    _write_predictions(args.output, test_ids, predictions)

    metrics_path = Path(args.output).with_name(
        Path(args.output).stem + "_validation.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "submission_threshold_objective": args.threshold_objective,
        "selected_threshold": threshold,
        "balanced_accuracy_reference_threshold": balanced_threshold,
        "f1_reference_threshold": f1_threshold,
        "feature_count_input": len(feature_names),
        "feature_count_retained": int(x_fit.shape[1]),
        "dropped_features": [
            {"name": name, "reason": reason}
            for name, reason in preprocessor.dropped_features_
        ],
        "training_class_counts": {
            str(int(value)): int(count)
            for value, count in zip(class_values, class_counts)
        },
        "balanced_training_counts": {
            "negative": balanced_counts[0],
            "positive": balanced_counts[1],
        },
        "majority_baseline": majority_baseline,
        "validation_metrics": validation_metrics,
        "threshold_selection_metrics": threshold_metrics,
        "balanced_accuracy_reference_metrics": balanced_threshold_metrics,
        "f1_reference_metrics": f1_threshold_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Predictions written to: {args.output}")
    print(f"Validation report written to: {metrics_path}")


if __name__ == "__main__":
    main()
