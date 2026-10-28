"""Generate a real-data SVM decision-threshold analysis for the report."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_svm import DEFAULT_DATA_DIR, add_intercept
from implementations import linear_svm_sgd
from src.data_io import load_training_data
from src.metrics import classification_metrics
from src.preprocessing import BRFSSPreprocessor, stratified_split_indices


DEFAULT_OUTPUT_DIR = Path("report/svm/assets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.0001)
    parser.add_argument("--max-iters", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=201,
        help="Number of score quantiles used for the threshold sweep.",
    )
    return parser.parse_args()


def evaluate_thresholds(y_true, scores, thresholds):
    """Return metric dictionaries for predictions at every score threshold."""
    rows = []
    for threshold in thresholds:
        predictions = np.where(scores >= threshold, 1, -1)
        metrics = classification_metrics(y_true, predictions)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def select_f1_threshold(rows):
    """Select by F1, then balanced accuracy, then closeness to zero."""
    return max(
        rows,
        key=lambda row: (
            row["f1_score"],
            row["balanced_accuracy"],
            -abs(row["threshold"]),
        ),
    )


def save_results(rows, selected, output_dir, configuration):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "svm_threshold_results.csv"
    fields = (
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    summary = {
        "configuration": configuration,
        "selected_by": "maximum positive-class F1; balanced accuracy tie-break",
        "selected_threshold": selected,
        "num_thresholds_evaluated": len(rows),
    }
    with (output_dir / "svm_threshold_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)


def plot_threshold_results(rows, selected, output_path):
    thresholds = np.array([row["threshold"] for row in rows])
    accuracy = np.array([row["accuracy"] for row in rows])
    f1 = np.array([row["f1_score"] for row in rows])
    precision = np.array([row["precision"] for row in rows])
    recall = np.array([row["recall"] for row in rows])

    # Retain every evaluated threshold in the CSV, but exclude only extreme
    # score endpoints from the visual range so the useful trade-off is legible.
    lower = float(np.quantile(thresholds, 0.01))
    upper = float(np.quantile(thresholds, 0.995))
    visible = (thresholds >= lower) & (thresholds <= upper)
    plot_thresholds = thresholds[visible]

    blue = "#2C6E9F"
    orange = "#D97706"
    charcoal = "#30343B"
    grid = "#C9CDD2"

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.25), sharey=True)
    panels = (
        (axes[0], accuracy[visible], f1[visible], "Accuracy", "Positive-class F1", "(a) Accuracy and F1"),
        (axes[1], precision[visible], recall[visible], "Precision", "Recall", "(b) Precision and recall"),
    )
    for axis, first, second, first_label, second_label, title in panels:
        axis.plot(plot_thresholds, first, color=blue, linewidth=2.1, label=first_label)
        axis.plot(plot_thresholds, second, color=orange, linewidth=2.1, label=second_label)
        axis.axvline(
            selected["threshold"],
            color="#666666",
            linestyle="--",
            linewidth=1.5,
            label="F1-selected threshold",
        )
        selected_first = selected["accuracy"] if axis is axes[0] else selected["precision"]
        selected_second = selected["f1_score"] if axis is axes[0] else selected["recall"]
        axis.scatter(
            [selected["threshold"]], [selected_first], color=blue,
            marker="o", s=42, zorder=3,
        )
        axis.scatter(
            [selected["threshold"]], [selected_second], color=orange,
            marker="s", s=42, zorder=3,
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("SVM decision-score threshold")
        axis.set_xlim(lower, upper)
        axis.set_ylim(0.0, 1.02)
        axis.grid(color=grid, linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#DDDDDD",
            framealpha=0.92,
            loc="best",
        )

    axes[0].set_ylabel("Validation score")
    axes[0].annotate(
        f"F1 = {selected['f1_score']:.3f}\nthreshold = {selected['threshold']:.3f}",
        xy=(selected["threshold"], selected["f1_score"]),
        xytext=(10, 22),
        textcoords="offset points",
        color=charcoal,
        fontsize=9.5,
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8},
    )
    figure.subplots_adjust(bottom=0.16, wspace=0.18)
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("max-rows must be non-negative")
    if args.num_thresholds < 3:
        raise ValueError("num-thresholds must be at least 3")
    max_rows = None if args.max_rows == 0 else args.max_rows

    y, x, _, feature_names = load_training_data(
        args.data_dir, dtype=np.float32, max_rows=max_rows
    )
    train_indices, validation_indices = stratified_split_indices(
        y, validation_ratio=args.validation_ratio, seed=args.seed
    )
    y_train = y[train_indices]
    y_validation = y[validation_indices]
    positive_class_weight = float(
        np.sum(y_train == -1) / np.sum(y_train == 1)
    )

    preprocessor = BRFSSPreprocessor()
    x_train = add_intercept(
        preprocessor.fit_transform(x[train_indices], feature_names)
    )
    x_validation = add_intercept(preprocessor.transform(x[validation_indices]))
    weights, objective = linear_svm_sgd(
        y_train,
        x_train,
        lambda_=args.lambda_,
        initial_w=np.zeros(x_train.shape[1], dtype=np.float64),
        max_iters=args.max_iters,
        gamma=args.gamma,
        seed=args.seed,
        positive_class_weight=positive_class_weight,
    )
    scores = x_validation @ weights

    quantiles = np.linspace(0.0, 1.0, args.num_thresholds)
    thresholds = np.unique(np.concatenate((np.quantile(scores, quantiles), [0.0])))
    rows = evaluate_thresholds(y_validation, scores, thresholds)
    selected = select_f1_threshold(rows)
    configuration = {
        "loaded_rows": int(y.size),
        "training_rows": int(train_indices.size),
        "validation_rows": int(validation_indices.size),
        "processed_features_including_intercept": int(x_train.shape[1]),
        "validation_ratio": args.validation_ratio,
        "lambda": args.lambda_,
        "gamma": args.gamma,
        "max_iters": args.max_iters,
        "seed": args.seed,
        "positive_class_weight": positive_class_weight,
        "objective_at_training_threshold": float(objective),
    }
    save_results(rows, selected, args.output_dir, configuration)
    plot_threshold_results(
        rows, selected, args.output_dir / "svm_threshold_results.png"
    )
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Selected threshold: {selected['threshold']:.6f}")
    print(f"Accuracy: {selected['accuracy']:.6f}")
    print(f"Balanced accuracy: {selected['balanced_accuracy']:.6f}")
    print(f"Precision: {selected['precision']:.6f}")
    print(f"Recall: {selected['recall']:.6f}")
    print(f"F1: {selected['f1_score']:.6f}")


if __name__ == "__main__":
    main()
