"""Reproduce SVM report statistics, CSV tables, and publication-ready figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_svm import DEFAULT_DATA_DIR, run_svm_experiment
from src.data_io import load_training_data


DEFAULT_OUTPUT_DIR = Path("report/svm/assets")
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
)
BLUE = "#35618D"
GOLD = "#D4A72C"
LIGHT_BLUE = "#A9C5DE"
CHARCOAL = "#24313D"
GRID = "#D9DEE3"


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
    parser.add_argument("--positive-class-weight", type=float, default=0.0)
    return parser.parse_args()


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=CHARCOAL)
    axis.title.set_color(CHARCOAL)


def plot_class_distribution(y, output_path):
    labels = ("No MICHD (-1)", "MICHD (+1)")
    counts = np.array([np.sum(y == -1), np.sum(y == 1)])
    percentages = 100.0 * counts / counts.sum()
    figure, axis = plt.subplots(figsize=(6.4, 3.5))
    bars = axis.barh(labels, counts, color=(LIGHT_BLUE, BLUE), height=0.56)
    axis.set_title("Training-label distribution")
    axis.set_xlabel("Number of respondents")
    axis.set_xlim(0, counts.max() * 1.18)
    for bar, count, percentage in zip(bars, counts, percentages):
        axis.text(
            bar.get_width() + counts.max() * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({percentage:.1f}%)",
            va="center",
            color=CHARCOAL,
            fontsize=10,
        )
    _style_axes(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(dummy_metrics, svm_metrics, output_path):
    labels = ("Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "Specificity")
    dummy = [dummy_metrics[key] for key in METRIC_KEYS]
    svm = [svm_metrics[key] for key in METRIC_KEYS]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.0))
    axis.bar(x - width / 2, dummy, width, label="All-negative baseline", color=LIGHT_BLUE)
    axis.bar(x + width / 2, svm, width, label="Weighted linear SVM", color=BLUE)
    axis.set_title("Validation metrics on the complete dataset split")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper center", ncol=2)
    _style_axes(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics, output_path):
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(4.8, 4.1))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Linear SVM validation confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks((0, 1), ("-1", "+1"))
    axis.set_yticks((0, 1), ("-1", "+1"))
    threshold = matrix.max() / 2
    names = (("TN", "FP"), ("FN", "TP"))
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{names[row][column]}\n{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else CHARCOAL,
                fontsize=11,
                fontweight="bold",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Respondents")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def generate_assets(y, results, output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    svm_metrics = results["svm_metrics"]
    dummy_metrics = results["dummy_metrics"]

    negative_count = int(np.sum(y == -1))
    positive_count = int(np.sum(y == 1))
    _write_csv(
        output_dir / "class_distribution.csv",
        ("label", "count", "percentage"),
        (
            (-1, negative_count, negative_count / y.size),
            (1, positive_count, positive_count / y.size),
        ),
    )
    _write_csv(
        output_dir / "validation_metrics.csv",
        ("model",) + METRIC_KEYS,
        (
            ("all_negative_baseline",) + tuple(dummy_metrics[key] for key in METRIC_KEYS),
            ("weighted_linear_svm",) + tuple(svm_metrics[key] for key in METRIC_KEYS),
        ),
    )
    _write_csv(
        output_dir / "confusion_matrix.csv",
        ("model", "tn", "fp", "fn", "tp"),
        (
            (
                "weighted_linear_svm",
                svm_metrics["true_negative"],
                svm_metrics["false_positive"],
                svm_metrics["false_negative"],
                svm_metrics["true_positive"],
            ),
        ),
    )
    serializable = {
        "configuration": config,
        "dataset": {
            "loaded_rows": int(y.size),
            "negative_rows": negative_count,
            "positive_rows": positive_count,
        },
        "experiment": {
            key: value
            for key, value in results.items()
            if key not in {"weights", "svm_metrics", "dummy_metrics", "dropped_features"}
        },
        "svm_metrics": svm_metrics,
        "dummy_metrics": dummy_metrics,
    }
    with (output_dir / "svm_statistics.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, indent=2)

    plot_class_distribution(y, output_dir / "class_distribution.png")
    plot_metric_comparison(
        dummy_metrics, svm_metrics, output_dir / "validation_metrics.png"
    )
    plot_confusion_matrix(svm_metrics, output_dir / "confusion_matrix.png")


def main():
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("max-rows must be non-negative")
    if args.positive_class_weight < 0:
        raise ValueError("positive-class-weight must be non-negative")
    max_rows = None if args.max_rows == 0 else args.max_rows
    class_weight = None if args.positive_class_weight == 0 else args.positive_class_weight
    y, x, _, feature_names = load_training_data(
        args.data_dir, dtype=np.float32, max_rows=max_rows
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
        positive_class_weight=class_weight,
        preprocessing="brfss",
    )
    config = {
        "validation_ratio": args.validation_ratio,
        "lambda": args.lambda_,
        "gamma": args.gamma,
        "max_iters": args.max_iters,
        "seed": args.seed,
        "positive_class_weight": results["positive_class_weight"],
        "preprocessing": "brfss",
    }
    generate_assets(y, results, args.output_dir, config)
    print(f"SVM report assets written to {args.output_dir.resolve()}")
    print(f"Balanced accuracy: {results['svm_metrics']['balanced_accuracy']:.4f}")
    print(f"F1 score: {results['svm_metrics']['f1_score']:.4f}")


if __name__ == "__main__":
    main()
