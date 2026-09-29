"""Binary-classification metrics implemented with NumPy only."""

import numpy as np


def _validate_binary_labels(y_true, y_pred, negative_label, positive_label):
    """Validate label vectors and return one-dimensional NumPy arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if negative_label == positive_label:
        raise ValueError("negative_label and positive_label must be different")

    allowed_labels = np.array([negative_label, positive_label])
    if not np.all(np.isin(y_true, allowed_labels)):
        raise ValueError("y_true contains labels outside the expected binary labels")
    if not np.all(np.isin(y_pred, allowed_labels)):
        raise ValueError("y_pred contains labels outside the expected binary labels")
    return y_true, y_pred


def binary_confusion_matrix(
    y_true, y_pred, negative_label=-1, positive_label=1
):
    """Return true-negative, false-positive, false-negative and true-positive counts."""
    y_true, y_pred = _validate_binary_labels(
        y_true, y_pred, negative_label, positive_label
    )

    true_negative = np.sum(
        (y_true == negative_label) & (y_pred == negative_label)
    )
    false_positive = np.sum(
        (y_true == negative_label) & (y_pred == positive_label)
    )
    false_negative = np.sum(
        (y_true == positive_label) & (y_pred == negative_label)
    )
    true_positive = np.sum(
        (y_true == positive_label) & (y_pred == positive_label)
    )

    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def _safe_divide(numerator, denominator):
    """Return zero when a classification metric has a zero denominator."""
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    y_true, y_pred, negative_label=-1, positive_label=1
):
    """Compute the core metrics for the Project 1 binary classifier.

    Returns counts together with accuracy, precision, recall (sensitivity),
    specificity, F1-score and balanced accuracy. A metric with an undefined
    zero denominator is reported as 0.0.
    """
    counts = binary_confusion_matrix(
        y_true,
        y_pred,
        negative_label=negative_label,
        positive_label=positive_label,
    )
    true_negative = counts["true_negative"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    true_positive = counts["true_positive"]

    total = true_negative + false_positive + false_negative + true_positive
    accuracy = _safe_divide(true_positive + true_negative, total)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    f1_score = _safe_divide(2.0 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy,
    }
