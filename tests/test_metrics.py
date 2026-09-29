"""Tests for binary classification metrics."""

import unittest

import numpy as np

from src.metrics import binary_confusion_matrix, classification_metrics


class BinaryMetricsTests(unittest.TestCase):
    """Verify counts and metrics, including imbalanced edge cases."""

    def test_confusion_matrix_counts_each_outcome(self):
        y_true = np.array([1, 1, -1, -1, 1, -1])
        y_pred = np.array([1, -1, 1, -1, 1, -1])

        counts = binary_confusion_matrix(y_true, y_pred)

        self.assertEqual(
            counts,
            {
                "true_negative": 2,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 2,
            },
        )

    def test_classification_metrics_match_known_values(self):
        y_true = np.array([1, 1, -1, -1, 1, -1])
        y_pred = np.array([1, -1, 1, -1, 1, -1])

        metrics = classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_score"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)

    def test_all_negative_prediction_exposes_imbalanced_baseline(self):
        y_true = np.array([-1] * 90 + [1] * 10)
        y_pred = np.full(y_true.shape, -1)

        metrics = classification_metrics(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 0.9)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_supports_zero_one_labels_when_requested(self):
        metrics = classification_metrics(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 0]),
            negative_label=0,
            positive_label=1,
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 2)

    def test_rejects_mismatched_shapes_and_unexpected_labels(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            classification_metrics(np.array([-1, 1]), np.array([-1]))

        with self.assertRaisesRegex(ValueError, "outside"):
            classification_metrics(np.array([-1, 0, 1]), np.array([-1, -1, 1]))


if __name__ == "__main__":
    unittest.main()
