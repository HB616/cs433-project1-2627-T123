"""Tests for the end-to-end SVM validation experiment."""

import unittest

import numpy as np

from evaluate_svm import add_intercept, run_svm_experiment


class SvmPipelineTests(unittest.TestCase):
    """Exercise splitting, preprocessing, training and evaluation together."""

    def test_add_intercept(self):
        x = np.array([[2.0, 3.0], [4.0, 5.0]])

        result = add_intercept(x)

        np.testing.assert_array_equal(
            result, np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
        )

    def test_pipeline_learns_separable_data_with_missing_values(self):
        negative = np.column_stack(
            (np.linspace(-3.0, -1.0, 20), np.ones(20))
        )
        positive = np.column_stack(
            (np.linspace(1.0, 3.0, 20), np.ones(20))
        )
        x = np.vstack((negative, positive))
        x[0, 0] = np.nan
        y = np.concatenate((-np.ones(20), np.ones(20)))

        results = run_svm_experiment(
            y,
            x,
            feature_names=("signal", "constant"),
            lambda_=0.01,
            gamma=0.05,
            max_iters=1000,
            seed=4,
        )

        self.assertEqual(results["num_training_rows"], 32)
        self.assertEqual(results["num_validation_rows"], 8)
        self.assertEqual(results["num_processed_features"], 2)
        self.assertEqual(results["dropped_features"], (("constant", "constant"),))
        self.assertGreaterEqual(results["svm_metrics"]["accuracy"], 0.8)
        self.assertGreater(
            results["svm_metrics"]["balanced_accuracy"],
            results["dummy_metrics"]["balanced_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
