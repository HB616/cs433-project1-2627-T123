"""Tests for the linear soft-margin SVM implementation."""

import unittest

import numpy as np

from implementations import linear_svm_sgd, predict_linear_svm


class LinearSvmTests(unittest.TestCase):
    """Verify learning behavior, reproducibility, and input validation."""

    def setUp(self):
        # The first column acts as the intercept.
        self.tx = np.array(
            [
                [1.0, -2.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [1.0, 2.0],
            ]
        )
        self.y = np.array([-1.0, -1.0, 1.0, 1.0])
        self.initial_w = np.zeros(self.tx.shape[1])

    def test_learns_a_linearly_separable_dataset(self):
        w, loss = linear_svm_sgd(
            self.y,
            self.tx,
            lambda_=0.01,
            initial_w=self.initial_w,
            max_iters=1000,
            gamma=0.05,
            seed=7,
        )

        predictions = predict_linear_svm(self.tx, w)

        np.testing.assert_array_equal(predictions, self.y)
        self.assertEqual(w.shape, (2,))
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_same_seed_produces_same_result(self):
        arguments = dict(
            lambda_=0.1,
            initial_w=self.initial_w,
            max_iters=100,
            gamma=0.02,
            seed=19,
        )

        first_w, first_loss = linear_svm_sgd(self.y, self.tx, **arguments)
        second_w, second_loss = linear_svm_sgd(self.y, self.tx, **arguments)

        np.testing.assert_array_equal(first_w, second_w)
        self.assertEqual(first_loss, second_loss)

    def test_zero_iterations_returns_initial_model_objective(self):
        w, loss = linear_svm_sgd(
            self.y,
            self.tx,
            lambda_=0.5,
            initial_w=self.initial_w,
            max_iters=0,
            gamma=0.1,
        )

        np.testing.assert_array_equal(w, self.initial_w)
        self.assertAlmostEqual(float(loss), 1.0)

    def test_rejects_zero_one_labels(self):
        with self.assertRaisesRegex(ValueError, "only -1 and 1"):
            linear_svm_sgd(
                np.array([0.0, 0.0, 1.0, 1.0]),
                self.tx,
                lambda_=0.1,
                initial_w=self.initial_w,
                max_iters=10,
                gamma=0.1,
            )

    def test_prediction_uses_positive_class_for_zero_score(self):
        predictions = predict_linear_svm(self.tx, np.zeros(2))

        np.testing.assert_array_equal(predictions, np.ones(self.y.size))

    def test_rejects_non_positive_class_weight(self):
        with self.assertRaisesRegex(ValueError, "finite positive scalar"):
            linear_svm_sgd(
                self.y,
                self.tx,
                lambda_=0.1,
                initial_w=self.initial_w,
                max_iters=10,
                gamma=0.1,
                positive_class_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
