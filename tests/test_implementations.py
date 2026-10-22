"""Tests for the implemented Project 1 regression methods."""

import unittest

import numpy as np

from implementations import least_squares, ridge_regression


class LeastSquaresTests(unittest.TestCase):
    """Verify exact fitting and rank-deficient behavior."""

    def test_recovers_exact_linear_relationship(self):
        tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y = np.array([1.0, 3.0, 5.0])

        w, loss = least_squares(y, tx)

        np.testing.assert_allclose(w, np.array([1.0, 2.0]))
        self.assertAlmostEqual(float(loss), 0.0)
        self.assertEqual(w.shape, (2,))
        self.assertEqual(loss.ndim, 0)

    def test_rank_deficient_matrix_returns_a_finite_solution(self):
        tx = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])

        w, loss = least_squares(y, tx)

        np.testing.assert_allclose(tx @ w, y, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(loss), 0.0)


class RidgeRegressionTests(unittest.TestCase):
    """Verify regularization scaling and returned loss semantics."""

    def setUp(self):
        self.tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.y = np.array([1.0, 3.0, 5.0])

    def test_zero_lambda_matches_least_squares(self):
        least_squares_w, least_squares_loss = least_squares(self.y, self.tx)

        ridge_w, ridge_loss = ridge_regression(self.y, self.tx, 0.0)

        np.testing.assert_allclose(ridge_w, least_squares_w)
        np.testing.assert_allclose(ridge_loss, least_squares_loss)

    def test_positive_lambda_shrinks_weight_norm(self):
        least_squares_w, _ = least_squares(self.y, self.tx)

        ridge_w, ridge_loss = ridge_regression(self.y, self.tx, 1.0)

        self.assertLess(np.linalg.norm(ridge_w), np.linalg.norm(least_squares_w))
        residuals = self.y - self.tx @ ridge_w
        expected_mse = np.sum(residuals**2) / (2.0 * self.y.size)
        self.assertAlmostEqual(float(ridge_loss), expected_mse)

    def test_negative_lambda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ridge_regression(self.y, self.tx, -0.1)


if __name__ == "__main__":
    unittest.main()
