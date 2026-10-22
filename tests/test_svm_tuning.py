"""Tests for SVM hyperparameter-search utilities."""

import unittest

import numpy as np

from tune_svm import run_grid_search


class SvmTuningTests(unittest.TestCase):
    """Verify that grid results are complete and correctly ranked."""

    def test_grid_search_returns_every_combination_in_ranked_order(self):
        x_train = np.array(
            [[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]]
        )
        y_train = np.array([-1.0, -1.0, 1.0, 1.0])

        results = run_grid_search(
            x_train,
            y_train,
            x_train,
            y_train,
            lambdas=(0.01, 0.1),
            class_weight_multipliers=(0.5, 1.0),
            gammas=(0.05,),
            iteration_values=(100,),
            seed=3,
        )

        self.assertEqual(len(results), 4)
        scores = [result["balanced_accuracy"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all("f1_score" in result for result in results))


if __name__ == "__main__":
    unittest.main()
