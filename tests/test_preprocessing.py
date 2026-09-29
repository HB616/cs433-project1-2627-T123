"""Tests for leakage-safe splitting and baseline preprocessing."""

import unittest

import numpy as np

from src.preprocessing import BasicPreprocessor, stratified_split_indices


class StratifiedSplitTests(unittest.TestCase):
    """Verify class balance, separation and reproducibility."""

    def test_split_preserves_class_ratio_and_has_no_overlap(self):
        y = np.array([-1] * 90 + [1] * 10)

        train_indices, validation_indices = stratified_split_indices(
            y, validation_ratio=0.2, seed=7
        )

        self.assertEqual(train_indices.size, 80)
        self.assertEqual(validation_indices.size, 20)
        self.assertEqual(np.mean(y[train_indices] == 1), 0.1)
        self.assertEqual(np.mean(y[validation_indices] == 1), 0.1)
        self.assertEqual(
            np.intersect1d(train_indices, validation_indices).size, 0
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_indices, validation_indices])),
            np.arange(y.size),
        )

    def test_split_is_reproducible(self):
        y = np.array([-1] * 20 + [1] * 4)

        first = stratified_split_indices(y, seed=123)
        second = stratified_split_indices(y, seed=123)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class BasicPreprocessorTests(unittest.TestCase):
    """Verify filtering, imputation, scaling and leakage protection."""

    def setUp(self):
        self.x_train = np.array(
            [
                [1.0, np.nan, 5.0, np.nan],
                [2.0, 10.0, 5.0, np.nan],
                [3.0, 20.0, 5.0, np.nan],
                [4.0, 30.0, 5.0, np.nan],
            ]
        )
        self.names = ("numeric", "with_missing", "constant", "all_missing")

    def test_fit_transform_removes_unusable_columns_and_imputes(self):
        preprocessor = BasicPreprocessor()

        transformed = preprocessor.fit_transform(
            self.x_train, feature_names=self.names
        )

        self.assertEqual(transformed.shape, (4, 2))
        self.assertFalse(np.isnan(transformed).any())
        self.assertEqual(
            preprocessor.feature_names_out_, ("numeric", "with_missing")
        )
        self.assertEqual(
            preprocessor.dropped_features_,
            (("constant", "constant"), ("all_missing", "all_missing")),
        )
        np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(transformed, axis=0), 1.0, atol=1e-12)

    def test_transform_reuses_training_statistics(self):
        preprocessor = BasicPreprocessor().fit(
            self.x_train, feature_names=self.names
        )
        validation = np.array([[5.0, np.nan, 999.0, 100.0]])

        transformed = preprocessor.transform(validation)

        self.assertEqual(transformed.shape, (1, 2))
        self.assertAlmostEqual(transformed[0, 1], 0.0)
        self.assertEqual(preprocessor.medians_[1], 20.0)

    def test_transform_before_fit_fails(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            BasicPreprocessor().transform(self.x_train)


if __name__ == "__main__":
    unittest.main()
