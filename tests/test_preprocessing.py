"""Tests for leakage-safe splitting and baseline preprocessing."""

import unittest

import numpy as np

from src.preprocessing import (
    BRFSSPreprocessor,
    BasicPreprocessor,
    stratified_split_indices,
)


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


class BRFSSPreprocessorTests(unittest.TestCase):
    """Verify codebook sentinels, one-hot encoding and leakage protection."""

    def setUp(self):
        self.names = ("PHYSHLTH", "GENHLTH", "SEX")
        self.x_train = np.array(
            [
                [0.0, 1.0, 1.0],
                [5.0, 2.0, 2.0],
                [77.0, 7.0, 1.0],
                [99.0, 9.0, 2.0],
            ]
        )

    def test_sentinels_become_missing_and_categories_are_one_hot(self):
        preprocessor = BRFSSPreprocessor()

        transformed = preprocessor.fit_transform(self.x_train, self.names)

        self.assertEqual(transformed.shape, (4, 7))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertEqual(
            preprocessor.feature_names_out_,
            (
                "PHYSHLTH",
                "PHYSHLTH==MISSING",
                "GENHLTH==1",
                "GENHLTH==2",
                "GENHLTH==MISSING",
                "SEX==1",
                "SEX==2",
            ),
        )
        np.testing.assert_array_equal(transformed[:, 1], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(transformed[:, 2:5].sum(axis=1), 1.0)
        np.testing.assert_array_equal(transformed[:, 5:7].sum(axis=1), 1.0)
        self.assertFalse(np.isnan(transformed).any())

    def test_transform_reuses_categories_and_handles_unseen_values(self):
        preprocessor = BRFSSPreprocessor().fit(self.x_train, self.names)
        validation = np.array(
            [
                [77.0, 3.0, 2.0],
                [10.0, 7.0, 3.0],
            ]
        )

        transformed = preprocessor.transform(validation)

        # GENHLTH=3 and SEX=3 were unseen during fit and therefore produce
        # all-zero category blocks instead of changing the learned shape.
        np.testing.assert_array_equal(transformed[0, 2:5], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(transformed[1, 2:5], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(transformed[1, 5:7], [0.0, 0.0])
        self.assertEqual(transformed.shape, (2, 7))

    def test_calculated_missingness_feature_is_kept_as_category(self):
        x = np.array([[0.0], [1.0], [9.0]])
        preprocessor = BRFSSPreprocessor()

        transformed = preprocessor.fit_transform(x, ("PAMISS1_",))

        self.assertEqual(transformed.shape, (3, 3))
        self.assertNotIn("PAMISS1_==MISSING", preprocessor.feature_names_out_)
        np.testing.assert_array_equal(transformed.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
