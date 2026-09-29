"""Tests for the Project 1 data-exploration utilities."""

import unittest

import numpy as np

from src.data_exploration import combine_profiles, profile_features


class DataExplorationTests(unittest.TestCase):
    """Verify missingness, constants and train/test comparison."""

    def test_profile_features_computes_core_statistics(self):
        x = np.array(
            [
                [1.0, np.nan, 7.0],
                [2.0, 3.0, 7.0],
                [3.0, np.nan, 7.0],
                [4.0, 5.0, 7.0],
            ]
        )

        rows = profile_features(x, ("numeric", "sparse", "constant"))

        self.assertEqual(rows[0]["unique_count"], 4)
        self.assertEqual(rows[0]["median"], 2.5)
        self.assertEqual(rows[1]["missing_count"], 2)
        self.assertEqual(rows[1]["missing_rate"], 0.5)
        self.assertIn("missing_ge_50pct", rows[1]["flags"])
        self.assertIn("constant", rows[2]["flags"])

    def test_combine_profiles_flags_large_missingness_gap(self):
        train = profile_features(
            np.array([[1.0], [2.0], [3.0], [4.0]]), ("feature",)
        )
        test = profile_features(
            np.array([[np.nan], [np.nan], [3.0], [4.0]]), ("feature",)
        )

        row = combine_profiles(train, test)[0]

        self.assertEqual(row["missing_rate_gap"], 0.5)
        self.assertIn("missing_gap_ge_10pct", row["flags"])

    def test_rejects_wrong_number_of_feature_names(self):
        with self.assertRaisesRegex(ValueError, "feature_names"):
            profile_features(np.ones((2, 2)), ("only_one",))


if __name__ == "__main__":
    unittest.main()
