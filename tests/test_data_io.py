"""Tests for the Project 1 CSV loader."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_io import (
    load_features,
    load_labels,
    load_test_data,
    load_training_data,
)


class DataIoTests(unittest.TestCase):
    """Verify parsing, missing values, labels and ID alignment."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        (self.data_dir / "x_train.csv").write_text(
            "Id,feature_a,feature_b\n10,1.5,\n11,2.5,3\n",
            encoding="utf-8",
        )
        (self.data_dir / "y_train.csv").write_text(
            "Id,_MICHD\n10,-1\n11,1\n",
            encoding="utf-8",
        )
        (self.data_dir / "x_test.csv").write_text(
            "Id,feature_a,feature_b\n20,4.5,5\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_load_features_preserves_ids_and_missing_values(self):
        x, ids, names = load_features(self.data_dir / "x_train.csv")

        np.testing.assert_array_equal(ids, np.array([10, 11]))
        self.assertEqual(names, ("feature_a", "feature_b"))
        self.assertEqual(x.shape, (2, 2))
        self.assertTrue(np.isnan(x[0, 1]))

    def test_load_labels_can_convert_for_logistic_regression(self):
        y, ids = load_labels(
            self.data_dir / "y_train.csv", logistic_labels=True
        )

        np.testing.assert_array_equal(ids, np.array([10, 11]))
        np.testing.assert_array_equal(y, np.array([0.0, 1.0]))

    def test_training_loader_checks_alignment(self):
        y, x, ids, names = load_training_data(self.data_dir)

        np.testing.assert_array_equal(y, np.array([-1.0, 1.0]))
        np.testing.assert_array_equal(ids, np.array([10, 11]))
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(names, ("feature_a", "feature_b"))

    def test_test_loader_handles_a_single_row(self):
        x, ids, names = load_test_data(self.data_dir)

        np.testing.assert_array_equal(ids, np.array([20]))
        np.testing.assert_array_equal(x, np.array([[4.5, 5.0]]))
        self.assertEqual(names, ("feature_a", "feature_b"))

    def test_training_loader_rejects_misaligned_ids(self):
        (self.data_dir / "y_train.csv").write_text(
            "Id,_MICHD\n10,-1\n12,1\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "IDs are not aligned"):
            load_training_data(self.data_dir)

    def test_label_loader_rejects_values_outside_minus_one_and_one(self):
        (self.data_dir / "y_train.csv").write_text(
            "Id,_MICHD\n10,0\n11,1\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Expected labels"):
            load_labels(self.data_dir / "y_train.csv")


if __name__ == "__main__":
    unittest.main()
