"""Tests for AIcrowd submission generation."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.submission import create_csv_submission


class SubmissionTests(unittest.TestCase):
    def test_writes_expected_header_ids_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "submission.csv"
            create_csv_submission(np.array([10, 11]), np.array([-1, 1]), output)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(
                rows, [["Id", "Prediction"], ["10", "-1"], ["11", "1"]]
            )

    def test_rejects_invalid_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only -1 and 1"):
                create_csv_submission(
                    np.array([1, 2]), np.array([0, 1]),
                    Path(directory) / "submission.csv",
                )

    def test_rejects_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "same length"):
                create_csv_submission(
                    np.array([1]), np.array([-1, 1]),
                    Path(directory) / "submission.csv",
                )


if __name__ == "__main__":
    unittest.main()
