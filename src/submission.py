"""Utilities for writing and validating AIcrowd prediction files."""

import csv
from pathlib import Path

import numpy as np


def create_csv_submission(ids, predictions, output_path):
    """Write an AIcrowd submission with columns ``Id`` and ``Prediction``."""
    ids = np.asarray(ids)
    predictions = np.asarray(predictions)
    if ids.ndim != 1 or predictions.ndim != 1:
        raise ValueError("ids and predictions must be one-dimensional")
    if ids.size != predictions.size:
        raise ValueError("ids and predictions must have the same length")
    if ids.size == 0:
        raise ValueError("ids and predictions must not be empty")
    if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
        raise ValueError("ids must contain finite integers")
    if np.unique(ids).size != ids.size:
        raise ValueError("ids must be unique")
    if not np.all((predictions == -1) | (predictions == 1)):
        raise ValueError("predictions must contain only -1 and 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Id", "Prediction"))
        writer.writerows(
            (int(sample_id), int(prediction))
            for sample_id, prediction in zip(ids, predictions)
        )
    return output_path
