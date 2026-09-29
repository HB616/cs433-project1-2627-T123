"""CSV loading utilities for CS-433 Project 1.

Only the Python standard library and NumPy are used. Empty CSV fields are
represented as ``np.nan``; deciding how to impute them belongs to the later
preprocessing stage.
"""

from pathlib import Path

import numpy as np


FEATURE_FILENAMES = ("x_train.csv", "x_test.csv")
LABEL_FILENAME = "y_train.csv"


def _read_header(csv_path):
    """Return a CSV header as a list of column names."""
    with csv_path.open("r", encoding="utf-8-sig") as stream:
        header_line = stream.readline().strip()

    if not header_line:
        raise ValueError(f"CSV file has no header: {csv_path}")

    header = header_line.split(",")
    if len(header) != len(set(header)):
        raise ValueError(f"CSV header contains duplicate column names: {csv_path}")
    return header


def _load_numeric_csv(csv_path, dtype, max_rows):
    """Load the numeric body of a CSV file, mapping empty fields to NaN."""
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        dtype=dtype,
        missing_values="",
        filling_values=np.nan,
        max_rows=max_rows,
    )

    if data.size == 0:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _validate_ids(raw_ids, csv_path):
    """Validate and convert numeric IDs to a one-dimensional integer array."""
    if not np.all(np.isfinite(raw_ids)):
        raise ValueError(f"IDs contain missing or non-finite values: {csv_path}")
    if not np.all(raw_ids == np.floor(raw_ids)):
        raise ValueError(f"IDs contain non-integer values: {csv_path}")

    ids = raw_ids.astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError(f"IDs are not unique: {csv_path}")
    return ids


def load_features(csv_path, dtype=np.float64, max_rows=None):
    """Load one feature CSV.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to ``x_train.csv`` or ``x_test.csv``.
    dtype : numpy dtype, optional
        Floating-point dtype used for the feature matrix.
    max_rows : int or None, optional
        Load only the first rows for quick inspection and tests. ``None`` loads
        the complete file.

    Returns
    -------
    x : numpy.ndarray
        Feature matrix with shape ``(num_samples, num_features)``. Empty fields
        are represented by ``np.nan``.
    ids : numpy.ndarray
        One-dimensional integer sample IDs.
    feature_names : tuple[str, ...]
        Feature names in the same order as the columns of ``x``.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or None")

    header = _read_header(csv_path)
    if len(header) < 2 or header[0] != "Id":
        raise ValueError(f"Expected first column to be 'Id': {csv_path}")

    data = _load_numeric_csv(csv_path, dtype=dtype, max_rows=max_rows)
    if data.shape[1] != len(header):
        raise ValueError(
            f"Header has {len(header)} columns but data has {data.shape[1]}: "
            f"{csv_path}"
        )

    ids = _validate_ids(data[:, 0], csv_path)
    x = data[:, 1:]
    feature_names = tuple(header[1:])
    return x, ids, feature_names


def load_labels(csv_path, logistic_labels=False, max_rows=None):
    """Load labels and their sample IDs from ``y_train.csv``.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to ``y_train.csv``.
    logistic_labels : bool, optional
        Convert labels from ``{-1, 1}`` to ``{0, 1}``, as required by the
        logistic-regression functions in the project specification.
    max_rows : int or None, optional
        Load only the first rows for quick inspection and tests.

    Returns
    -------
    y : numpy.ndarray
        One-dimensional label vector.
    ids : numpy.ndarray
        One-dimensional integer sample IDs.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or None")

    header = _read_header(csv_path)
    if header != ["Id", "_MICHD"]:
        raise ValueError(
            f"Expected label columns ['Id', '_MICHD'], got {header}: {csv_path}"
        )

    data = _load_numeric_csv(csv_path, dtype=np.float64, max_rows=max_rows)
    if data.shape[1] != 2:
        raise ValueError(f"Expected two label columns, got {data.shape[1]}")

    ids = _validate_ids(data[:, 0], csv_path)
    y = data[:, 1]
    label_values = set(np.unique(y).tolist())
    if not label_values.issubset({-1.0, 1.0}):
        raise ValueError(f"Expected labels in {{-1, 1}}, got {label_values}")

    if logistic_labels:
        y = (y == 1.0).astype(np.float64)
    return y, ids


def load_training_data(
    data_dir, dtype=np.float64, logistic_labels=False, max_rows=None
):
    """Load aligned training features, labels, IDs and feature names."""
    data_dir = Path(data_dir)
    x, feature_ids, feature_names = load_features(
        data_dir / "x_train.csv", dtype=dtype, max_rows=max_rows
    )
    y, label_ids = load_labels(
        data_dir / LABEL_FILENAME,
        logistic_labels=logistic_labels,
        max_rows=max_rows,
    )

    if not np.array_equal(feature_ids, label_ids):
        raise ValueError("x_train.csv and y_train.csv IDs are not aligned")
    return y, x, feature_ids, feature_names


def load_test_data(data_dir, dtype=np.float64, max_rows=None):
    """Load test features, IDs and feature names."""
    data_dir = Path(data_dir)
    return load_features(
        data_dir / "x_test.csv", dtype=dtype, max_rows=max_rows
    )

