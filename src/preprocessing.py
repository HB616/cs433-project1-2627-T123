"""Leakage-safe splitting and baseline preprocessing for Project 1."""

import numpy as np


def stratified_split_indices(y, validation_ratio=0.2, seed=42):
    """Return reproducible train/validation indices with preserved class ratios.

    Each class is shuffled and split independently. Both output index arrays are
    shuffled before being returned, so their row order does not reveal classes.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if y.size < 2:
        raise ValueError("At least two samples are required")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    classes, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        rare_classes = classes[counts < 2].tolist()
        raise ValueError(
            "Every class needs at least two samples for a stratified split; "
            f"too few samples for {rare_classes}"
        )

    random_generator = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []

    for class_value, class_count in zip(classes, counts):
        class_indices = np.flatnonzero(y == class_value)
        random_generator.shuffle(class_indices)
        validation_count = int(round(class_count * validation_ratio))
        validation_count = min(max(validation_count, 1), class_count - 1)
        validation_parts.append(class_indices[:validation_count])
        train_parts.append(class_indices[validation_count:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    random_generator.shuffle(train_indices)
    random_generator.shuffle(validation_indices)
    return train_indices, validation_indices


class BasicPreprocessor:
    """Median-impute and standardize features using training data only.

    Columns that are entirely non-finite or constant among their finite
    training values are removed. The fitted column mask, medians, means and
    scales are then reused unchanged for validation and test data.

    This baseline intentionally does not reinterpret BRFSS survey sentinel
    codes. Those transformations require a verified feature codebook.
    """

    def __init__(self):
        self.num_input_features_ = None
        self.keep_mask_ = None
        self.medians_ = None
        self.means_ = None
        self.scales_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.dropped_features_ = None

    def fit(self, x, feature_names=None):
        """Learn columns, imputation values and scaling from training data."""
        x = self._validate_matrix(x)
        num_features = x.shape[1]
        if feature_names is None:
            feature_names = tuple(f"feature_{index}" for index in range(num_features))
        else:
            feature_names = tuple(feature_names)
            if len(feature_names) != num_features:
                raise ValueError(
                    "feature_names must match the number of columns in x"
                )

        keep_mask = np.ones(num_features, dtype=bool)
        dropped_features = []
        for index in range(num_features):
            finite_values = x[np.isfinite(x[:, index]), index]
            if finite_values.size == 0:
                keep_mask[index] = False
                dropped_features.append((feature_names[index], "all_missing"))
            elif np.unique(finite_values).size == 1:
                keep_mask[index] = False
                dropped_features.append((feature_names[index], "constant"))

        if not np.any(keep_mask):
            raise ValueError("No usable feature columns remain after filtering")

        selected = x[:, keep_mask].copy()
        selected[~np.isfinite(selected)] = np.nan
        medians = np.nanmedian(selected, axis=0)
        imputed = np.where(np.isnan(selected), medians, selected)
        means = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0)
        scales[scales == 0.0] = 1.0

        self.num_input_features_ = num_features
        self.keep_mask_ = keep_mask
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.feature_names_in_ = feature_names
        self.feature_names_out_ = tuple(
            name for name, keep in zip(feature_names, keep_mask) if keep
        )
        self.dropped_features_ = tuple(dropped_features)
        return self

    def transform(self, x):
        """Apply the fitted training transformation to new rows."""
        self._check_is_fitted()
        x = self._validate_matrix(x)
        if x.shape[1] != self.num_input_features_:
            raise ValueError(
                f"Expected {self.num_input_features_} columns, got {x.shape[1]}"
            )

        selected = x[:, self.keep_mask_].copy()
        selected[~np.isfinite(selected)] = np.nan
        imputed = np.where(np.isnan(selected), self.medians_, selected)
        return (imputed - self.means_) / self.scales_

    def fit_transform(self, x, feature_names=None):
        """Fit on training data and return its transformed matrix."""
        return self.fit(x, feature_names=feature_names).transform(x)

    @staticmethod
    def _validate_matrix(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional feature matrix")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("x must contain at least one row and one column")
        return x

    def _check_is_fitted(self):
        if self.keep_mask_ is None:
            raise RuntimeError("BasicPreprocessor must be fitted before transform")
