"""Leakage-safe splitting and baseline preprocessing for Project 1."""

import numpy as np

from src.brfss_2015 import CATEGORICAL_FEATURES, SENTINEL_VALUES


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


class BRFSSPreprocessor:
    """Apply codebook-aware missing handling and dense one-hot encoding.

    The transformation is learned only from the matrix passed to :meth:`fit`.
    Numeric columns are median-imputed and standardized; a binary missingness
    indicator is added when missing values occur in the training partition.
    Categorical columns are one-hot encoded with categories observed during
    fitting and, when needed, a separate missing category. Unseen validation or
    test categories produce an all-zero block instead of changing the output
    shape.

    The default sentinel and categorical metadata describe the released 2015
    BRFSS variables. ``max_missing_rate`` can optionally remove very sparse
    optional-module columns; its default keeps every non-empty feature so this
    class changes only the requested missing-value and encoding behavior.
    """

    def __init__(
        self,
        sentinel_values=None,
        categorical_features=None,
        max_missing_rate=1.0,
        output_dtype=np.float32,
    ):
        if not 0.0 < max_missing_rate <= 1.0:
            raise ValueError("max_missing_rate must be in (0, 1]")
        self.sentinel_values = dict(
            SENTINEL_VALUES if sentinel_values is None else sentinel_values
        )
        self.categorical_features = frozenset(
            CATEGORICAL_FEATURES
            if categorical_features is None
            else categorical_features
        )
        self.max_missing_rate = float(max_missing_rate)
        self.output_dtype = np.dtype(output_dtype)
        if not np.issubdtype(self.output_dtype, np.floating):
            raise ValueError("output_dtype must be a floating-point dtype")

        self.num_input_features_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.feature_specs_ = None
        self.dropped_features_ = None
        self.num_output_features_ = None

    def fit(self, x, feature_names):
        """Learn categories and numeric statistics from training data only."""
        x = BasicPreprocessor._validate_matrix(x)
        feature_names = tuple(feature_names)
        if len(feature_names) != x.shape[1]:
            raise ValueError("feature_names must match the number of columns in x")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")

        specs = []
        output_names = []
        dropped = []
        for column_index, feature_name in enumerate(feature_names):
            values, missing = self._clean_column(x[:, column_index], feature_name)
            missing_rate = float(np.mean(missing))
            finite_values = values[~missing]
            if finite_values.size == 0:
                dropped.append((feature_name, "all_missing"))
                continue
            if missing_rate >= self.max_missing_rate:
                dropped.append((feature_name, "missing_rate"))
                continue

            if feature_name in self.categorical_features:
                categories = np.unique(finite_values)
                has_missing = bool(np.any(missing))
                if categories.size == 1 and not has_missing:
                    dropped.append((feature_name, "constant"))
                    continue
                specs.append(
                    ("categorical", column_index, feature_name, categories, has_missing)
                )
                output_names.extend(
                    f"{feature_name}=={self._format_category(category)}"
                    for category in categories
                )
                if has_missing:
                    output_names.append(f"{feature_name}==MISSING")
                continue

            median = float(np.median(finite_values))
            imputed = np.where(missing, median, values)
            mean = float(np.mean(imputed))
            scale = float(np.std(imputed))
            if scale == 0.0:
                dropped.append((feature_name, "constant"))
                continue
            has_missing = bool(np.any(missing))
            specs.append(
                ("numeric", column_index, feature_name, median, mean, scale, has_missing)
            )
            output_names.append(feature_name)
            if has_missing:
                output_names.append(f"{feature_name}==MISSING")

        if not specs:
            raise ValueError("No usable feature columns remain after preprocessing")

        self.num_input_features_ = x.shape[1]
        self.feature_names_in_ = feature_names
        self.feature_names_out_ = tuple(output_names)
        self.feature_specs_ = tuple(specs)
        self.dropped_features_ = tuple(dropped)
        self.num_output_features_ = len(output_names)
        return self

    def transform(self, x):
        """Apply the fitted codebook-aware transformation without refitting."""
        self._check_is_fitted()
        x = BasicPreprocessor._validate_matrix(x)
        if x.shape[1] != self.num_input_features_:
            raise ValueError(
                f"Expected {self.num_input_features_} columns, got {x.shape[1]}"
            )

        transformed = np.zeros(
            (x.shape[0], self.num_output_features_), dtype=self.output_dtype
        )
        output_index = 0
        for spec in self.feature_specs_:
            kind, column_index, feature_name = spec[:3]
            values, missing = self._clean_column(x[:, column_index], feature_name)
            if kind == "categorical":
                categories, has_missing = spec[3:]
                width = categories.size
                transformed[:, output_index : output_index + width] = (
                    values[:, None] == categories[None, :]
                )
                output_index += width
                if has_missing:
                    transformed[:, output_index] = missing
                    output_index += 1
            else:
                median, mean, scale, has_missing = spec[3:]
                imputed = np.where(missing, median, values)
                transformed[:, output_index] = (imputed - mean) / scale
                output_index += 1
                if has_missing:
                    transformed[:, output_index] = missing
                    output_index += 1
        return transformed

    def fit_transform(self, x, feature_names):
        """Fit on training data and return its transformed representation."""
        return self.fit(x, feature_names).transform(x)

    def _clean_column(self, column, feature_name):
        values = np.asarray(column, dtype=np.float64).copy()
        missing = ~np.isfinite(values)
        for sentinel in self.sentinel_values.get(feature_name, ()):
            missing |= values == sentinel
        values[missing] = np.nan
        return values, missing

    @staticmethod
    def _format_category(category):
        return str(int(category)) if float(category).is_integer() else str(category)

    def _check_is_fitted(self):
        if self.feature_specs_ is None:
            raise RuntimeError("BRFSSPreprocessor must be fitted before transform")
