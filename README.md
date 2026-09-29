# EPFL CS-433 Machine Learning - Project 1

Heart-disease risk prediction project for the EPFL CS-433 Machine Learning course, Fall 2026.

## Team

- Member 1: to be added
- Member 2: to be added
- Member 3: to be added

## Project objective

Build a reproducible binary-classification pipeline for predicting the `_MICHD` label from the BRFSS health survey data. The project includes implementations of six required machine-learning methods, data preparation, local validation, feature engineering, model comparison, and generation of an AIcrowd submission.

## Repository structure

```text
.
├── README.md
├── implementations.py       # Six required course methods
├── run.py                   # Reproduce the final prediction file
├── explore_data.py          # Reproducible data-quality report
├── src/
│   ├── data_io.py           # NumPy CSV loader and validation
│   ├── data_exploration.py  # Feature profiling utilities
│   ├── preprocessing.py     # Stratified split and baseline preprocessing
│   └── metrics.py           # Binary classification evaluation
├── notebooks/               # Exploratory analysis only
├── report/                  # LaTeX report source and figures
└── tests/                   # Team-written tests
```

The CSV dataset is intentionally excluded from Git. By default, `run.py` expects it in:

```text
../../dataset/
```

That directory must contain:

```text
x_train.csv
y_train.csv
x_test.csv
sample_submission.csv
```

## Allowed dependencies

- Python standard library
- NumPy
- Visualization libraries only for visualization

External machine-learning libraries such as pandas, scikit-learn, PyTorch and TensorFlow are not allowed for Project 1.

## Loading the dataset

```python
from src.data_io import load_test_data, load_training_data

y, x, train_ids, feature_names = load_training_data("../../dataset")
x_test, test_ids, test_feature_names = load_test_data("../../dataset")
```

For logistic regression, request labels in `{0, 1}`:

```python
y_logistic, x, train_ids, feature_names = load_training_data(
    "../../dataset", logistic_labels=True
)
```

Empty CSV fields are loaded as `np.nan`. Imputation and feature-specific missing-value handling must be performed explicitly during preprocessing.

## Exploring data quality

Generate a full feature profile and a concise Markdown summary:

```bash
python explore_data.py --data-dir ../../dataset
```

For a quicker development run, inspect only the first 5,000 rows:

```bash
python explore_data.py --data-dir ../../dataset --max-rows 5000
```

The generated files are written to `outputs/eda/` and include missing-value
rates, unique-value counts, numeric quantiles, constant-column flags, and
train/test missing-rate differences. The `outputs/` directory is excluded from
Git because these reports can always be regenerated.

## Baseline preprocessing

The baseline preprocessing utilities provide a reproducible stratified split
and a leakage-safe transformation:

```python
from src.preprocessing import BasicPreprocessor, stratified_split_indices

train_idx, validation_idx = stratified_split_indices(y, seed=42)

preprocessor = BasicPreprocessor()
x_train = preprocessor.fit_transform(
    x[train_idx], feature_names=feature_names
)
x_validation = preprocessor.transform(x[validation_idx])
x_test_processed = preprocessor.transform(x_test)
```

`BasicPreprocessor` removes all-missing and constant training columns, imputes
remaining missing values with training medians, and standardizes using training
means and standard deviations. Validation and test data never influence these
parameters. BRFSS categorical and sentinel survey codes are intentionally left
unchanged until their meanings are verified against the feature codebook.

## Classification metrics

All models are evaluated through the same NumPy-only metrics implementation:

```python
from src.metrics import classification_metrics

metrics = classification_metrics(y_validation, y_pred)
print(metrics["recall"], metrics["f1_score"], metrics["balanced_accuracy"])
```

The returned dictionary contains confusion-matrix counts, accuracy, precision,
recall, specificity, F1-score, and balanced accuracy. Balanced accuracy and
recall are especially important because the Project 1 target is imbalanced.

## Required implementations

`implementations.py` must provide:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`

Each function returns `(w, loss)`, where `w` is a one-dimensional NumPy array.

## Running the project

The final command will be documented once the training and inference pipeline is implemented. The intended interface is:

```bash
python run.py --data-dir ../../dataset --output outputs/submission.csv
```

## Reproducibility

Before final submission, this repository must document:

- data-cleaning decisions;
- feature transformations;
- validation split or cross-validation procedure;
- model and hyperparameter choices;
- the exact command that reproduces the submitted prediction file.
