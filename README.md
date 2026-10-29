# EPFL CS-433 Machine Learning - Project 1

Team T123: Haobin Wang, Yunxin Huang, Chaoqi Xu

This repository contains our NumPy-only solution for predicting the BRFSS
`_MICHD` label. It includes the six required course implementations, data
loading and preprocessing, leakage-safe validation, model experiments, report
sources, and the script that reproduces our best AIcrowd submission.

## Reproduce the best submission

From the repository root, run:

```bash
python3 run.py \
  --data-dir ../../dataset \
  --output outputs/best-submission.csv
```

`run.py` performs the complete workflow: it loads the raw course CSVs, fits
preprocessing on all labeled rows, trains the final class-weighted linear SVM,
predicts the test set, validates the IDs and labels, and writes the AIcrowd CSV.
No pretrained weights are required. Full training uses dense one-hot features
and 200,000 stochastic updates, so it can take several minutes and requires
sufficient memory.

With the released course dataset, the expected output is:

```text
rows:                 109379
positive predictions: 16277
negative predictions: 93102
labels:               {-1, +1}
SHA-256:              8205f3b0616353fc47157c9e88a8062e2d1ee6fba9f94c350326943702b88282
```

This file corresponds to AIcrowd submission `338115`, which achieved positive-
class F1 `0.431` and accuracy `0.866`. The local validation estimates were F1
`0.432` and accuracy `0.868`.

### Final model configuration

- Model: class-weighted linear soft-margin SVM implemented in NumPy
- Preprocessing: codebook-aware BRFSS missing-value handling, median imputation,
  numeric standardization, missingness indicators, and categorical one-hot
  encoding
- Training data: all 328,135 labeled rows
- Processed representation: 1,195 columns including the intercept
- Positive-class hinge-loss weight: `num_negative / num_positive = 10.3248`
- L2 strength: `lambda = 0.01`
- Learning rate: `0.0001`
- Stochastic updates: `200000`
- Random seed: `42`
- Decision threshold: `0.5008938105844928`

The threshold is applied after training: a test example is predicted as `+1`
when `x @ w >= 0.5008938105844928`, and as `-1` otherwise. Passing
`--decision-threshold 0` reproduces the earlier operating point, not the best
submission.

## Data placement

The dataset is supplied by the course and is intentionally excluded from Git.
Place the unchanged files in one directory:

```text
dataset/
├── x_train.csv
├── y_train.csv
├── x_test.csv
└── sample_submission.csv    # optional reference; run.py does not read it
```

The default repository layout expects this directory at `../../dataset`.
Alternatively, pass any location through `--data-dir`.

The loaders verify headers, unique integer IDs, feature alignment, and labels.
Empty feature fields become `np.nan`. Training labels use `{-1, +1}`; the
loader can convert them to `{0, 1}` for the required logistic-regression
functions. No external dataset is used.

## Final pipeline

### Data preparation and feature generation

`BRFSSPreprocessor` learns every transformation from its training input only:

1. Map only feature-specific sentinel codes documented in the official 2015
   BRFSS codebook (for example, "Don't know" and "Refused") to missing values.
2. Drop unusable all-missing or constant columns.
3. Median-impute numeric features and standardize them with training means and
   standard deviations.
4. Add a binary missingness indicator for numeric columns that contain missing
   values.
5. One-hot encode categorical variables using categories observed during fit,
   including a separate missing category when required. Unseen categories in
   validation or test rows map to an all-zero block.
6. Add an intercept column.

The final fit on all labeled rows drops three raw features and produces 1,195
columns including the intercept. Test data never influence preprocessing.

### Validation and model selection

We used one 80/20 stratified holdout with seed `42`: 262,508 training rows and
65,627 validation rows. We did not use k-fold cross-validation. All
preprocessing was fitted on the training partition and reused unchanged for
validation. Because only 8.83% of labels are positive, we compared accuracy,
balanced accuracy, precision, recall, specificity, and positive-class F1.

The final SVM hyperparameters were selected from reproducible holdout
experiments. We then evaluated 302 thresholds derived from validation decision
scores. Threshold `0.5008938105844928` maximized positive-class F1, with
balanced accuracy as the tie-break. Finally, `run.py` refits the same pipeline
on all labeled rows before predicting the competition test set.

For comparison, the report also evaluates:

- L2-regularized logistic regression with balanced sampling and Basic
  preprocessing;
- a NumPy random forest with a functional-limitation feature;
- an all-negative reference showing why raw accuracy is misleading.

## Six required implementations

The root-level `implementations.py` provides the exact required functions:

```python
mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
least_squares(y, tx)
ridge_regression(y, tx, lambda_)
logistic_regression(y, tx, initial_w, max_iters, gamma)
reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
```

All return `(w, loss)` with `w` as a one-dimensional array. The MSE includes
the required factor `0.5`; SGD uses mini-batch size 1; least squares does not
use `numpy.linalg.lstsq`; and the returned ridge/logistic losses exclude the
regularization penalty, as required by the project specification.

`implementations.py` also contains the custom linear SVM and random-forest
components used in the experiments.

## Reproduce validation experiments

Evaluate the selected SVM on the complete 80/20 split:

```bash
python3 evaluate_svm.py \
  --data-dir ../../dataset \
  --max-rows 0 \
  --preprocessing brfss \
  --lambda 0.01 \
  --gamma 0.0001 \
  --max-iters 200000 \
  --seed 42
```

Regenerate the threshold analysis and report assets:

```bash
python3 generate_svm_threshold_figure.py \
  --data-dir ../../dataset \
  --max-rows 0 \
  --lambda 0.01 \
  --gamma 0.0001 \
  --max-iters 200000 \
  --seed 42 \
  --output-dir report/svm/assets

python3 generate_svm_report_assets.py \
  --data-dir ../../dataset \
  --max-rows 0 \
  --output-dir report/svm/assets
```

Random-forest experiments and their saved logs are under `rf_experiments/`.
The report source is under `report/`.

## Dependencies and rule compliance

Required for training and submission generation:

- Python 3
- NumPy

Matplotlib is used only by optional report-figure scripts. The modeling pipeline
does not use pandas, scikit-learn, PyTorch, TensorFlow, or another external ML
library. It does not use external training data.

## Tests

Run the repository tests from the project root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q .
```

For a fast end-to-end smoke test without replacing the final submission:

```bash
python3 run.py \
  --data-dir ../../dataset \
  --max-train-rows 2000 \
  --max-iters 100 \
  --output outputs/smoke-test.csv
```

The official public grading tests can also be run from the course repository's
`projects/project1/grading_tests` directory by passing this repository path to
`--github_link`. Those tests check the required files, signatures, docstrings,
formatting, and numerical outputs.

## Repository structure

```text
.
├── README.md
├── implementations.py          # Six required methods + custom models
├── run.py                      # Exact best-submission reproduction
├── evaluate_svm.py             # Leakage-safe SVM validation
├── tune_svm.py                 # SVM hyperparameter experiments
├── explore_data.py             # Data-quality profiling
├── src/
│   ├── brfss_2015.py           # Verified BRFSS codebook metadata
│   ├── data_io.py              # CSV loading and ID/label validation
│   ├── preprocessing.py        # Splits and fitted transformations
│   ├── metrics.py              # NumPy classification metrics
│   └── submission.py           # AIcrowd CSV validation and writing
├── rf_experiments/             # Random-forest experiments and logs
├── report/                     # LaTeX sources and generated figures
└── tests/                      # Repository unit tests
```

Generated predictions, raw data, caches, and temporary artifacts are excluded
from Git. Submit the repository using a GitHub URL pinned to the final commit.
