# Linear SVM Tuning Record

**Date:** 2026-10-15  
**Status:** Development search and complete-dataset confirmation completed.

## Goal

Tune the class-weighted Linear Soft-Margin SVM while keeping the validation
split and preprocessing identical for every configuration. The primary ranking
metric was **balanced accuracy**, with F1-score, recall, precision and ordinary
accuracy retained for comparison.

## Experimental setup

- Data: first 50,000 rows of the provided Project 1 training CSV files.
- Split: stratified 80% training / 20% validation, `seed=42`.
- Training rows: 40,000.
- Validation rows: 10,000.
- Raw features: 321.
- Processed features: 316, including the added intercept.
- Columns removed from this split: 6 all-missing or constant columns.
- Preprocessing: training-only median imputation and standardization; the
  learned transformation was reused unchanged on validation data.
- Model: primal Linear Soft-Margin SVM trained with stochastic subgradient
  descent and weighted hinge loss.
- Automatic positive-class weight:

  $$
  \frac{N_{negative}}{N_{positive}}\approx10.5507.
  $$

No external machine-learning library was used.

## Dummy baseline

The all-negative classifier demonstrates why ordinary accuracy is misleading
for this imbalanced target.

| Metric | All-negative baseline |
|---|---:|
| Accuracy | 0.9134 |
| Balanced accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-score | 0.0000 |
| Specificity | 1.0000 |

Confusion matrix: TN = 9,134, FP = 0, FN = 866, TP = 0.

## Search procedure

### Stage 1: regularization and class weighting

- `lambda`: 0.001, 0.01, 0.1
- class-weight multiplier: 0.5, 0.75, 1.0, 1.25
- fixed `gamma=0.001`
- fixed `max_iters=100000`

Best balanced accuracy in this stage: 0.7222, obtained with `lambda=0.1`
and class-weight multiplier 1.25.

### Stage 2: learning rate and training length

- fixed `lambda=0.1`
- class-weight multiplier: 1.0, 1.25, 1.5
- `gamma`: 0.0003, 0.001, 0.003
- `max_iters`: 100,000, 200,000

`gamma=0.0003` was consistently better than the larger values. Increasing the
training budget to 200,000 updates did not consistently improve validation
performance.

### Stage 3: local refinement

- `lambda`: 0.03, 0.1, 0.3
- class-weight multiplier: 0.75, 1.0, 1.25
- fixed `gamma=0.0003`
- fixed `max_iters=100000`

## Recommended configurations

### Candidate A: balanced accuracy and recall priority

```text
lambda                  = 0.1
gamma                   = 0.0003
max_iters               = 100000
class_weight_multiplier = 1.0
positive_class_weight   = 10.5507
```

| Metric | Candidate A |
|---|---:|
| Accuracy | 0.7172 |
| Balanced accuracy | **0.7454** |
| Precision | 0.2038 |
| Recall | **0.7794** |
| F1-score | 0.3231 |

This configuration detected the largest useful share of positive cases while
maintaining the best balanced accuracy found in the search.

### Candidate B: precision and F1 compromise

```text
lambda                  = 0.03
gamma                   = 0.0003
max_iters               = 100000
class_weight_multiplier = 0.75
positive_class_weight   = 7.9130
```

| Metric | Candidate B |
|---|---:|
| Accuracy | **0.7704** |
| Balanced accuracy | 0.7452 |
| Precision | **0.2320** |
| Recall | 0.7148 |
| F1-score | **0.3503** |

Candidate B has almost the same balanced accuracy as Candidate A, but makes
fewer false-positive predictions and achieves the better F1-score.

## Improvement over the original SVM configuration

The original setting was `lambda=0.01`, `gamma=0.001`, 100,000 updates and the
automatic class weight.

| Metric | Original | Candidate A | Candidate B |
|---|---:|---:|---:|
| Accuracy | 0.7318 | 0.7172 | **0.7704** |
| Balanced accuracy | 0.7037 | **0.7454** | 0.7452 |
| Precision | 0.1949 | 0.2038 | **0.2320** |
| Recall | 0.6697 | **0.7794** | 0.7148 |
| F1-score | 0.3019 | 0.3231 | **0.3503** |

## Interpretation and limitations

- Ordinary accuracy should not be used alone: the dummy classifier reaches
  91.34% accuracy while detecting no positive cases.
- Candidate A is appropriate if missing a positive case is especially costly.
- Candidate B provides the stronger F1/precision trade-off and is a useful
  second candidate for final model comparison.
- These results use the first 50,000 rows rather than a random 50,000-row sample
  or the complete dataset. The ranking must be confirmed on the complete data.
- Reusing a single validation split during tuning can gradually overfit that
  split. Final claims should be checked with the full dataset and preferably
  cross-validation or an untouched holdout.
- Feature engineering has not yet been applied beyond baseline imputation,
  standardization, constant-column removal and an intercept.

## Reproduction

Candidate A evaluation:

```bash
python3 evaluate_svm.py \
  --max-rows 50000 \
  --lambda 0.1 \
  --gamma 0.0003 \
  --max-iters 100000 \
  --positive-class-weight 0 \
  --seed 42
```

Candidate B uses a positive class weight of approximately 7.913:

```bash
python3 evaluate_svm.py \
  --max-rows 50000 \
  --lambda 0.03 \
  --gamma 0.0003 \
  --max-iters 100000 \
  --positive-class-weight 7.913 \
  --seed 42
```

The full grid can be reproduced with `tune_svm.py`; see the repository README
for its command-line interface.

## Complete-dataset confirmation

The candidate region was subsequently checked on all 328,135 labeled rows,
using the same 80/20 stratified split procedure and `seed=42`:

- Training rows: 262,508.
- Validation rows: 65,627.
- Processed features: 316 including the intercept.
- Automatic positive-class weight: approximately 10.325.
- `gamma=0.0003` and `max_iters=100000` for every configuration.

| $\lambda$ | Class-weight multiplier | Balanced accuracy | F1 | Recall | Precision | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 1.00 | **0.7528** | 0.3404 | 0.7743 | 0.2182 | 0.7350 |
| 0.10 | 1.00 | 0.7496 | 0.3310 | **0.7862** | 0.2097 | 0.7194 |
| 0.10 | 0.75 | 0.7461 | 0.3543 | 0.7187 | 0.2351 | 0.7686 |
| 0.03 | 0.75 | 0.7453 | **0.3589** | 0.7068 | **0.2405** | **0.7770** |

### Updated recommendation

- Use `lambda=0.03`, automatic class weight, `gamma=0.0003` and 100,000
  updates when balanced accuracy is the primary metric.
- Keep `lambda=0.03` with a 0.75 class-weight multiplier as the F1/precision
  alternative.
- The full-data experiment supports the learning-rate and iteration choices
  found on the development subset, but changes the preferred regularization
  strength from 0.1 to 0.03.

Complete-data reproduction:

```bash
python3 tune_svm.py \
  --max-rows 0 \
  --lambdas 0.03,0.1 \
  --class-weight-multipliers 0.75,1 \
  --gammas 0.0003 \
  --iteration-values 100000 \
  --seed 42
```

## Next experiment

Begin feature engineering while keeping the complete-data SVM configuration as
the fixed baseline. Compare each feature change against this baseline before
combining changes, then compare the resulting SVM fairly against Regularized
Logistic Regression and Random Forest.
