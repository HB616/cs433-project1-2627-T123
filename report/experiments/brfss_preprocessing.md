# BRFSS Sentinel and One-Hot Preprocessing

**Date:** 2026-10-21  
**Status:** Development tuning and complete-dataset confirmation completed.

## Change

The new `BRFSSPreprocessor` uses the official CDC 2015 BRFSS codebook to:

- convert per-variable `Don't know/Not sure` and `Refused` codes to missing;
- retain calculated variables that explicitly describe missingness;
- median-impute and standardize numeric features;
- add numeric missingness indicators;
- one-hot encode categorical features, including a training-observed missing
  category;
- learn all statistics and category sets from the training partition only;
- keep the output shape fixed when validation/test contains an unseen category.

## Controlled comparison

Both runs used the first 50,000 rows, the same stratified 80/20 split with
`seed=42`, automatic positive-class weighting, `lambda=0.03`, `gamma=0.0003`
and 100,000 stochastic updates.

| Metric | Basic preprocessing | BRFSS preprocessing |
|---|---:|---:|
| Processed features (with intercept) | 316 | 1,187 |
| Accuracy | 0.7353 | **0.8195** |
| Balanced accuracy | 0.7433 | **0.7522** |
| Precision | 0.2113 | **0.2765** |
| Recall | **0.7529** | 0.6709 |
| F1-score | 0.3300 | **0.3916** |
| Specificity | 0.7336 | **0.8336** |

The feature engineering improves balanced accuracy by 0.0089 and F1 by 0.0616.
It trades some recall for substantially better precision and specificity.

## Reproduction

```bash
python3 evaluate_svm.py \
  --data-dir ../../dataset \
  --max-rows 50000 \
  --lambda 0.03 \
  --gamma 0.0003 \
  --max-iters 100000 \
  --preprocessing brfss \
  --max-missing-rate 1 \
  --seed 42
```

## Hyperparameter tuning on 50,000 rows

The transformed feature space was retuned rather than reusing the parameters
selected for the numeric baseline.

### Stage 1: regularization and class weighting

- `lambda`: 0.01, 0.03, 0.1
- class-weight multiplier: 0.5, 0.75, 1.0, 1.25
- fixed `gamma=0.0003`
- fixed `max_iters=100000`

The best balanced-accuracy region was `lambda=0.01` to `0.03` with a class-
weight multiplier of 1.25. Lower class weights produced better F1 and
precision but lower recall.

### Stage 2: learning rate and training length

- `lambda`: 0.01, 0.03
- class-weight multiplier: 1.0, 1.25
- `gamma`: 0.0001, 0.0003, 0.001
- `max_iters`: 100,000 and 200,000

The best development result was:

```text
lambda                  = 0.01
class_weight_multiplier = 1.0
gamma                   = 0.0001
max_iters               = 200000
```

| Metric | Development result |
|---|---:|
| Balanced accuracy | **0.7712** |
| F1-score | 0.3574 |
| Recall | **0.7933** |
| Precision | 0.2307 |
| Accuracy | 0.7530 |

## Complete-dataset confirmation

The candidate region was checked on all 328,135 labeled rows with the same
stratified 80/20 split and `seed=42`:

- Training rows: 262,508
- Validation rows: 65,627
- Processed features: 1,194 including the intercept
- Automatic positive-class weight: approximately 10.325

| Class-weight multiplier | Gamma | Iterations | Balanced accuracy | F1 | Recall | Precision | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.0001 | 200,000 | **0.7687** | 0.4016 | **0.7168** | 0.2790 | 0.8114 |
| 1.0 | 0.0001 | 100,000 | 0.7534 | 0.4101 | 0.6566 | 0.2982 | 0.8332 |
| 1.0 | 0.0003 | 200,000 | 0.7480 | 0.4107 | 0.6385 | 0.3027 | 0.8382 |
| 0.5 | 0.0001 | 200,000 | 0.7423 | **0.4252** | 0.6047 | **0.3279** | **0.8557** |

### Updated recommendation

- Use `lambda=0.01`, automatic class weight, `gamma=0.0001` and 200,000
  updates when balanced accuracy and recall are the primary objectives.
- Keep the same `lambda`, `gamma` and iteration count with a 0.5 class-weight
  multiplier as the F1/precision alternative.
- Relative to the previous complete-data numeric baseline (balanced accuracy
  0.7528, F1 0.3404), the selected BRFSS representation reaches 0.7687
  balanced accuracy and 0.4016 F1.

Complete-data reproduction:

```bash
python3 tune_svm.py \
  --data-dir ../../dataset \
  --max-rows 0 \
  --preprocessing brfss \
  --max-missing-rate 1 \
  --lambdas 0.01 \
  --class-weight-multipliers 0.5,1 \
  --gammas 0.0001,0.0003 \
  --iteration-values 100000,200000 \
  --seed 42
```

## Interpretation

The codebook-aware representation improves both balanced accuracy and F1 on
the complete validation split. The automatic class weight remains preferable
when detecting positive cases is important; halving it reduces false positives
and gives the stronger F1/precision compromise. These results still use one
holdout split, so final report claims should acknowledge split sensitivity.
