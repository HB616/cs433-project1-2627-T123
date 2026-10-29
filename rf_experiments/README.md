# Random Forest 

Random forest implementation and experiments for binary classification on the BRFSS 2015 dataset.

## 1. File Structure

- `../implementations.py` — Random forest training and prediction functions.
- `run_random_forest.py` — Train and evaluate the model on an 80/20 stratified train-validation split; compare classification thresholds.
- `run_submit.py` — Retrain on the full training dataset and generate AIcrowd submissions.
- `../src/preprocessing.py` — Basic preprocessing pipelines.
- `../src/brfss_2015.py` — BRFSS preprocessing pipelines.
- `../src/data_io.py` — Dataset loading utilities.
- `rf_output/` — Generated submission files.

## 2. Run Experiments

Run from the `rf_experiments/` directory.

**Basic preprocessing (recommended):**

```bash
python run_random_forest.py --basic TRUE --num-trees 60 --max-depth 12 --thresholds 0.1,0.15,0.18,0.2,0.25,0.3,0.4,0.5
```

**BRFSS preprocessing:**

```bash
python run_random_forest.py --basic FALSE
```

Optional arguments include `--class-balance`, `--num-trees`, `--max-depth`, and `--max-features`.

Validation metrics (accuracy, precision, recall, F1, and balanced accuracy) are printed in the terminal.

## 3. Generate Best Submission

```bash
python run_submit.py --basic TRUE --num-trees 50 --max-depth 12 --threshold 0.18
```

The script trains on the full training dataset and saves:

- `rf_output/submission_basic.csv` — AIcrowd submission (`Id`, `Prediction`).
- `rf_output/submission_basic_with_prob.csv` — Predictions and positive-class probabilities.

## 4. Dataset

Place the following files in `../dataset/`:

- `x_train.csv`
- `y_train.csv`
- `x_test.csv`

All preprocessing statistics are fitted on training data only.
