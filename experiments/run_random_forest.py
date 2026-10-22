"""
Random-forest experiment

Quick smoke test (small subset, few trees):
    python experiments/run_random_forest.py --max-rows 50000 --num-trees 10 --max-depth 8

Hyperparameters: python experiments/run_random_forest.py
Criterion: python experiments/run_random_forest.py --criterion gini / entrophy
max_features: python experiments/run_random_forest.py --max-features 8 / 17
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# make the project root importable when the script is run from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_training_data                      # noqa: E402
from src.preprocessing import (                                 # noqa: E402
    BasicPreprocessor,
    stratified_split_indices,
)
from src.metrics import classification_metrics                  # noqa: E402
from implementations import (                                   # noqa: E402
    random_forest_classifier,
    random_forest_predict_proba,
)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset", help="Directory containing the 4 CSV files.")
    parser.add_argument("--max-rows", type=int, default=None, help="Only load the first N rows (quick smoke test).")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # random forest hyperparameters
    parser.add_argument("--criterion", type=str, default="gini",choices=["gini", "entropy"])
    parser.add_argument("--num-trees", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--max-features", type=str, default="sqrt", help="'sqrt', 'log', or an integer.")
    parser.add_argument("--class-balance", type=float, default=None,
                        help="Oversample positives by this factor per tree, e.g. 9, 10, 11. ")

    # evaluation
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5",help="Comma-separated thresholds to evaluate.")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _parse_max_features(value):
    """Convert the CLI string to whatever random_forest_classifier expects."""
    if value in ("sqrt", "log"):
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--max-features must be 'sqrt', 'log' or an integer"
        )


def evaluate_at_thresholds(y_true, prob, thresholds):
    """Return a dict {threshold: metrics} for a range of thresholds."""
    results = {}
    for t in thresholds:
        y_pred = (prob >= t).astype(int)
        metrics = classification_metrics(
            y_true, y_pred, negative_label=0, positive_label=1
        )
        results[t] = metrics
    return results


def print_metrics_table(results):
    header = (
        f"{'thr':>5} | {'acc':>7} | {'prec':>7} | {'rec':>7} | "
        f"{'spec':>7} | {'F1':>7} | {'bal_acc':>8}"
    )
    print(header)
    print("-" * len(header))
    for t, m in results.items():
        print(
            f"{t:>5.2f} | {m['accuracy']:>7.4f} | {m['precision']:>7.4f} | "
            f"{m['recall']:>7.4f} | {m['specificity']:>7.4f} | "
            f"{m['f1_score']:>7.4f} | {m['balanced_accuracy']:>8.4f}"
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print("=" * 80)
    print("Random Forest Experiment")
    print("=" * 80)
    print(f"data_dir         : {args.data_dir}")
    print(f"max_rows         : {args.max_rows}")
    print(f"validation_ratio : {args.validation_ratio}")
    print(f"seed             : {args.seed}")
    print(f"criterion        : {args.criterion}")
    print(f"num_trees        : {args.num_trees}")
    print(f"max_depth        : {args.max_depth}")
    print(f"min_samples_split: {args.min_samples_split}")
    print(f"max_features     : {args.max_features}")
    print(f"thresholds       : {thresholds}")
    print(f"class_balance    : {args.class_balance}")
    print("-" * 80)

    # ------------------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------------------
    print("[1/6] Loading data ...")
    t0 = time.time()
    y, x, ids, feature_names = load_training_data(
        args.data_dir, logistic_labels=True, max_rows=args.max_rows)
    print(f"      x: {x.shape}, y: {y.shape}, "
          f"pos_rate={np.mean(y):.4f}  ({time.time() - t0:.2f}s)")

    # ------------------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------------------
    print("[2/6] Stratified split ...")
    train_idx, val_idx = stratified_split_indices(
        y, validation_ratio=args.validation_ratio, seed=args.seed
    )
    x_train_raw = x[train_idx]
    y_train = y[train_idx]
    x_val_raw = x[val_idx]
    y_val = y[val_idx]
    print(f"      train: {x_train_raw.shape}, val: {x_val_raw.shape}")
    print(f"      train pos_rate={np.mean(y_train):.4f}, "
          f"val pos_rate={np.mean(y_val):.4f}")

    # ------------------------------------------------------------------------------
    # 3. Preprocess (median impute + standardize, leakage-safe)
    # ------------------------------------------------------------------------------
    print("[3/6] Preprocessing ...")
    pre = BasicPreprocessor()
    x_train = pre.fit_transform(x_train_raw, feature_names=feature_names)
    x_val = pre.transform(x_val_raw)
    print(f"      kept {x_train.shape[1]}/{x.shape[1]} features")
    print(f"      dropped: {len(pre.dropped_features_)}")

    # ------------------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------------------
    print("[4/6] Training random forest ...")
    max_features = _parse_max_features(args.max_features)
    t0 = time.time()
    model, train_loss = random_forest_classifier(
        y_train,
        x_train,
        criterion=args.criterion,
        num_trees=args.num_trees,
        max_tree_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features_mode=max_features,
        seed=args.seed,
        class_balance=args.class_balance,
    )
    train_time = time.time() - t0
    print(f"      train_time = {train_time:.2f}s")
    print(f"      train_loss = {train_loss:.6f}")

    # ------------------------------------------------------------------------------
    # 5. Predict + evaluate on validation
    # ------------------------------------------------------------------------------
    print("[5/6] Evaluating on validation set ...")
    t0 = time.time()
    prob_val = random_forest_predict_proba(model, x_val)
    pred_time = time.time() - t0
    print(f"      predict_time = {pred_time:.2f}s "
          f"({len(y_val) / pred_time:.0f} samples/s)")
    print(f"      prob_val: min={prob_val.min():.4f}, "
          f"max={prob_val.max():.4f}, mean={prob_val.mean():.4f}")

    print()
    print("Validation metrics at different thresholds")
    results = evaluate_at_thresholds(y_val, prob_val, thresholds)
    print_metrics_table(results)

    # best threshold by F1
    best_t = max(results, key=lambda t: results[t]["f1_score"])
    best_f1 = results[best_t]["f1_score"]
    print()
    print(f">>> Best threshold by F1: {best_t} (F1 = {best_f1:.4f})")

    # ------------------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------------------
    print("[6/6] Summary")
    print("-" * 80)
    print(f"{'Model':<25} {'criterion':<10} {'F1@0.5':>10} {'F1@best':>10} "
          f"{'train_time(s)':>14}")
    print(
        f"{'RandomForest':<25} {args.criterion:<10} "
        f"{results[0.5]['f1_score']:>10.4f} {best_f1:>10.4f} "
        f"{train_time:>14.2f}"
    )


if __name__ == "__main__":
    main()