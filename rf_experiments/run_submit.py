"""Generate final submission for AIcrowd (Robust Path Version)"""

import argparse
import sys
import csv
from pathlib import Path
import numpy as np

# =========================================================================
# 核心修复：动态获取项目根目录，无论从哪里运行脚本都不会找不到模块
# =========================================================================
# 无论 run_submit.py 在 proj1/ 还是 proj1/rf_experiments/，都统一定位到 proj1/
CURRENT_FILE = Path(__file__).resolve()
ROOT = CURRENT_FILE.parent if CURRENT_FILE.parent.name == "proj1" else CURRENT_FILE.parent.parent

# 强制插入 sys.path 开头
sys.path.insert(0, str(ROOT))

from src.data_io import load_training_data, load_test_data
from src.preprocessing import BasicPreprocessor, BRFSSPreprocessor
from implementations import random_forest_classifier, random_forest_predict_proba

# ==========================================
# 从 run_random_forest.py 复制过来的特征工程函数
# ==========================================
def append_limitation_count(x, feature_names):
    """Append a numeric Yes count, or NaN unless all five answers are valid."""
    x = np.asarray(x)
    feature_names = tuple(feature_names)
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the columns of a 2D x")
    if "limitation_count" in feature_names:
        raise ValueError("limitation_count already exists")
    required = ("BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON")
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(f"Missing required limitation columns: {', '.join(missing)}")
    answers = x[:, [feature_names.index(name) for name in required]]
    valid = np.all(np.isfinite(answers) & ((answers == 1) | (answers == 2)), axis=1)
    counts = np.where(valid, np.sum(answers == 1, axis=1), np.nan)
    return np.column_stack((x, counts)), feature_names + ("limitation_count",)

# ==========================================
# 主流程
# ==========================================
def _parse_bool(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("expected TRUE or FALSE (case-insensitive)")


def _parse_threshold(value):
    try:
        threshold = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("threshold must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and in [0, 1]")
    return threshold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic", type=_parse_bool, default=False)
    parser.add_argument("--threshold", type=_parse_threshold, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    # 数据路径配置
    data_dir = ROOT / "dataset"
    output_dir = Path("C:/Users/Administrator/Desktop/proj1/rf_experiments/rf_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "basic" if args.basic else "brfss"
    output_csv = output_dir / f"submission_{mode}.csv"
    output_full_csv = output_dir / f"submission_{mode}_with_prob.csv"

    print(f"项目根目录: {ROOT}")
    print(f"数据集目录: {data_dir}")
    print(f"输出目录  : {output_dir}")

    print("\n[1/6] 正在加载数据 ...")
    y_train, x_train, _, feature_names = load_training_data(data_dir, logistic_labels=True)
    x_test, test_ids, test_feature_names = load_test_data(data_dir)
    if tuple(feature_names) != tuple(test_feature_names):
        raise ValueError("Training and test feature names and column order must match")

    print("[2/6] 特征工程 ...")
    # 添加限制项计数（保持训练集和测试集处理逻辑绝对一致）
    x_train_raw, augmented_names = append_limitation_count(x_train, feature_names)
    x_test_raw, _ = append_limitation_count(x_test, test_feature_names)

    print("[3/6] 预处理 ...")
    # 注意：使用你们自定义的 BRFSSPreprocessor，它会处理特定标签和独热编码
    pre = BasicPreprocessor() if args.basic else BRFSSPreprocessor()
    print(f"basic            : {str(args.basic).upper()}")
    print(f"preprocessor     : {type(pre).__name__}")
    print(f"input_features   : {x_train.shape[1]} raw, {len(augmented_names)} with engineered feature")
    x_train_proc = pre.fit_transform(x_train_raw, feature_names=augmented_names)
    x_test_proc = pre.transform(x_test_raw)
    print(f"output_features  : train={x_train_proc.shape[1]}, test={x_test_proc.shape[1]}")
    print(f"      特征维度从 {x_train_raw.shape[1]} 变成了 {x_train_proc.shape[1]}")

    print("[4/6] 训练随机森林 ...")
    # ==============================================================
    # ⚠️ 请确保这里的参数是你跑验证集时效果最好的一组
    # ==============================================================
    model, _ = random_forest_classifier(
        y_train,
        x_train_proc,
        criterion="gini",           
        num_trees=60,               
        max_tree_depth=12,          
        min_samples_split=20,
        max_features_mode="sqrt",   
        seed=42,
        class_balance=None          
    )

    print("[5/6] 预测测试集 ...")
    prob_test = random_forest_predict_proba(model, x_test_proc)

    # ==============================================================
    # ⚠️ 请确保这里的阈值是你跑验证集时选出的最佳 F1 阈值
    # ==============================================================
    print(f"n_features_per_split: {model['n_features_per_split']}")
    print(f"threshold        : {args.threshold}")
    best_threshold = args.threshold
    y_pred_binary = (prob_test >= best_threshold).astype(int)
    
    # AIcrowd 要求输出 -1 和 1
    y_pred_aicrowd = np.where(y_pred_binary == 1, 1, -1)

    print("[6/6] 保存结果 ...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for i, pred in zip(test_ids, y_pred_aicrowd):
            writer.writerow([i, pred])
    
    with open(output_full_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Probability", "Prediction_01", "Prediction_AICrowd"])
        for i, p, pred01, pred_ai in zip(test_ids, prob_test, y_pred_binary, y_pred_aicrowd):
            writer.writerow([i, f"{p:.4f}", pred01, pred_ai])

    print(f"✅ AIcrowd 提交文件已保存至: {output_csv}")

if __name__ == "__main__":
    main()
