#!/bin/bash

# 处理不平衡数据的关键实验 0 / 9134:0.0866 = 10.547
echo ">>> [1/3] calss balance parameter = 5"
python experiments/run_random_forest.py --class-balance 5 --criterion gini --num-trees 50 --max-depth 10 \
    2>&1 | tee experiments/rf_balance_5.log

echo ">>> [2/3] calss balance parameter = 2"
python experiments/run_random_forest.py --class-balance 2  --criterion gini --num-trees 50 --max-depth 10 \
    2>&1 | tee experiments/rf_balance2.log

echo ">>> [3/3] calss balance parameter = 11"
python experiments/run_random_forest.py --class-balance 11 --criterion gini --num-trees 50 --max-depth 10 \
    2>&1 | tee experiments/rf_balance11.log

# 或者
for k in 9 10 11; do
  echo "===== class_balance = ${k} ====="
  python experiments/run_random_forest.py \
      --class-balance ${k} \
      --criterion gini \
      --num-trees 50 \
      --max-depth 10 \
      --thresholds 0.3,0.4,0.5,0.6,0.7 \
  | tee experiments/rf_balance_${k}.log
done