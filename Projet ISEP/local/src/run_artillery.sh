#!/bin/bash
# Heavy artillery chain: waits for current train_strong to finish, then runs:
#   1) extra seeds with base=64 (architectural diversity)
#   2) full-fit on 100% data with 5 seeds at base=48 and 2 at base=64
# Logs everything to artillery.log
set -e
cd "$(dirname "$0")/.."
LOG=outputs/artillery.log

echo "[$(date)] artillery chain starting" >> "$LOG"

# wait until the current train_strong process is done
while pgrep -f "train_strong.py" > /dev/null; do
    sleep 30
done
echo "[$(date)] train_strong done, launching diversified runs" >> "$LOG"

# 1) extra seeds with base=64 (architectural diversity)
python3 src/train_strong.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_strong_v2 \
  --folds 5 --seeds 555 999 --epochs 30 --batch 128 --base 64 --drop 0.35 \
  >> "$LOG" 2>&1
echo "[$(date)] base=64 multi-seed done" >> "$LOG"

# 2) full-fit on 100% data with multiple seeds at base=48
python3 src/train_full_fit.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_full_fit_b48 \
  --seeds 42 1234 7 555 999 --epochs 35 --base 48 --drop 0.3 \
  >> "$LOG" 2>&1
echo "[$(date)] full-fit base=48 done" >> "$LOG"

# 3) full-fit base=64
python3 src/train_full_fit.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_full_fit_b64 \
  --seeds 42 1234 7 --epochs 35 --base 64 --drop 0.35 \
  >> "$LOG" 2>&1
echo "[$(date)] full-fit base=64 done" >> "$LOG"

echo "[$(date)] artillery chain complete" >> "$LOG"
