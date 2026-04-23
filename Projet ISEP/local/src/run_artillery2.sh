#!/bin/bash
# Restart of full-fit chain (slimmed for time): base=48 (3 seeds) then base=64 (2 seeds).
# Uses unbuffered Python so the log updates live.
set -e
cd "$(dirname "$0")/.."
LOG=outputs/artillery2.log
echo "[$(date)] artillery2 starting" >> "$LOG"

python3 -u src/train_full_fit.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_full_fit_b48 \
  --seeds 42 1234 7 --epochs 35 --base 48 --drop 0.3 \
  >> "$LOG" 2>&1
echo "[$(date)] full-fit base=48 done" >> "$LOG"

python3 -u src/train_full_fit.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_full_fit_b64 \
  --seeds 42 1234 --epochs 35 --base 64 --drop 0.35 \
  >> "$LOG" 2>&1
echo "[$(date)] full-fit base=64 done" >> "$LOG"

echo "[$(date)] artillery2 complete" >> "$LOG"
