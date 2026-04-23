# MPA-MLF Final Project 2026

This folder contains a complete Python pipeline for the room occupancy classification project.

Contents:
- `src/train_cv.py`: cross-validated training, out-of-fold evaluation, ensembling, and submission generation
- `src/build_report.py`: PDF report generation from saved experiment artifacts
- `requirements.txt`: minimal package list for the pipeline
- `outputs/`: generated metrics, checkpoints, figures, and submission files

Default workflow:

```bash
python3 src/train_cv.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_resnet18_rgb \
  --experiments resnet18:rgb

python3 src/train_feature_model.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_feature_hgb

python3 src/train_cv.py \
  --data-root ../data \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/run_tiny \
  --experiments tinycnn:rgb

python3 src/blend_runs.py \
  --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
  --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
  --output-dir ./outputs/final_best \
  --component feature_hgb ./outputs/run_feature_hgb 0.58 \
  --component resnet18_rgb ./outputs/run_resnet18_rgb/resnet18__rgb 0.10 \
  --component tiny_rgb ./outputs/run_tiny/tinycnn__rgb 0.32

python3 src/build_report.py \
  --run-dir ./outputs/final_best \
  --output-pdf ./outputs/room_occupancy_report.pdf
```

Notes:
- The dataset is expected outside of this folder so it can be excluded from GitHub.
- The submission format follows the provided Kaggle sample file.
