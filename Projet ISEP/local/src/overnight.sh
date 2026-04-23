#!/bin/bash
# Overnight pipeline to push OOF score as high as possible.
# Each step logs to outputs/overnight.log and has safety timeouts.
# If a step hangs, the next step gets the OOF/test files produced so far.
#
# Phases:
#   1) full-fit base=48 (5 seeds) — 100% data train, no CV → test-only probs
#   2) full-fit base=64 (3 seeds) → test-only probs
#   3) intermediate blend (CV + full-fit + features) → save stack_test for pseudo
#   4) pseudo-labeling round — retrain CNN on train + high-conf test → test probs
#   5) final blend (everything) → 3 candidate submissions

set +e
cd "$(dirname "$0")/.."
LOG=outputs/overnight.log
mkdir -p outputs
: > "$LOG"

run_with_timeout() {
    local tmo_min=$1; shift
    local desc=$1; shift
    echo "[$(date)] START $desc (timeout ${tmo_min}m)" >> "$LOG"
    ( "$@" ) >> "$LOG" 2>&1 &
    local pid=$!
    local elapsed=0
    while kill -0 $pid 2>/dev/null; do
        sleep 30
        elapsed=$((elapsed + 30))
        # hang detection: check CPU every 3 min
        if [ $((elapsed % 180)) -eq 0 ]; then
            local cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
            if [ -n "$cpu" ]; then
                echo "[$(date)]   heartbeat $desc cpu=${cpu}% elapsed=${elapsed}s" >> "$LOG"
            fi
        fi
        if [ $elapsed -ge $((tmo_min * 60)) ]; then
            echo "[$(date)] TIMEOUT $desc (${elapsed}s) — killing pid $pid" >> "$LOG"
            kill -9 $pid 2>/dev/null
            pkill -9 -P $pid 2>/dev/null
            break
        fi
    done
    wait $pid 2>/dev/null
    local rc=$?
    echo "[$(date)] END $desc rc=$rc" >> "$LOG"
    return $rc
}

echo "[$(date)] OVERNIGHT START" >> "$LOG"

# Phase 1: full-fit base=48 (5 seeds)
run_with_timeout 50 "fullfit_b48" \
    python3 -u src/train_full_fit.py \
        --data-root ../data \
        --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
        --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
        --output-dir ./outputs/run_full_fit_b48 \
        --seeds 42 1234 7 555 999 --epochs 35 --base 48 --drop 0.3

# Phase 2: full-fit base=64 (3 seeds)
run_with_timeout 40 "fullfit_b64" \
    python3 -u src/train_full_fit.py \
        --data-root ../data \
        --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
        --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
        --output-dir ./outputs/run_full_fit_b64 \
        --seeds 42 1234 7 --epochs 35 --base 64 --drop 0.35

# Phase 3: intermediate blend
run_with_timeout 5 "blend_intermediate" \
    python3 -u src/blend_all.py \
        --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
        --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
        --cnn-dir ./outputs/run_strong \
        --cnn-dir ./outputs/run_strong_v2 \
        --features-dir ./outputs/run_features_v2 \
        --full-test ./outputs/run_full_fit_b48/test_full_probs.npy \
        --full-test ./outputs/run_full_fit_b64/test_full_probs.npy \
        --output-dir ./outputs/run_blend_intermediate

# Phase 4: pseudo-labeling round — uses stack_test from intermediate blend
if [ -f outputs/run_blend_intermediate/stack_test.npy ] 2>/dev/null || true; then
    # copy the test probs so pseudo script can read cleanly
    STACK_TEST=outputs/run_blend_intermediate/stack_test.npy
    # blend_all saves only *_test.npy under test_probs of stack — find it via its files:
    if [ ! -f "$STACK_TEST" ]; then
        STACK_TEST=$(ls outputs/run_blend_intermediate/*test*.npy 2>/dev/null | head -1)
    fi
    echo "[$(date)] pseudo will use $STACK_TEST" >> "$LOG"

    run_with_timeout 55 "pseudo_b48" \
        python3 -u src/train_pseudo.py \
            --data-root ../data \
            --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
            --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
            --stack-test-probs "$STACK_TEST" \
            --output-dir ./outputs/run_pseudo_b48 \
            --seeds 42 1234 7 555 --epochs 30 --base 48 --drop 0.3 \
            --threshold 0.99

    run_with_timeout 40 "pseudo_b64" \
        python3 -u src/train_pseudo.py \
            --data-root ../data \
            --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
            --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
            --stack-test-probs "$STACK_TEST" \
            --output-dir ./outputs/run_pseudo_b64 \
            --seeds 42 1234 --epochs 30 --base 64 --drop 0.35 \
            --threshold 0.99
fi

# Phase 5: final blend (everything)
FULLTESTS=""
for f in outputs/run_full_fit_b48/test_full_probs.npy \
         outputs/run_full_fit_b64/test_full_probs.npy \
         outputs/run_pseudo_b48/test_pseudo_probs.npy \
         outputs/run_pseudo_b64/test_pseudo_probs.npy ; do
    [ -f "$f" ] && FULLTESTS="$FULLTESTS --full-test $f"
done
run_with_timeout 5 "blend_final" \
    python3 -u src/blend_all.py \
        --train-csv ../MPA_MLF_final_project2026/y_train_v2.csv \
        --sample-submission ../MPA_MLF_final_project2026/y_test_submission_example_v2.csv \
        --cnn-dir ./outputs/run_strong \
        --cnn-dir ./outputs/run_strong_v2 \
        --features-dir ./outputs/run_features_v2 \
        $FULLTESTS \
        --output-dir ./outputs/FINAL

echo "[$(date)] OVERNIGHT DONE" >> "$LOG"
