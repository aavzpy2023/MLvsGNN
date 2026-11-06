#!/usr/bin/env bash
# Continues the joint re-run from step 03 (01 and 02 of the new protocol are complete and their checkpoints are kept).
# Usage: MLGNN_INNER_REPEATS=5 bash rerun_from03.sh      (run it inside tmux)
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
PY=/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python
LOG=logs/rerun_from03_$(date +%Y%m%d_%H%M%S).status
step() { echo "$(date '+%F %T') START $*" | tee -a "$LOG"; "$@"; local rc=$?; echo "$(date '+%F %T') END   exit=$rc $*" | tee -a "$LOG"; return $rc; }

echo "inner repeats: ${MLGNN_INNER_REPEATS:-5} | inner score: ${MLGNN_INNER_SCORE:-pooled}" | tee -a "$LOG"

# 03 resumes from results/03_y_randomization_checkpoint.csv if it is interrupted; 05 needs the mask of 04
step $PY run_pipeline.py --only 03 04 05 07 08
# Outlier sensitivity and scaffold split of the classical models (same nested CV). Their checkpoints were moved aside
# by rerun_all.sh, so they start from scratch with the new protocol.
step $PY run_pipeline.py --only S01 S02
step $PY run_pipeline.py --only SC01
# Models that do not change are re-summarized against the new classical results; 09 retrains (about 85 min)
step $PY run_pipeline.py --only 09 12 14 19 18 10 13 figures
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
