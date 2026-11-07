#!/usr/bin/env bash
# Re-runs the two steps left behind by code changes, then the summaries that consume their results.
# SC09 ran 2026-10-17, before 09_gnn_baseline.py moved onto qsartools (26d4eb2); step 15 ran 2026-10-18,
# before the only commit of its file (ac3ffd2). Usage: bash logs/rerun_stale_sc09_15.sh   (run it inside tmux)
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
PY=/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python
export MLGNN_TF_PYTHON=/home/andrey/anaconda3/envs/karimova_repro/bin/python
LOG=logs/rerun_stale_$(date +%Y%m%d_%H%M%S).status
step() { echo "$(date '+%F %T') START $*" | tee -a "$LOG"; "$@"; local rc=$?; echo "$(date '+%F %T') END   exit=$rc $*" | tee -a "$LOG"; return $rc; }

# The scaffold D-MPNN writes to scaffold/ (SCAFFOLD_ENV in run_pipeline.py); it leaves the random-fold step 09 alone
step $PY run_pipeline.py --only SC09
step $PY run_pipeline.py --only 15
# 13 re-summarizes the scaffold comparison, 18 the audit of the original DNN, then the figures and the manuscript check
step $PY run_pipeline.py --only 13 18 figures
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
