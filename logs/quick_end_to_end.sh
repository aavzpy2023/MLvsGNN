#!/usr/bin/env bash
# End-to-end smoke run of the 32 steps with the reduced 'quick' profile. Everything is written to quick_run/, so the
# published results cannot be touched. Usage: bash logs/quick_end_to_end.sh   (run it inside tmux)
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
export MLGNN_TF_PYTHON=/home/andrey/anaconda3/envs/karimova_repro/bin/python
LOG=logs/quick_end_to_end_$(date +%Y%m%d_%H%M%S).status
echo "$(date '+%F %T') START full --quick run" | tee -a "$LOG"
/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python run_pipeline.py --quick
STATUS=$?
echo "$(date '+%F %T') END exit=$STATUS" | tee -a "$LOG"
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
