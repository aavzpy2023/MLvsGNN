#!/usr/bin/env bash
# Cost calibration for the generalization benchmark: one MoleculeACE target (ABL1, n=794, closest to the
# 859 of the main benchmark) through the classical nested CV and the default D-MPNN.
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
D=CHEMBL1862_Ki
export MLGNN_PROFILE=generalization
export MLGNN_TRAIN_FILE="$PWD/data/moleculeace/$D.csv"
export MLGNN_OUT_DIR="$PWD/generalization/$D"
LOG=logs/calibrate_${D}_$(date +%Y%m%d_%H%M%S).status
mkdir -p "$MLGNN_OUT_DIR"
for STEP in 01 09; do
  echo "$(date '+%F %T') START $STEP" >> "$LOG"
  /home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python run_pipeline.py --only $STEP
  echo "$(date '+%F %T') END $STEP exit=$?" >> "$LOG"
done
echo "$(date '+%F %T') ALL DONE" >> "$LOG"
