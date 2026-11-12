#!/usr/bin/env bash
# Step 12 (nested-search D-MPNN) on the two MoleculeACE targets where the default D-MPNN loses most.
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
PY=/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python
STATUS=logs/gnn_tuned_gen_$(date +%Y%m%d_%H%M%S).status
export MLGNN_PROFILE=generalization
for D in CHEMBL4203_Ki CHEMBL4616_EC50; do
  export MLGNN_TRAIN_FILE="$PWD/data/moleculeace/$D.csv"
  export MLGNN_OUT_DIR="$PWD/generalization/$D"
  echo "$(date '+%F %T') START $D 12" >> "$STATUS"
  $PY run_pipeline.py --only 12
  echo "$(date '+%F %T') END $D 12 exit=$?" >> "$STATUS"
done
echo "$(date '+%F %T') ALL DONE" >> "$STATUS"
