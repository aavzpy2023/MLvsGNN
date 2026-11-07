#!/usr/bin/env bash
# Generalization benchmark: every MoleculeACE target of data/moleculeace/index.csv through the classical nested CV
# (step 01) and the default D-MPNN (step 09), one target at a time so that each run gets the whole machine.
#
# Resumable: a step whose result file already exists is skipped, so the driver can be stopped and restarted without
# losing finished targets. It waits first for any pipeline started elsewhere, to avoid oversubscribing the CPU.
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
PY=/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python
STATUS=logs/generalization_$(date +%Y%m%d_%H%M%S).status
export MLGNN_PROFILE=generalization

say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

say "WAITING for any running pipeline"
while pgrep -f "run_pipeline.py --only" > /dev/null; do sleep 300; done
say "START driver"

# Smallest target first, so that a protocol error surfaces in three hours instead of six.
TARGETS=$(tail -n +2 data/moleculeace/index.csv | cut -d, -f1)
for D in $TARGETS; do
  export MLGNN_TRAIN_FILE="$PWD/data/moleculeace/$D.csv"
  export MLGNN_OUT_DIR="$PWD/generalization/$D"
  mkdir -p "$MLGNN_OUT_DIR"
  for STEP in 01 09; do
    case $STEP in
      01) MARKER="$MLGNN_OUT_DIR/results/01_nested_cv_final_results.csv" ;;
      09) MARKER="$MLGNN_OUT_DIR/results/09_gnn_results.csv" ;;
    esac
    if [ -s "$MARKER" ]; then
      say "SKIP $D $STEP (done)"
      continue
    fi
    say "START $D $STEP"
    $PY run_pipeline.py --only $STEP
    ST=$?
    say "END $D $STEP exit=$ST"
    [ $ST -ne 0 ] && say "FAILED $D $STEP, continuing with the next target" && break
  done
done
say "ALL DONE"
