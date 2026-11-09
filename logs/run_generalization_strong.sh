#!/usr/bin/env bash
# Strong neural baselines on the generalization cohort: the pretrained CheMeleon model and the five-seed D-MPNN
# ensemble (step 19) on every MoleculeACE target, so that the cohort compares the classical ensembles against the
# neural models that reached parity on TgDHFR, and not only against the default D-MPNN of step 09.
#
# Waits for the step 01/09 cohort to finish before starting, then runs TWO targets at a time, each capped at six
# workers of four threads: a single target is five folds per variant and would leave half of the 48 threads idle,
# while two uncapped processes would ask for 96. Resumable, because step 19 keeps a per-(variant, fold) checkpoint.
#
# Expected cost on this workstation: about 4.5 h per CheMeleon fold, so roughly 30 h for the twelve targets.
cd /home/andrey/Projects/__GITHUB__aavzpy2023/analysispaperKarimova2025 || exit 1
PY=/home/andrey/anaconda3/envs/mlvsgnn_repro/bin/python
STATUS=logs/generalization_strong_$(date +%Y%m%d_%H%M%S).status
CHEMELEON=$HOME/.chemprop/chemeleon_mp.pt
CONCURRENT=2

export MLGNN_PROFILE=generalization
export MLGNN_K19_VARIANTS=chemeleon,dmpnn_ens5
export MLGNN_K19_WORKERS=6
export MLGNN_K19_THREADS=4

say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

TARGETS=$(tail -n +2 data/moleculeace/index.csv | cut -d, -f1)

[ -s "$CHEMELEON" ] || { say "ABORT: $CHEMELEON not found (download it from Zenodo record 15460715)"; exit 1; }

# The cohort driver runs one target at a time, so between targets no pipeline process exists: wait on its output
# instead of on the process table, and only then on any pipeline still running.
say "WAITING for the step 01/09 cohort to finish"
while :; do
  MISSING=0
  for D in $TARGETS; do
    [ -s "generalization/$D/results/09_gnn_results.csv" ] || MISSING=$((MISSING + 1))
  done
  [ "$MISSING" -eq 0 ] && ! pgrep -f "run_pipeline.py --only" > /dev/null && break
  sleep 300
done
say "START driver ($CONCURRENT targets at a time, $MLGNN_K19_WORKERS workers x $MLGNN_K19_THREADS threads each)"

# One target, in its own process: the driver runs several of these at once.
run_target() {
  local D=$1
  local OUT="$PWD/generalization/$D"
  local CHECKPOINT="$OUT/results/19_gnn_strong_checkpoint.csv"
  # Ten rows (two variants x five folds) means the target is already scored; step 19 resumes a partial checkpoint.
  if [ -s "$CHECKPOINT" ] && [ "$(($(wc -l < "$CHECKPOINT") - 1))" -ge 10 ]; then
    say "SKIP $D 19 (done)"
    return 0
  fi
  mkdir -p "$OUT"
  say "START $D 19"
  MLGNN_TRAIN_FILE="$PWD/data/moleculeace/$D.csv" MLGNN_OUT_DIR="$OUT" $PY run_pipeline.py --only 19
  say "END $D 19 exit=$?"
}

PENDING=""
for D in $TARGETS; do
  PENDING="$PENDING $D"
  if [ "$(echo $PENDING | wc -w)" -ge "$CONCURRENT" ]; then
    for T in $PENDING; do run_target "$T" & done
    wait
    PENDING=""
  fi
done
for T in $PENDING; do run_target "$T" & done
wait

say "ALL DONE"
