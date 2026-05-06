#!/bin/bash
# -----------------------------------------------------------------------------
# Single-PC equivalent of run_wd_array_seqreps.sbatch.
# Runs the weight-decay and/or dropout sweeps on CIFAR-100 sequentially.
#
# Usage:
#   SWEEP_MODE=wd      bash run_wd_array_seqreps.sh   # 10 weight-decay points
#   SWEEP_MODE=dropout bash run_wd_array_seqreps.sh   # 20 dropout points
#   SWEEP_MODE=both    bash run_wd_array_seqreps.sh   # both (default)
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SWEEP_MODE=${SWEEP_MODE:-"both"}
echo "SWEEP_MODE=${SWEEP_MODE}"
if [[ ! "$SWEEP_MODE" =~ ^(wd|dropout|both)$ ]]; then
  echo "ERROR: SWEEP_MODE must be 'wd', 'dropout', or 'both'. Got: $SWEEP_MODE"
  exit 1
fi

# --- conda env ---
set +u
source ~/.bashrc || true
set -u
conda activate mdl
echo "[Python] $(which python)"
python -V

# --- paths ---
SCRIPT="./mdl_preq_blockwise_test.py"
SWEEP="./sweep_wd_mdl.py"
DATA="./data_cifar100"
OUT="./runs_cifar100"
SWEEP_OUT="./runs_sweeps_cifar100"

# --- training knobs ---
BASE_SEED=1000
BATCH_SIZE=256
WORKERS=4
LR=0.1
MIN_LR_FACTOR=1e-3
LABEL_SMOOTH=0.1

WD_LIST=(0.0 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2)
DROPOUT_LIST=(0.0 0.02632 0.05263 0.07895 0.10526 0.13158 0.15789 0.18421 0.21053 0.23684 0.26316 0.28947 0.31579 0.34211 0.36842 0.39474 0.42105 0.44737 0.47368 0.5)
REPEATS=(0 1 2)
ARCHS=(resnet34)

mkdir -p logs "${OUT}" "${SWEEP_OUT}"
echo "Host: $(hostname)"; date
nvidia-smi -L || true

# Build the (PARAM_NAME, WD, DROPOUT) job list according to SWEEP_MODE
JOBS=()
if [[ "$SWEEP_MODE" == "wd" || "$SWEEP_MODE" == "both" ]]; then
  for V in "${WD_LIST[@]}"; do JOBS+=("WD ${V} 0.0"); done
fi
if [[ "$SWEEP_MODE" == "dropout" || "$SWEEP_MODE" == "both" ]]; then
  for V in "${DROPOUT_LIST[@]}"; do JOBS+=("DROPOUT 0.0 ${V}"); done
fi

for JOB in "${JOBS[@]}"; do
  read -r PARAM_NAME WD DROPOUT <<< "${JOB}"
  PARAM_VALUE=$([[ "${PARAM_NAME}" == "WD" ]] && echo "${WD}" || echo "${DROPOUT}")
  echo "Running ${PARAM_NAME}=${PARAM_VALUE}  (WD=${WD}, DROPOUT=${DROPOUT})"

  WD_TOKEN=$(python - <<PY
wd=float("${WD}")
print(f"{wd:.1e}".replace("+","").replace("-","m"))
PY
)
  DROPOUT_TOKEN=$(python - <<PY
dropout=float("${DROPOUT}")
print(f"{dropout:.5f}".rstrip('0').rstrip('.'))
PY
)

  for ARCH in "${ARCHS[@]}"; do
    if [[ "$SWEEP_MODE" == "wd" ]]; then
      JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/wd_sweep/wd_${WD_TOKEN}"
    elif [[ "$SWEEP_MODE" == "dropout" ]]; then
      JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/dropout_sweep/dropout_${DROPOUT_TOKEN}"
    else
      if [[ "${PARAM_NAME}" == "WD" ]]; then
        JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/both_sweeps/wd_part/wd_${WD_TOKEN}"
      else
        JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/both_sweeps/dropout_part/dropout_${DROPOUT_TOKEN}"
      fi
    fi
    mkdir -p "${JOB_SWEEP_OUT_ARCH}"

    for REP in "${REPEATS[@]}"; do
      echo "==== ARCH=${ARCH} WD=${WD} DROPOUT=${DROPOUT} repeat=${REP} ===="
      python "${SWEEP}" \
        --script "${SCRIPT}" \
        --data "${DATA}" --out "${OUT}" --sweep_out "${JOB_SWEEP_OUT_ARCH}" \
        --wds ${WD} \
        --base_seed ${BASE_SEED} --repeat_iter ${REP} \
        --batch_size ${BATCH_SIZE} --workers ${WORKERS} \
        --optimizer sgd --lr ${LR} --label_smoothing ${LABEL_SMOOTH} \
        --min_lr_factor ${MIN_LR_FACTOR} --arch ${ARCH} --dropout ${DROPOUT} --quiet \
        2>&1 | tee "logs/${ARCH}_${PARAM_NAME}_${PARAM_VALUE}_rep${REP}.log"
    done
  done
done
echo "All tasks done at $(date)"
