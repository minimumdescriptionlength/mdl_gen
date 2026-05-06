#!/bin/bash
#SBATCH -J wd_grid_seqreps
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19

set -euo pipefail

# Avoid .pyc/__pycache__ races across array tasks
export PYTHONDONTWRITEBYTECODE=1

# --- sweep mode configuration ---
# Set SWEEP_MODE environment variable to control which experiments to run:
#   SWEEP_MODE=wd       (default) - weight decay sweep only
#   SWEEP_MODE=dropout  - dropout sweep only
#   SWEEP_MODE=both     - both sweeps (requires larger array, adjust --array accordingly)
SWEEP_MODE=${SWEEP_MODE:-"both"}
echo "SWEEP_MODE=${SWEEP_MODE}"

# Validate sweep mode
if [[ ! "$SWEEP_MODE" =~ ^(wd|dropout|both)$ ]]; then
  echo "ERROR: SWEEP_MODE must be 'wd', 'dropout', or 'both'. Got: $SWEEP_MODE"
  exit 1
fi

# --- pick your Python (CONDA env) ---
set +u  # Disable unbound variable check for bashrc sourcing
source ~/.bashrc || true
set -u  # Re-enable unbound variable check
conda activate mdl
echo "[Python] $(which python)"
python -V

# --- paths ---
SCRIPT="./mdl_preq_blockwise.py"
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
FIXED_EPOCHS=0
LABEL_SMOOTH=0.1

# --- sweep lists ---
WD_LIST=(0.0 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2)
REPEATS=(0 1 2)
ARCHS=(resnet34)

# --- dropout sweep: 20 points from 0% to 50% ---
DROPOUT_LIST=(0.0 0.02632 0.05263 0.07895 0.10526 0.13158 0.15789 0.18421 0.21053 0.23684 0.26316 0.28947 0.31579 0.34211 0.36842 0.39474 0.42105 0.44737 0.47368 0.5)

# --- determine sweep parameters based on mode ---
if [[ "$SWEEP_MODE" == "wd" ]]; then
  # Weight decay sweep (original): indices 0-9
  if [[ $SLURM_ARRAY_TASK_ID -ge 10 ]]; then
    echo "SWEEP_MODE=wd only uses array indices 0-9. Task $SLURM_ARRAY_TASK_ID skipped."
    exit 0
  fi
  PARAM_IDX=$SLURM_ARRAY_TASK_ID
  WD=${WD_LIST[$PARAM_IDX]}
  DROPOUT=0.0
  PARAM_NAME="WD"
  PARAM_VALUE=$WD
  
elif [[ "$SWEEP_MODE" == "dropout" ]]; then
  # Dropout sweep: all indices 0-19 used for dropout
  PARAM_IDX=$SLURM_ARRAY_TASK_ID
  DROPOUT=${DROPOUT_LIST[$PARAM_IDX]}
  WD=0.0
  PARAM_NAME="DROPOUT"
  PARAM_VALUE=$DROPOUT
  
elif [[ "$SWEEP_MODE" == "both" ]]; then
  # Both sweeps: indices 0-9 for WD, indices 10-19 for DROPOUT
  if [[ $SLURM_ARRAY_TASK_ID -lt 10 ]]; then
    PARAM_IDX=$SLURM_ARRAY_TASK_ID
    WD=${WD_LIST[$PARAM_IDX]}
    DROPOUT=0.0
    PARAM_NAME="WD"
    PARAM_VALUE=$WD
  else
    PARAM_IDX=$((SLURM_ARRAY_TASK_ID - 10))
    DROPOUT=${DROPOUT_LIST[$PARAM_IDX]}
    WD=0.0
    PARAM_NAME="DROPOUT"
    PARAM_VALUE=$DROPOUT
  fi
fi

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID: Running ${PARAM_NAME}=${PARAM_VALUE} sweep"
echo "Assigned WD=${WD}, DROPOUT=${DROPOUT}"

# unique per-WD/dropout sweep output (avoid CSV collisions)
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

# --- dirs ---
mkdir -p logs "${OUT}" "${SWEEP_OUT}"

# --- preflight ---
echo "Host: $(hostname)"; date
nvidia-smi || true
echo "[Check] SCRIPT: $( [ -f "${SCRIPT}" ] && echo ok || echo MISSING )"
echo "[Check] SWEEP : $( [ -f "${SWEEP}" ] && echo ok || echo MISSING )"
echo "[Check] DATA  : $( [ -d "${DATA}" ] && echo ok || echo will-download )"

for ARCH in "${ARCHS[@]}"; do
  echo "--- ARCH=${ARCH} ---"
  
  # Organize output by sweep mode and parameters
  if [[ "$SWEEP_MODE" == "wd" ]]; then
    JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/wd_sweep/wd_${WD_TOKEN}"
  elif [[ "$SWEEP_MODE" == "dropout" ]]; then
    JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/dropout_sweep/dropout_${DROPOUT_TOKEN}"
  elif [[ "$SWEEP_MODE" == "both" ]]; then
    if [[ $SLURM_ARRAY_TASK_ID -lt 10 ]]; then
      JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/both_sweeps/wd_part/wd_${WD_TOKEN}"
    else
      JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/${ARCH}/both_sweeps/dropout_part/dropout_${DROPOUT_TOKEN}"
    fi
  fi
  
  mkdir -p "${JOB_SWEEP_OUT_ARCH}"
  for REP in "${REPEATS[@]}"; do
    echo "==== ARCH=${ARCH} WD=${WD}, DROPOUT=${DROPOUT}, repeat=${REP} ===="
    python "${SWEEP}" \
      --script "${SCRIPT}" \
      --data "${DATA}" --out "${OUT}" --sweep_out "${JOB_SWEEP_OUT_ARCH}" \
      --wds ${WD} \
      --base_seed ${BASE_SEED} --repeat_iter ${REP} \
      --batch_size ${BATCH_SIZE} --workers ${WORKERS} \
      --optimizer sgd --lr ${LR} --label_smoothing ${LABEL_SMOOTH} \
      --min_lr_factor ${MIN_LR_FACTOR} --arch ${ARCH} --dropout ${DROPOUT} --quiet
      # optionally: --fixed_epochs ${FIXED_EPOCHS}
  done
  echo "Done ARCH=${ARCH}, WD=${WD}, DROPOUT=${DROPOUT} (all repeats)."
done

echo "Done array task ${SLURM_ARRAY_TASK_ID}: WD=${WD}, DROPOUT=${DROPOUT} (all architectures and repeats)."
