#!/bin/bash
#SBATCH -J arch_grid_seqreps
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

# Avoid .pyc/__pycache__ races across array tasks
export PYTHONDONTWRITEBYTECODE=1

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

# --- fixed parameters for this sweep ---
DROPOUT=0.0
WD=0.0

# --- sweep lists ---
ARCH_LIST=(resnet18 resnet34 resnet50 resnet101 resnet152)
REPEATS=(0 1 2)

# --- dirs ---
mkdir -p logs "${OUT}" "${SWEEP_OUT}"

# --- preflight ---
echo "Host: $(hostname)"; date
nvidia-smi || true
echo "[Check] SCRIPT: $( [ -f "${SCRIPT}" ] && echo ok || echo MISSING )"
echo "[Check] SWEEP : $( [ -f "${SWEEP}" ] && echo ok || echo MISSING )"
echo "[Check] DATA  : $( [ -d "${DATA}" ] && echo ok || echo will-download )"

ARCH=${ARCH_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Assigned ARCH=${ARCH} (dropout=${DROPOUT}, wd=${WD})"

# unique per-arch sweep output (avoid collision)
echo "--- ARCH=${ARCH} ---"
JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/architectures/${ARCH}"
mkdir -p "${JOB_SWEEP_OUT_ARCH}"
for REP in "${REPEATS[@]}"; do
  echo "==== ARCH=${ARCH}, dropout=${DROPOUT}, wd=${WD}, repeat=${REP} ===="
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
echo "Done ARCH=${ARCH}, dropout=${DROPOUT}, wd=${WD} (all repeats)."
