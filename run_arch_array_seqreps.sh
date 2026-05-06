#!/bin/bash
# -----------------------------------------------------------------------------
# Single-PC equivalent of run_arch_array_seqreps.sbatch.
# Sweeps 5 ResNet architectures x 3 repeats on CIFAR-100 sequentially.
#
# Usage:
#   bash run_arch_array_seqreps.sh                 # run all 5 archs
#   bash run_arch_array_seqreps.sh resnet18        # one arch only
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# --- conda env ---
set +u
source ~/.bashrc || true
set -u
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
LABEL_SMOOTH=0.1
DROPOUT=0.0
WD=0.0

# --- sweep lists (override first arg to run a single arch) ---
if [[ $# -ge 1 ]]; then
  ARCH_LIST=("$1")
else
  ARCH_LIST=(resnet18 resnet34 resnet50 resnet101 resnet152)
fi
REPEATS=(0 1 2)

mkdir -p logs "${OUT}" "${SWEEP_OUT}"
echo "Host: $(hostname)"; date
nvidia-smi -L || true

for ARCH in "${ARCH_LIST[@]}"; do
  echo "--- ARCH=${ARCH} (dropout=${DROPOUT}, wd=${WD}) ---"
  JOB_SWEEP_OUT_ARCH="${SWEEP_OUT}/architectures/${ARCH}"
  mkdir -p "${JOB_SWEEP_OUT_ARCH}"
  for REP in "${REPEATS[@]}"; do
    echo "==== ARCH=${ARCH}, repeat=${REP} ===="
    python "${SWEEP}" \
      --script "${SCRIPT}" \
      --data "${DATA}" --out "${OUT}" --sweep_out "${JOB_SWEEP_OUT_ARCH}" \
      --wds ${WD} \
      --base_seed ${BASE_SEED} --repeat_iter ${REP} \
      --batch_size ${BATCH_SIZE} --workers ${WORKERS} \
      --optimizer sgd --lr ${LR} --label_smoothing ${LABEL_SMOOTH} \
      --min_lr_factor ${MIN_LR_FACTOR} --arch ${ARCH} --dropout ${DROPOUT} --quiet \
      2>&1 | tee "logs/arch_${ARCH}_rep${REP}.log"
  done
  echo "Done ARCH=${ARCH} (all repeats)."
done
echo "All tasks done at $(date)"
