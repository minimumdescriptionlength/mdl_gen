#!/bin/bash
# -----------------------------------------------------------------------------
# Single-PC equivalent of run_cifar_ib_grid_full.sbatch.
# Iterates the same 540-cell CIFAR-10 IB grid sequentially on the local GPU.
# Skips cells whose summary.json already exists, so you can Ctrl-C and resume.
#
# Usage:
#   bash run_cifar_ib_grid_full.sh                # run all 540 cells in order
#   bash run_cifar_ib_grid_full.sh 0 99           # only run task ids 0..99
#   bash run_cifar_ib_grid_full.sh 42 42          # only run task id 42
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

SCRIPT="./mdl_preq_cifar_ib.py"
DATA="./data"
OUT="./runs_cifar_ib"
mkdir -p logs_cifar_ib "${OUT}"

# ---- grid axes (verbatim from run_cifar_ib_grid_full.sbatch) ---------------
ARCHS=(PreResNet56 PreResNet83 PreResNet110)
WDS=(1e-5 1e-4 1e-3)
BSS=(64 128 1024)
SEEDS=(0 1 2 3)
INSTS=(0 1 2 3 4)

START_TID=${1:-0}
END_TID=${2:-539}

echo "Host: $(hostname)  range=[${START_TID}..${END_TID}]"
date
nvidia-smi -L || true

for TID in $(seq "${START_TID}" "${END_TID}"); do
  T=${TID}
  BS_I=$((T % 3));    T=$((T / 3))
  WD_I=$((T % 3));    T=$((T / 3))
  ARCH_I=$((T % 3));  T=$((T / 3))
  INST_I=$((T % 5));  T=$((T / 5))
  SEED_I=$((T % 4))

  ARCH=${ARCHS[${ARCH_I}]}
  WD=${WDS[${WD_I}]}
  BS=${BSS[${BS_I}]}
  SEED=${SEEDS[${SEED_I}]}
  INST=${INSTS[${INST_I}]}

  RUN_NAME="grid_${ARCH}_wd${WD}_bs${BS}_seed${SEED}_inst${INST}"
  SUMM="${OUT}/${RUN_NAME}/summary.json"

  echo ""
  echo "=========================================="
  echo "Task ${TID}/539  cell: arch=${ARCH} wd=${WD} bs=${BS} seed=${SEED} inst=${INST}"
  echo "Run : ${RUN_NAME}"
  echo "=========================================="

  if [[ -f "${SUMM}" ]]; then
    echo "[skip] ${SUMM} already exists."
    continue
  fi

  t0=$(date +%s)
  python "${SCRIPT}" \
    --data_dir "${DATA}" --out "${OUT}" \
    --run_name "${RUN_NAME}" \
    --arch "${ARCH}" --wd "${WD}" --batch_size "${BS}" \
    --seed "${SEED}" --data_instance "${INST}" \
    --workers 4 \
    2>&1 | tee "logs_cifar_ib/${RUN_NAME}.log"
  t1=$(date +%s)
  echo "Cell ${RUN_NAME} wallclock: $((t1 - t0))s  (~$(( (t1 - t0) / 60 )) min)"
done

echo "All tasks done at $(date)"
