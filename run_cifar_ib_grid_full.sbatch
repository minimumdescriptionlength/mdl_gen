#!/bin/bash
#SBATCH -J cifar_ib_full
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-08:00:00
#SBATCH --mem=8G
#SBATCH --output=logs_cifar_ib/%x_%A_%a.out
#SBATCH --error=logs_cifar_ib/%x_%A_%a.err
#SBATCH --array=0-539%20

# -----------------------------------------------------------------------------
# Full 540-cell CIFAR-10 grid from Kawaguchi et al. (ICML 2023), combining
# what was previously phase 1 (108 cells) and phase 2 (432 cells) into a
# single sbatch.
#
# Grid: 3 archs x 3 wds x 3 batch sizes x 4 seeds x 5 data_instances = 540.
#
# Each task is independent, writes its own runs_cifar_ib/<name>/summary.json,
# and is SKIPPED if that summary.json already exists -- so re-running this
# sbatch after a partial failure (or after the previous phase-1/phase-2
# split) resumes cleanly without recomputing finished cells.
#
# Throttle %20 limits simultaneous tasks to 20.  Bump it on idle queue:
#   scontrol update job=<ARRAY_ID> ArrayTaskThrottle=28
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

echo "Host: $(hostname)  task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
date
nvidia-smi -L || true

# ---- grid axes (verbatim from information-bottleneck/MI/make_scripts.py) ----
ARCHS=(PreResNet56 PreResNet83 PreResNet110)
WDS=(1e-5 1e-4 1e-3)
BSS=(64 128 1024)
SEEDS=(0 1 2 3)
INSTS=(0 1 2 3 4)

# ---- decode SLURM_ARRAY_TASK_ID into (seed, inst, arch, wd, bs) ------------
# Factor order (innermost -> outermost): bs, wd, arch, inst, seed.
# Sizes:                                 3,  3,   3,   5,    4   -> 540.
TID=${SLURM_ARRAY_TASK_ID}
BS_I=$((TID % 3));    TID=$((TID / 3))
WD_I=$((TID % 3));    TID=$((TID / 3))
ARCH_I=$((TID % 3));  TID=$((TID / 3))
INST_I=$((TID % 5));  TID=$((TID / 5))
SEED_I=$((TID % 4))

ARCH=${ARCHS[${ARCH_I}]}
WD=${WDS[${WD_I}]}
BS=${BSS[${BS_I}]}
SEED=${SEEDS[${SEED_I}]}
INST=${INSTS[${INST_I}]}

RUN_NAME="grid_${ARCH}_wd${WD}_bs${BS}_seed${SEED}_inst${INST}"
SUMM="${OUT}/${RUN_NAME}/summary.json"

echo ""
echo "=========================================="
echo "Cell: arch=${ARCH} wd=${WD} bs=${BS} seed=${SEED} inst=${INST}"
echo "Run : ${RUN_NAME}"
echo "=========================================="

if [[ -f "${SUMM}" ]]; then
  echo "[skip] ${SUMM} already exists -- this cell was completed in an earlier run."
  exit 0
fi

t0=$(date +%s)
python "${SCRIPT}" \
  --data_dir "${DATA}" --out "${OUT}" \
  --run_name "${RUN_NAME}" \
  --arch "${ARCH}" --wd "${WD}" --batch_size "${BS}" \
  --seed "${SEED}" --data_instance "${INST}" \
  --workers 4
t1=$(date +%s)
echo "Cell ${RUN_NAME} wallclock: $((t1 - t0))s  (~$(( (t1 - t0) / 60 )) min)"
echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
