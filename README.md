
# Does Compression Imply Generalization? A Minimum Description Length Perspective

## 1. Environment

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml     # creates env named `mdl`
conda activate mdl
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

Clone the `information-bottleneck/` repository (required for CIFAR-10 experiments):
```bash
git clone git@github.com:xu-ji/information-bottleneck.git
```

---

## 2.1 CIFAR-10: Kawaguchi et al. (ICML 2023) IB grid

Replays the 540-cell CIFAR-10 grid with **prequential MDL** code length as a
predictor of generalization gap:

```
3 archs (PreResNet56/83/110)  ×  3 wds (1e-5, 1e-4, 1e-3)
×  3 batch sizes (64, 128, 1024)  ×  4 seeds  ×  5 data instances  =  540 cells
```

Launch the full grid (cluster):

```bash
sbatch run_cifar_ib_grid_full.sbatch
```

Single-PC equivalent (runs all 540 cells sequentially on the local GPU,
optionally restricted to a task-id range):

```bash
bash run_cifar_ib_grid_full.sh             # all 540 cells
bash run_cifar_ib_grid_full.sh 0 99        # only task ids 0..99
```

Each cell writes to `runs_cifar_ib/<run_name>/summary.json`. Outputs include:
prequential bits, model code length per image (`D_bits_per_img`), final 
train/test accuracy, and generalization gaps. Existing `summary.json` files
are skipped, so re-running resumes after a Ctrl-C.

---

## 2.2 CIFAR-100: Architecture and regularization sweeps

Two complementary sweeps on CIFAR-100:

### 2.2.1 Architecture sweep (ResNet18/34/50/101/152)

Cluster:

```bash
sbatch run_arch_array_seqreps.sbatch
```

Single-PC equivalent (run all 5 archs sequentially, or one at a time):

```bash
bash run_arch_array_seqreps.sh             # all 5 archs
bash run_arch_array_seqreps.sh resnet18    # one arch only
```

Sweeps 5 ResNet architectures with 3 repeats each, fixed dropout=0.0 and wd=0.0.
Outputs to `runs_cifar100/` with per-architecture aggregation in
`runs_sweeps_cifar100/architectures/<arch>/`.

### 2.2.2 Weight decay & dropout sweeps (ResNet34)

Cluster:

```bash
sbatch run_wd_array_seqreps.sbatch
```

Single-PC equivalent:

```bash
bash run_wd_array_seqreps.sh
```

By default runs both:
- **Weight decay:** 10 values from 0.0 to 1e-2
- **Dropout:** 20 values from 0% to 50%

Control via `SWEEP_MODE` environment variable (works for both cluster and PC):

```bash
SWEEP_MODE=wd      sbatch run_wd_array_seqreps.sbatch     # weight decay only
SWEEP_MODE=dropout bash   run_wd_array_seqreps.sh         # dropout only
SWEEP_MODE=both    bash   run_wd_array_seqreps.sh         # both (default)
```

Outputs to `runs_cifar100/` with per-sweep aggregation in
`runs_sweeps_cifar100/<arch>/{wd_sweep,dropout_sweep}/`.

---

## 3.1 Compute correlation metrics CIFAR-10

```bash
conda run -n mdl python compute_metrics_cifar_ib.py \
  --runs_dir ./runs_cifar_ib \
  --out_csv  ./results/cifar_ib_correlations.csv \
  --out_json ./results/cifar_ib_correlations.json
```

Computes Pearson, Spearman, and Kendall correlations between `D_bits_per_img`
and generalization gap targets (loss gap and accuracy gap) with `train_acc > 0.85` filter.

---

## 3.2 Compute correlation metrics CIFAR-100

```bash
conda run -n mdl python compute_metrics_cifar_ib.py \
  --runs_dir ./runs_cifar100 \
  --out_csv  ./results/cifar100_correlations.csv \
  --out_json ./results/cifar100_correlations.json
```

Uses the same correlation metrics as CIFAR-10. Optionally pass `--no_filter` to
skip the `train_acc > 0.85` filter.
