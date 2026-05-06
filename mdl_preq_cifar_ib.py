#!/usr/bin/env python3
"""
Prequential MDL on Kawaguchi et al. (ICML 2023) CIFAR-10 setting.

What this script does, at a glance
----------------------------------
For a single (arch, wd, batch_size, seed, data_instance) cell of their grid:
  1. Build the same 30%-CIFAR-10 subsample they train on.
  2. Replay the Dawid "prequential" coding procedure on that subsample:
     - split the training set into a growing sequence of blocks D_1, D_2, ...
     - for each block i: train on the prefix D_{<i}, then use the resulting
       model to *code* (= evaluate CE on) the held-out block D_i, accumulating
       the bits into L_preq.
  3. Re-evaluate the final model on the full training set to get L_data and
     report L_model = L_preq - L_data as the prequential model code length.
  4. Pair (L_model, acc_gap) -- the two numbers we correlate against the
     paper's I(S; theta_l) to test whether MDL tracks generalization gap the
     way their information-bottleneck quantities do.

Recipe provenance (ported verbatim from information-bottleneck/dnn/swag_repo)
----------------------------------------------------------------------------
  - Architecture : PreResNet-{56, 83, 110}, imported directly from their repo
  - Data         : 30% random subset of CIFAR-10 train, drawn exactly as
                   gen_data.py does (set_seed(seed), then sequentially
                   np.random.choice for instances 0..4)
  - Augmentation : RandomCrop(32, pad=4) + HFlip + standard CIFAR normalize
  - Optimizer    : SGD(lr=0.1, momentum=0.9, weight_decay=wd)
  - LR schedule  : their piecewise shape (const -> linear decay to swa_lr ->
                   const at swa_lr), with swa_start=161/200 = 0.805 of epochs
                   and swa_lr=0.05.  We keep the SHAPE but scale the time
                   axis to each block's epoch budget.
  - No AMP, no label smoothing (matches their train.py).

Prequential layer on top
------------------------
  - Block schedule: cumulative prefix sizes that grow geometrically from
    `start_min` to n, in about `max_blocks` steps.
  - Size-aware epoch tiers for each block's training budget:
      prefix fraction r < 0.05 -> 40 epochs
                       < 0.20 -> 80
                       < 0.50 -> 120
                       else   -> 200   (matches their straight-training total)
  - Block 0 is coded with a uniform-over-classes prior (log2(10) bits per
    sample): no model has seen the data yet, so the encoder / decoder agree
    on the trivial uniform code.
  - Warm-start across blocks by default (the optimizer keeps training the
    same weights on a growing prefix).  `--cold_start` re-inits the model
    per block, which gives the textbook prequential construction but at
    ~40x the compute.
  - summary.json uses the same keys as mdl_preq_blockwise_test.py so the
    existing aggregate_from_runs.py and compute_table1_preq_metrics.py work
    unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# The PreResNet definitions live inside the cloned IB repo.  We import them
# in place rather than copying them so we stay bit-for-bit compatible with
# their models as a submodule.  sys.path hack keeps the script standalone.
IB_ROOT = Path(__file__).resolve().parent / "information-bottleneck"
if str(IB_ROOT) not in sys.path:
    sys.path.insert(0, str(IB_ROOT))
from dnn.swag_repo.swag.models.preresnet import (  # noqa: E402
    PreResNet56,
    PreResNet83,
    PreResNet110,
)

# Their PreResNetXX classes aren't nn.Modules; they're thin "spec" objects
# exposing `.base` (the real nn.Module class) plus `.args` / `.kwargs` that
# pin the depth.  Instantiation is spec.base(*spec.args, **spec.kwargs).
ARCH_LOOKUP = {
    "PreResNet56": PreResNet56,
    "PreResNet83": PreResNet83,
    "PreResNet110": PreResNet110,
}

# Constants from information-bottleneck/dnn/swag_repo/MI/make_scripts.py and
# experiments/train/train.py.  We re-use them literally so the trained model
# is the same one their Table 3 ran I(S; theta_l) on.
NUM_CLASSES = 10
IB_EPOCHS = 200        # their --epochs
IB_SWA_START = 161     # their --swa_start; SWA-collection begins at this epoch
IB_LR_INIT = 0.1       # their --lr_init
IB_SWA_LR = 0.05       # their --swa_lr (the floor of the LR schedule)
IB_MOMENTUM = 0.9      # their optimizer momentum
IB_SUBSIZE = 0.3       # their --subsize in gen_data.py
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def ib_set_seed(seed: int) -> None:
    """Verbatim port of IB's set_seed (dnn/swag_repo/MI/util.py).

    We call this exactly once, before drawing the data-instance subsample,
    so that the np.random.choice draws inside ib_instance_indices replay the
    same sequence their gen_data.py produced.  Any subsequent randomness
    (shuffles, model init) should be seeded *separately* -- see
    set_training_seed -- or the downstream RNG state will drift.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_training_seed(seed: int) -> None:
    """Seed only torch's RNGs for our training path.

    Deliberately does NOT touch numpy/random state, so calling this after
    ib_instance_indices doesn't disturb the subsample draw.  Used to make
    model init, DataLoader shuffling, and cold-start reseeding reproducible.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str) -> nn.Module:
    """Instantiate one of their PreResNet variants.

    Their wrapper-class pattern stores the real module class on `.base` and
    keeps the depth in `.kwargs["depth"]`.  For CIFAR we override num_classes
    to 10; everything else follows their default.
    """
    spec = ARCH_LOOKUP[arch]
    return spec.base(*spec.args, num_classes=NUM_CLASSES, **spec.kwargs)


def get_cifar_datasets(data_dir: str):
    """Build three CIFAR-10 views sharing one set of underlying images.

    We need two distinct views of the TRAIN split because:
      - training passes must see augmented images (random crop + hflip);
      - per-block *evaluation* of bits must be deterministic and therefore
        must NOT augment, otherwise L_preq measures the augmentation noise
        instead of the model's predictive code.
    The test split is used only for final accuracy / bit reporting.
    """
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_aug = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_train)
    train_eval = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=tf_eval)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_eval)
    return train_aug, train_eval, test


def ib_instance_indices(seed: int, data_instance: int, n: int,
                        num_datasets: int = 5, subsize: float = IB_SUBSIZE) -> np.ndarray:
    """Reproduce the indices that gen_data.py saves as instance `data_instance`.

    In their pipeline a "(seed, instance)" pair is a 30%-subset of CIFAR-10
    train, and its identity is defined by the RNG consumption order inside
    gen_data.py:

        set_seed(seed)
        for i in range(num_datasets):        # num_datasets = 5
            idx_i = np.random.choice(n, size=int(0.3 * n), replace=False)
            save(idx_i as instance i)

    To hand back instance `data_instance` bit-for-bit, we have to consume
    the RNG through instances 0..(data_instance - 1) first and only keep
    the draw that matches.  (Skipping ahead by seeding differently would
    produce a different subset of images.)
    """
    assert 0 <= data_instance < num_datasets
    ib_set_seed(seed)
    k = int(subsize * n)
    idx = None
    for i in range(num_datasets):
        cur = np.random.choice(n, size=k, replace=False)
        if i == data_instance:
            idx = cur
    assert idx is not None
    return idx


def make_timesteps(
    n: int,
    start_frac: float = 0.005,
    start_min: int = 100,
    max_blocks: int = 40,
    factor: Optional[float] = None,
) -> List[int]:
    """Cumulative prefix sizes for the prequential block schedule.

    We want block sizes that grow geometrically, so early (information-rich)
    blocks are small and later (marginal) blocks are large.  The schedule is:

        |D_1| = max(start_min, round(start_frac * n))
        |D_{i+1}| / |D_i| ~= factor       (chosen so we reach n in ~max_blocks
                                           steps when factor is not given)
        final step always equals n

    The returned list is STRICTLY INCREASING and always ends at n, so blocks
    are formed as (0, steps[0]), (steps[0], steps[1]), ..., (steps[-2], n).

    Copied verbatim from mdl_preq_blockwise_test.py so the two scripts
    partition D the same way for a given (start_frac, start_min, max_blocks).
    """
    if n <= 0:
        return []
    start = max(1, min(n, int(round(start_frac * n))))
    start = max(start, min(n, start_min))
    steps = [start]
    if steps[0] == n:
        return steps
    if factor is None:
        K = max(2, max_blocks)
        factor = (n / steps[0]) ** (1.0 / (K - 1))
    while steps[-1] < n:
        nxt = int(round(steps[-1] * factor))
        if nxt <= steps[-1]:
            nxt = steps[-1] + 1
        if nxt >= n:
            break
        steps.append(nxt)
    if steps[-1] != n:
        steps.append(n)
    out, seen = [], set()
    for s in steps:
        if 1 <= s <= n and s not in seen:
            out.append(s); seen.add(s)
    if out[-1] != n:
        out.append(n)
    return out


def epochs_for_prefix(prefix_size: int, n_total: int) -> int:
    """How many epochs to train on a given prefix.

    Rationale: at a tiny prefix (say 1% of n) the model overfits fast and
    more epochs mostly memorize noise; at a large prefix (>= half of n) we
    want the full IB 200-epoch budget so the final cell's model matches what
    their straight training would produce.  The tier cutoffs are the same
    ones we use on full-CIFAR in mdl_preq_blockwise_test.py.
    """
    r = prefix_size / n_total
    if r < 0.05:
        return 40
    if r < 0.20:
        return 80
    if r < 0.50:
        return 120
    return 200


def ib_lr_at_epoch(epoch: int, total_ep: int) -> float:
    """IB's piecewise LR schedule, scaled to `total_ep` epochs.

    Their train.py uses:

        t      = epoch / swa_start                     # 161 / 200 = 0.805
        factor = 1.0                          if t <= 0.5
                 1.0 - (1 - lr_ratio)*(t-0.5)/0.4      if 0.5 < t <= 0.9
                 lr_ratio                     otherwise
        lr     = lr_init * factor

    i.e. LR holds at lr_init for the first half of pre-SWA training, then
    decays linearly to swa_lr over the next 40%, then stays at swa_lr.

    When a block has fewer than IB_EPOCHS epochs we preserve this shape by
    replacing swa_start with total_ep * (IB_SWA_START / IB_EPOCHS).  That
    keeps each block in the same regime (warmup-flat / decay / floor) as a
    proportional slice of their original schedule.
    """
    swa_start_eff = max(1.0, total_ep * (IB_SWA_START / IB_EPOCHS))
    t = epoch / swa_start_eff
    lr_ratio = IB_SWA_LR / IB_LR_INIT
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return IB_LR_INIT * factor


def train_one_epoch(model, device, dataset, batch_size, optimizer, workers,
                    generator=None):
    """One pass over `dataset` with SGD + cross-entropy (mean reduction).

    Runs AUGMENTED training transforms (the dataset passed in should be the
    `train_aug` view).  Matches IB's train.py: no AMP, no label smoothing,
    mean-reduction cross-entropy.  `generator` makes the shuffle order
    reproducible from a per-block seed.
    """
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True,
                        generator=generator)
    crit = nn.CrossEntropyLoss(reduction="mean")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_block_bits(model, device, dataset, batch_size, workers) -> float:
    """Code length, in bits, that `model` assigns to `dataset`.

    In MDL terms this is a single block's contribution to L_preq.  For the
    block to be legitimately "coded" the model must not have seen it during
    training -- that is the caller's responsibility.  The returned value is
    the SUM of per-sample negative log2 probabilities: concretely we sum
    CE-in-nats (reduction="sum") and divide by ln(2) to convert to bits.
    Uses the un-augmented eval transforms so the number is a deterministic
    property of (model, block).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        total += crit(model(x), y).item()
    return total / math.log(2)


@torch.no_grad()
def evaluate_full(model, device, dataset, batch_size, workers):
    """Total bits AND 0/1 accuracy over a whole dataset.

    Used twice at the end of a run: on the full (ordered) training instance
    to get L_data, and on the CIFAR-10 test split to get test_bits / test_acc.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total / math.log(2), correct / n


def train_prefix(model, device, prefix_aug, base_ep: int, batch_size: int,
                 workers: int, wd: float, block_seed: int):
    """Train `model` on `prefix_aug` for `base_ep` epochs.

    Builds a fresh SGD optimizer every call.  The IB recipe specifies
    weight_decay and starts the LR schedule from lr_init, both of which are
    most naturally expressed by re-creating the optimizer at the top of each
    block rather than trying to rewind state on an existing one.  The
    per-epoch LR is set from ib_lr_at_epoch so the schedule's shape rescales
    cleanly to `base_ep`.
    """
    optimizer = optim.SGD(model.parameters(), lr=IB_LR_INIT,
                          momentum=IB_MOMENTUM, weight_decay=wd)
    for ep in range(base_ep):
        lr = ib_lr_at_epoch(ep, base_ep)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        gen = torch.Generator().manual_seed((block_seed * 1_000_003 + ep * 1_009) & 0xFFFFFFFF)
        train_one_epoch(model, device, prefix_aug, batch_size, optimizer,
                        workers, generator=gen)


def main():
    ap = argparse.ArgumentParser()

    # paths / naming
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./runs_cifar_ib")
    ap.add_argument("--run_name", type=str, required=True)

    # IB grid coordinates -- these five fields identify one cell of the
    # 4 seeds x 5 instances x 3 bs x 3 wd x 3 arch = 540-cell grid.
    ap.add_argument("--arch", type=str, required=True, choices=list(ARCH_LOOKUP))
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="their `seed`; controls subsample order + model init")
    ap.add_argument("--data_instance", type=int, required=True,
                    help="their instance id in [0, num_datasets)")
    ap.add_argument("--num_datasets", type=int, default=5,
                    help="matches gen_data.py default")
    ap.add_argument("--subsize", type=float, default=IB_SUBSIZE)

    # prequential knobs (mirror mdl_preq_blockwise_test.py)
    ap.add_argument("--start_frac", type=float, default=0.005)
    ap.add_argument("--start_min", type=int, default=100)
    ap.add_argument("--max_blocks", type=int, default=40)
    ap.add_argument("--factor", type=float, default=None)

    # prequential policy switches
    ap.add_argument("--cold_start", action="store_true",
                    help="re-initialize model per block (default: warm-start)")
    ap.add_argument("--workers", type=int, default=4)

    # unit-test knobs -- see run_cifar_ib_smoke.sbatch for the testing ladder.
    ap.add_argument("--smoke_test", action="store_true",
                    help="build model + 1 minibatch forward/backward + exit")
    ap.add_argument("--quick_blocks", type=int, default=0,
                    help="if >0, override max_blocks to this (for fast e2e test)")
    ap.add_argument("--quick_epochs", type=int, default=0,
                    help="if >0, cap every block's epochs to this (for fast e2e test)")

    args = ap.parse_args()

    run_dir = os.path.join(args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] {vars(args)}")
    print(f"[cfg] device={device}")

    # ---- Data ---------------------------------------------------------
    # Fetch CIFAR-10 (aug/eval/test views) once, then pick the subset of
    # indices that identifies "their instance" for the given seed.  All
    # downstream operations index into ds_train_aug / ds_train_eval through
    # these indices.
    ds_train_aug, ds_train_eval, ds_test = get_cifar_datasets(args.data_dir)
    n_full = len(ds_train_aug)
    inst_idx = ib_instance_indices(args.seed, args.data_instance, n_full,
                                   num_datasets=args.num_datasets,
                                   subsize=args.subsize)
    n = len(inst_idx)
    print(f"[data] CIFAR10 seed={args.seed} inst={args.data_instance} "
          f"n_full={n_full} n_instance={n}  sample_idx[:5]={inst_idx[:5].tolist()}")

    # ---- Unit A: smoke test ------------------------------------------
    # Verify wiring only (import, model build, one fwd/bwd on real data).
    # Intentionally does not write summary.json, so it can't be mistaken
    # for a real run's output.
    if args.smoke_test:
        set_training_seed(args.seed)
        model = build_model(args.arch).to(device)
        model.train()
        sub = Subset(ds_train_aug, inst_idx[:args.batch_size].tolist())
        loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
        opt = optim.SGD(model.parameters(), lr=IB_LR_INIT,
                        momentum=IB_MOMENTUM, weight_decay=args.wd)
        crit = nn.CrossEntropyLoss()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"[smoke] 1 step: logits={tuple(logits.shape)} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.2f}s")
            break
        print("[smoke] ok")
        return

    # ---- Ordering within the instance --------------------------------
    # `inst_idx` defines *which* 15k CIFAR images belong to this cell; the
    # prequential construction also cares about the ORDER in which those
    # images are revealed.  We fix a training-order permutation from a seed
    # that's deterministic in (args.seed, args.data_instance) and save it so
    # the per-block composition is fully reproducible.
    derived_seed = args.seed * 1_000 + args.data_instance * 7 + 1
    set_training_seed(derived_seed)
    perm = torch.randperm(n).tolist()
    ordered_idx = inst_idx[np.array(perm)]
    torch.save(torch.tensor(ordered_idx), os.path.join(run_dir, "order.pt"))

    # ---- Block schedule ----------------------------------------------
    # max_blocks is overridden by --quick_blocks for fast e2e tests; in a
    # real run we hit the default 40-block geometric schedule.
    max_blocks = args.quick_blocks if args.quick_blocks > 0 else args.max_blocks
    steps = make_timesteps(n, start_frac=args.start_frac,
                           start_min=args.start_min,
                           max_blocks=max_blocks, factor=args.factor)
    blocks = [(0, steps[0])] + [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
    print(f"[preq] n={n}  blocks({len(steps)})={steps[:6]} ... {steps[-3:]}")

    # ---- Block 0: uniform code ---------------------------------------
    # Before any training data is observed, encoder / decoder share only
    # the class prior, which we take to be uniform over K classes.  That
    # makes block 0 cost |D_0| * log2(K) bits, independent of the model.
    # Everything after block 0 is "model-coded".
    csv_path = os.path.join(run_dir, "blocks.csv")
    with open(csv_path, "w") as f:
        f.write("block_id,start,end,block_size,epochs,bits_block,cum_bits,elapsed_s\n")

    preq_bits_total = 0.0
    b0_start, b0_end = blocks[0]
    b0_size = b0_end - b0_start
    uniform_block0_bits = b0_size * math.log2(NUM_CLASSES)
    preq_bits_total += uniform_block0_bits
    with open(csv_path, "a") as f:
        f.write(f"0,{b0_start},{b0_end},{b0_size},0,"
                f"{uniform_block0_bits:.6f},{preq_bits_total:.6f},0.0\n")
    print(f"[preq] block 0: size={b0_size} uniform_bits={uniform_block0_bits:.2f}")

    # ---- Model (warm-start path) -------------------------------------
    # In warm-start mode the model is instantiated once and carried across
    # all blocks: each block's training simply continues from the previous
    # block's end state on a strictly larger prefix.  Cold-start re-builds
    # the model inside the block loop.
    model: Optional[nn.Module] = None
    if not args.cold_start:
        set_training_seed(derived_seed)
        model = build_model(args.arch).to(device)

    # ---- Prequential loop -------------------------------------------------
    # For each i >= 1:
    #   (a) train on prefix D_{<i} with the IB recipe (`train_prefix`);
    #   (b) evaluate CE-in-bits on the held-out block D_i (`eval_block_bits`);
    #   (c) add that to the running L_preq.
    # The model NEVER sees D_i before it is coded in step (b); that is what
    # makes the running sum a valid prequential code length.
    t_run = time.time()
    for bid, (start, end) in enumerate(blocks[1:], start=1):
        prefix_idx = ordered_idx[:start].tolist()
        block_idx = ordered_idx[start:end].tolist()
        prefix_aug = Subset(ds_train_aug, prefix_idx)
        block_eval = Subset(ds_train_eval, block_idx)

        # Cold-start: each block gets a freshly-initialized model seeded
        # from (derived_seed, bid) so the prequential code corresponds to
        # the textbook "retrain from scratch on every prefix" construction.
        if args.cold_start:
            set_training_seed(derived_seed * 10_007 + bid)
            model = build_model(args.arch).to(device)

        base_ep = epochs_for_prefix(start, n)
        if args.quick_epochs > 0:
            base_ep = min(base_ep, args.quick_epochs)

        t0 = time.time()
        train_prefix(model, device, prefix_aug, base_ep, args.batch_size,
                     args.workers, args.wd, block_seed=derived_seed + bid)
        bits = eval_block_bits(model, device, block_eval, args.batch_size, args.workers)
        preq_bits_total += bits
        dt = time.time() - t0

        with open(csv_path, "a") as f:
            f.write(f"{bid},{start},{end},{end - start},{base_ep},"
                    f"{bits:.6f},{preq_bits_total:.6f},{dt:.2f}\n")
        print(f"[preq] block {bid}: prefix={start} block_size={end - start} "
              f"epochs={base_ep} bits={bits:.2f} cum={preq_bits_total:.2f} "
              f"elapsed={dt:.1f}s")

    # ---- Final evaluations ------------------------------------------
    # L_data = code length of ALL training samples under the final model;
    # acc_gap comes from pairing this against the held-out CIFAR test split.
    # In warm-start mode `model` is the state after training on the full n
    # (the last prefix equals n); in cold-start mode it is the last block's
    # freshly-retrained model.  Either way we evaluate the exact same model
    # that produced the last contribution to L_preq.
    full_train_eval = Subset(ds_train_eval, ordered_idx.tolist())
    train_bits, train_acc = evaluate_full(model, device, full_train_eval,
                                          args.batch_size, args.workers)
    test_bits, test_acc = evaluate_full(model, device, ds_test,
                                        args.batch_size, args.workers)
    total_wall = time.time() - t_run

    # ---- MDL decomposition ------------------------------------------
    # L_preq  = total bits the prequential procedure spent coding D.
    # L_data  = total bits the FINAL model assigns to D when it can see D.
    # D_bits  = L_preq - L_data: our estimator of the "model code length",
    #           i.e. the extra bits the prequential protocol pays on top of
    #           the final model's own data fit.  This is what we correlate
    #           with the paper's I(S; theta_l).
    # D_bits_star subtracts the fixed uniform-block-0 cost so the resulting
    #           number reflects only the model-coded portion of L_preq.
    L_preq = preq_bits_total
    L_data = train_bits
    D_bits = L_preq - L_data
    D_bits_star = (L_preq - uniform_block0_bits) - L_data

    summary = {
        # IB grid coords (so aggregation can group by them)
        "arch": args.arch,
        "wd": args.wd,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_instance": args.data_instance,

        # prequential knobs
        "cold_start": bool(args.cold_start),
        "timesteps": steps,
        "max_blocks_effective": max_blocks,
        "quick_epochs_cap": args.quick_epochs,

        # compute / data size
        "total_train_samples": n,

        # MDL quantities -- same keys as mdl_preq_blockwise_test.py so
        # aggregate_from_runs.py and compute_table1_preq_metrics.py accept
        # this summary.json unmodified.
        "uniform_block0_bits": uniform_block0_bits,
        "prequential_bits": L_preq,
        "prequential_bits_per_img": L_preq / n,
        "final_train_bits": L_data,
        "final_train_bits_per_img": L_data / n,
        "D_bits": D_bits,
        "D_bits_per_img": D_bits / n,
        "D_bits_star": D_bits_star,
        "D_bits_star_per_img": D_bits_star / n,
        "test_bits": test_bits,
        "test_bits_per_img": test_bits / len(ds_test),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "acc_gap": train_acc - test_acc,
        "loss_gap_bits_per_sample": (test_bits / len(ds_test)) - (train_bits / n),

        # recipe provenance (for reviewers)
        "optimizer": "SGD",
        "momentum": IB_MOMENTUM,
        "lr_init": IB_LR_INIT,
        "swa_lr": IB_SWA_LR,
        "swa_start_frac": IB_SWA_START / IB_EPOCHS,
        "ib_subsize": args.subsize,

        # wallclock
        "wall_s": total_wall,
    }

    print("==== Summary ====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
