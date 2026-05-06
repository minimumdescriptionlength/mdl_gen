#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, sys
from typing import List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# SWAG imports for MI_theta computation
# NOTE: Moved to lazy import in main() to avoid hanging on module-level initialization
has_swag = False

# --------------------- utils ---------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def tee_stdout(log_path: str):
    class Tee(object):
        def __init__(self, fname):
            self.file = open(fname, "a")
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    sys.stdout = Tee(log_path)

def seed_worker(worker_id: int):
    """Deterministic NumPy/Python RNGs inside DataLoader workers."""
    wseed = torch.initial_seed() % 2**32
    np.random.seed(wseed)
    random.seed(wseed)

def mix_seed(seed: int, bid: int) -> int:
    """Per-block 32-bit seed (keeps each block’s shuffle/aug reproducible)."""
    return (seed * 1_000_003 + bid * 1_009) & 0xFFFFFFFF

# --------------------- data & model ---------------------

def get_datasets(data_dir: str, dataset: str = "cifar10"):
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        tf_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_aug = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_train)
        train_eval = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=tf_eval)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_eval)
    elif dataset == "cifar100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        tf_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_aug = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=tf_train)
        train_eval = torchvision.datasets.CIFAR100(data_dir, train=True, download=False, transform=tf_eval)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=tf_eval)
    else:  # mnist
        mean = (0.1307,)
        std = (0.3081,)
        tf_train = T.Compose([
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_aug = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tf_train)
        train_eval = torchvision.datasets.MNIST(data_dir, train=True, download=False, transform=tf_eval)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf_eval)
    return train_aug, train_eval, test

def add_layer_index_to_modules(model: nn.Module) -> nn.Module:
    """Add layer_index attribute to all modules (required by SWAG for per-layer operations)."""
    layer_mapping = {
        'conv1': 0,      # initial conv layer
        'layer1': 1,     # residual layer 1
        'layer2': 2,     # residual layer 2
        'layer3': 3,     # residual layer 3
        'layer4': 4,     # residual layer 4
        'fc': 5,         # final classification layer
    }
    
    for name, module in model.named_modules():
        # Extract the main layer name (first component before '.')
        main_layer = name.split('.')[0] if '.' in name else name
        
        if main_layer in layer_mapping:
            module.layer_index = layer_mapping[main_layer]
        elif not hasattr(module, 'layer_index'):
            module.layer_index = 0  # Default to layer 0
    
    return model

def subsample_indices(n: int, frac: float, subsample_seed: int) -> List[int]:
    """
    Deterministic training-set subsample. Given total size `n`, fraction
    `frac in (0, 1]`, and integer `subsample_seed`, returns a list of
    `round(frac * n)` indices drawn without replacement.

    The SAME function is used by the prequential script and train_normal.py,
    so a `(subsample_frac, subsample_seed)` pair identifies exactly the
    same subset S_k across the two pipelines - enabling the paired
    per-S_k comparison between MDL and IB bounds.
    """
    if frac >= 1.0:
        return list(range(n))
    m = max(1, int(round(frac * n)))
    g = torch.Generator().manual_seed(int(subsample_seed))
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:m]


class SWAGBaseResNet(nn.Module):
    """
    Pickle-safe top-level wrapper used as the `base` class for SWAG. SWAG
    constructs `self.base = base(**kwargs)`, so passing this class plus
    kwargs (`arch=..., dataset=..., dropout=...`) yields a CIFAR-ready
    ResNet with `layer_index` attributes already set by `build_model`.
    Defined at module level so `torch.save` / `torch.load` can round-trip
    the SWAG object (function-local classes fail to unpickle).
    """

    def __init__(self, arch: str, dataset: str, dropout: float):
        super().__init__()
        self.base = build_model(arch, dataset, dropout)

    def forward(self, x):
        return self.base(x)


def build_model(arch: str = "resnet18", dataset: str = "cifar10", dropout_rate: float = 0.0):
    """Build a ResNet model with optional dropout. 
    
    Args:
        arch: 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'
        dataset: 'cifar10', 'cifar100', or 'mnist'
        dropout_rate: Dropout probability (0.0 = no dropout)
    """
    num_classes = 10  if dataset in ["cifar10", "mnist"] else 100
    in_channels = 1 if dataset == "mnist" else 3
    
    # Build base model based on architecture
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif arch == "resnet34":
        m = torchvision.models.resnet34(weights=None, num_classes=num_classes)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif arch == "resnet101":
        m = torchvision.models.resnet101(weights=None, num_classes=num_classes)
    elif arch == "resnet152":
        m = torchvision.models.resnet152(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    
    # Add layer_index to all modules (REQUIRED for SWAG compatibility)
    m = add_layer_index_to_modules(m)
    
    # Add dropout to residual blocks if dropout_rate > 0
    if dropout_rate > 0:
        # Insert dropout layers into residual blocks
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(m, layer_name)
            for block in layer:
                # Add dropout before the final activation of each residual block
                # Insert after conv layers but before ReLU in the identity path doesn't make sense
                # So we add it as a separate layer after the block
                original_forward = block.forward
                
                def make_forward_with_dropout(original_fn, dropout_p):
                    def forward_with_dropout(x):
                        x = original_fn(x)
                        x = nn.functional.dropout(x, p=dropout_p, training=True)
                        return x
                    return forward_with_dropout
                
                block.forward = make_forward_with_dropout(original_forward, dropout_rate)
    
    return m

# --------------------- step schedule ---------------------

# In order to calculate L(D, M), we need to break D into blocks D_1, D_2, ..., and feed each block to train the NN one by one
# This function controls the sizes of these blocks
def make_timesteps(
    n: int,
    start_frac: float = 0.005,         # start at ~0.5% of n
    start_min: int = 100,            # but at least 1k samples
    max_blocks: int = 40,             # aim for ≤12 cumulative steps incl. final n
    factor: Optional[float] = None    # if provided, use this geometric factor |D_2| = |D_1| * (factor + 1)
) -> List[int]:
    """
    Return cumulative prefix sizes (strictly increasing) ending at n.
    Strategy:
      - start = max(round(start_frac * n), start_min)
      - grow by a geometric factor until reaching n
      - if 'factor' is None, choose it so we reach n in ~max_blocks steps
      - always include 'n' as the last step
    """
    if n <= 0:
        return []
    start = max(1, min(n, int(round(start_frac * n))))
    start = max(start, min(n, start_min))
    steps = [start]
    if steps[0] == n:
        return steps

    if factor is None:
        K = max(2, max_blocks)  # at least [start, n]
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

    # de-dup + clamp
    out, seen = [], set()
    for s in steps:
        if 1 <= s <= n and s not in seen:
            out.append(s); seen.add(s)
    if out[-1] != n:
        out.append(n)
    return out

def epochs_for_prefix(prefix_size: int, n_total: int) -> int:
    """Fewer epochs early; ≥100 later. Tweak thresholds as desired."""
    r = prefix_size / n_total
    if r < 0.05:   return 40
    if r < 0.20:   return 80
    if r < 0.50:   return 120
    return 200

# --------------------- train/eval ---------------------

class CESmooth(nn.Module):
    """Cross-entropy with optional label smoothing (0 = standard CE)."""
    def __init__(self, smoothing: float = 0.0, reduction: str = "sum"):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits, target):
        if self.smoothing <= 0.0:
            return nn.functional.cross_entropy(logits, target, reduction=self.reduction)
        n_class = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-true_dist * log_probs).sum(dim=-1)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss

def build_scheduler(optimizer, total_steps: int, eta_min: float,
                    warmup_frac: float, warmup_start_factor: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = int(round(total_steps * warmup_frac))
    # ensure at least 1 cosine step if warmup is used
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)

    if warmup_steps > 0:
        sched1 = LinearLR(optimizer,
                          start_factor=warmup_start_factor,
                          end_factor=1.0,
                          total_iters=warmup_steps)
        sched2 = CosineAnnealingLR(optimizer,
                                   T_max=max(1, total_steps - warmup_steps),
                                   eta_min=eta_min)
        return SequentialLR(optimizer, [sched1, sched2], milestones=[warmup_steps])
    else:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

def train_one_epoch(model: nn.Module, device, dataset: Subset, batch_size: int,
                    optimizer, scaler, workers: int,
                    generator: torch.Generator | None = None,
                    scheduler = None, crit: nn.Module | None = None):
    model.train()
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=True,
                        generator=generator,
                        worker_init_fn=seed_worker)
    crit = crit or CESmooth(smoothing=0.0, reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Use AMP but ensure input and model stay float32
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float32):
            logits = model(x)
            loss = crit(logits, y) / y.size(0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()  # per-batch stepping

@torch.no_grad()
def eval_block_bits(model: nn.Module, device, dataset: Subset, batch_size: int, workers: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        total += crit(model(x), y).item()
    return total / math.log(2)

@torch.no_grad()
def evaluate_full(model: nn.Module, device, dataset: Subset, batch_size: int, workers: int):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    bits = total / math.log(2)
    acc = correct / n
    return bits, acc

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./runs")
    ap.add_argument("--run_name", type=str, default="blockwise_no_select")
    ap.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "mnist"], help="dataset to use")
    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], help="model architecture")

    ap.add_argument("--base_seed", type=int, default=0, help="base seed; final seed = base_seed + repeat_iter")
    ap.add_argument("--repeat_iter", type=int, default=0, help="repeat index controlling data order")

    # Training-set subsampling, for the paired MDL-vs-IB protocol.
    # When --subsample_frac < 1.0, the prequential training operates on a
    # random subset S_k of the full CIFAR training set - matching what
    # train_normal.py does for the same (subsample_frac, subsample_seed).
    # This lets each S_k produce one (MDL bound, preq gen error) pair here
    # and one (IB posterior, non-preq gen error) pair there, both measured
    # on the SAME data for direct comparison.
    ap.add_argument("--subsample_frac", type=float, default=1.0,
                    help="fraction of training set to use (1.0 = full; 0.3 matches paper's Table-2 setting)")
    ap.add_argument("--subsample_seed", type=int, default=0,
                    help="seed controlling which subsample is drawn (independent of model seed)")

    # Epoch policy
    ap.add_argument("--fixed_epochs", type=int, default=0, help="if >0, train exactly this many epochs per block")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)

    # Optimizer/schedule
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout rate (0.0 = no dropout)")

    ap.add_argument("--min_lr_factor", type=float, default=1e-2, help="eta_min = lr * min_lr_factor")
    ap.add_argument("--warmup_frac", type=float, default=0.05, help="fraction of steps for linear warmup")
    ap.add_argument("--warmup_start_factor", type=float, default=0.1, help="LR multiplier at start of warmup")

    # Step schedule knobs
    ap.add_argument("--start_frac", type=float, default=0.005)
    ap.add_argument("--start_min", type=int, default=100)
    ap.add_argument("--max_blocks", type=int, default=40)
    ap.add_argument("--factor", type=float, default=None)

    # SWAG arguments for MI_theta computation
    ap.add_argument(
        "--swa_start", type=int, default=100,
        help="epoch to start SWA/SWAG model collection (default: 100). "
             "Ignored when --cold_start is set; see --cold_swa_start_frac."
    )
    ap.add_argument(
        "--max_num_models", type=int, default=15,
        help="max number of models to collect in SWAG (default: 15)"
    )

    # Cold-start prequential training (ablation).
    # When set, model and optimiser are re-initialised from a deterministic
    # per-block seed at the start of every block, so theta_k is a function of
    # data only (no warm-start trajectory dependence). The final-block model is
    # essentially equivalent to normal SGD from scratch on the full training set.
    ap.add_argument(
        "--cold_start", action="store_true",
        help="re-initialise model and optimiser at the start of every block (default: warm start)"
    )
    ap.add_argument(
        "--cold_swa_start_frac", type=float, default=0.75,
        help="when --cold_start is set, begin SWAG collection after this fraction of the FINAL block's epochs (default: 0.75)"
    )

    args = ap.parse_args()

    # Prepare run dir & logging
    run_dir = os.path.join(args.out, args.run_name)
    ensure_dir(run_dir)
    tee_stdout(os.path.join(run_dir, "train.log"))

    # Seed & config
    seed = int(args.base_seed) + int(args.repeat_iter)
    print("Config:", {**vars(args), "derived_seed": seed})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "derived_seed": seed}, f, indent=2)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    ds_train_aug, ds_train_eval, ds_test = get_datasets(args.data, args.dataset)
    n_full = len(ds_train_aug)

    # Optional subsampling: when subsample_frac < 1.0, restrict the training
    # universe to a random subset S_k of the full dataset. perm_list ends up
    # containing ABSOLUTE indices into ds_train_aug (so Subset() keeps
    # working unchanged downstream), but drawn only from the subsample.
    sub_idx = subsample_indices(n_full, args.subsample_frac, args.subsample_seed)
    n = len(sub_idx)
    if args.subsample_frac < 1.0:
        print(f"[DATA] full={n_full}  frac={args.subsample_frac}  "
              f"subsample_seed={args.subsample_seed}  n_used={n}")
        torch.save({"indices": sub_idx, "subsample_seed": args.subsample_seed,
                    "subsample_frac": args.subsample_frac, "n_full": n_full},
                   os.path.join(run_dir, "subsample.pt"))

    # Fixed random order from derived seed, over the (possibly subsampled) universe
    g = torch.Generator().manual_seed(seed)
    perm_local = torch.randperm(n, generator=g).tolist()
    perm_list = [sub_idx[i] for i in perm_local]  # absolute indices into ds_train_aug
    torch.save(torch.tensor(perm_list), os.path.join(run_dir, "order.pt"))

    # Timesteps and blocks
    steps = make_timesteps(n, start_frac=args.start_frac, start_min=args.start_min,
                           max_blocks=args.max_blocks, factor=args.factor)
    print(f"Timesteps ({len(steps)}): {steps[:8]} ... {steps[-3:]}")
    blocks = [(0, steps[0])] + [(steps[i], steps[i+1]) for i in range(len(steps)-1)]

    # Model & optimizer
    model = build_model(args.arch, args.dataset, dropout_rate=args.dropout).to(device).float()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())
    crit = CESmooth(smoothing=args.label_smoothing, reduction="sum")

    # Initialize SWAG for MI_theta computation (lazy import to avoid hanging)
    # Try to import SWAG only when needed
    swag_model = None
    has_swag = False
    try:
        from dnn.swag_repo.swag.posteriors.swag import SWAG
        has_swag = True
        print("[SWAG] SWAG module imported successfully")
    except Exception as e:
        print(f"[WARNING] SWAG not available. MI_theta computation will not be possible.")
        print(f"[DEBUG] SWAG import error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        has_swag = False
    
    # Use the pickle-safe top-level SWAGBaseResNet class so a saved SWAG
    # object can round-trip torch.save/torch.load. (The legacy code here
    # used a function-local class, which is not importable at unpickle
    # time.) The current save code only emits state_dict, but keeping the
    # top-level class removes a latent foot-gun.
    if has_swag:
        try:
            swag_model = SWAG(
                SWAGBaseResNet,
                no_cov_mat=False,
                max_num_models=args.max_num_models,
                arch=args.arch,
                dataset=args.dataset,
                dropout=args.dropout,
            )
            swag_model.to(device)
            print(f"[SWAG] Initialized SWAG model for {args.arch} with layer_index support")
            print(f"[SWAG] Collecting up to {args.max_num_models} models after epoch {args.swa_start}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize SWAG: {e}")
            import traceback
            traceback.print_exc()
            swag_model = None

    # CSV header (no selection columns)
    csv_path = os.path.join(run_dir, "blocks.csv")
    with open(csv_path, "w") as f:
        f.write("block_id,start,end,block_size,epochs,bits_block,cum_bits\n")

    # Prequential accumulation
    K = 10
    preq_bits_total = 0.0
    
    # Track data partitions for MI_theta computation
    data_partitions = {}  # Will store block info for each block

    # calculate $L(M, D)$
    # Block 0: uniform
    b0_start, b0_end = blocks[0]
    b0_size = b0_end - b0_start
    uniform_block0_bits = b0_size * math.log2(K)
    preq_bits_total += uniform_block0_bits
    with open(csv_path, "a") as f:
        f.write(f"0,{b0_start},{b0_end},{b0_size},0,{uniform_block0_bits:.6f},{preq_bits_total:.6f}\n")
    print(f"Block 0: size={b0_size}, uniform bits={uniform_block0_bits:.2f}, cum={preq_bits_total:.2f}")
    
    data_partitions[0] = {
        "start": b0_start,
        "end": b0_end,
        "size": b0_size,
        "indices": [perm_list[i] for i in range(b0_end)]
    }

    # Prefix indices
    prefix_idx = [perm_list[i] for i in range(b0_end)]

    # Iterate remaining blocks (single path: NO e-loop / NO selection overhead)
    total_steps_last_block = 0  # for summary text
    eta_min_summary = 0.0
    global_epoch = 0  # Track global epoch count for SWAG collection
    
    n_train_blocks = len(blocks) - 1  # excludes block 0 (uniform)
    for bid, (start, end) in enumerate(blocks[1:], start=1):
        block_ids = [perm_list[i] for i in range(start, end)]
        block_eval = Subset(ds_train_eval, block_ids)
        prefix_aug = Subset(ds_train_aug, prefix_idx)

        # COLD-START: re-init model + optimiser from scratch with a deterministic
        # per-block seed before training on the expanded prefix.
        if args.cold_start:
            reinit_seed = mix_seed(seed, bid)
            set_seed(reinit_seed)
            model = build_model(args.arch, args.dataset, dropout_rate=args.dropout).to(device).float()
            if args.optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                      nesterov=True, weight_decay=args.wd)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())
            print(f"[COLD START] Block {bid}: re-initialised model/optimiser (reinit_seed={reinit_seed})")

        # epochs for this block
        base_ep = args.fixed_epochs if args.fixed_epochs > 0 else epochs_for_prefix(len(prefix_idx), n)
        base_ep = max(1, int(base_ep))


        # per-batch warmup + cosine over total_steps
        steps_per_epoch = max(1, math.ceil(len(prefix_aug) / args.batch_size))
        total_steps = steps_per_epoch * base_ep
        eta_min = args.lr * args.min_lr_factor
        scheduler = build_scheduler(optimizer, total_steps, eta_min, args.warmup_frac, args.warmup_start_factor)

        # deterministic generator for this block
        gen = torch.Generator().manual_seed(mix_seed(seed, bid))

        # SWAG collection threshold (per-block) for cold-start: last
        # (1 - cold_swa_start_frac) of the FINAL block's epochs. In warm-start,
        # we fall back to the global-epoch threshold.
        is_final_block = (bid == n_train_blocks)
        cold_swa_start_block_epoch = int(round(args.cold_swa_start_frac * base_ep))

        # train base_ep epochs on prefix
        for epoch_num in range(base_ep):
            global_epoch += 1
            block_epoch = epoch_num + 1

            train_one_epoch(model, device, prefix_aug, args.batch_size,
                            optimizer, scaler, args.workers,
                            generator=gen, scheduler=scheduler, crit=crit)

            # Collect SWAG checkpoint AFTER training if in SWA phase (for MI_theta posterior).
            # Warm start: threshold on global_epoch (existing behaviour).
            # Cold start: only the final block's trajectory is meaningful, so restrict
            # collection to that block and use a per-block threshold.
            if swag_model is not None:
                if args.cold_start:
                    should_collect = is_final_block and (block_epoch >= cold_swa_start_block_epoch)
                else:
                    should_collect = (global_epoch >= args.swa_start)
                if should_collect:
                    try:
                        import copy
                        model_copy = copy.deepcopy(model)
                        swag_model.collect_model(model_copy)
                        del model_copy

                        if args.cold_start:
                            if block_epoch == cold_swa_start_block_epoch:
                                print(f"[SWAG] Started collecting checkpoints in final block at block_epoch {block_epoch}")
                        else:
                            if global_epoch == args.swa_start:
                                print(f"[SWAG] Started collecting checkpoints at global_epoch {global_epoch}")
                    except Exception as e:
                        print(f"[SWAG] Warning at global_epoch {global_epoch}: Failed to collect model: {e}")

        # evaluate bits on this block
        bits_block = eval_block_bits(model, device, block_eval, args.batch_size, args.workers)
        preq_bits_total += bits_block

        with open(csv_path, "a") as f:
            f.write(f"{bid},{start},{end},{end-start},{base_ep},{bits_block:.6f},{preq_bits_total:.6f}\n")
        print(f"Block {bid}: size={end-start}, epochs={base_ep}, bits={bits_block:.2f}, cum={preq_bits_total:.2f}")

        # Store block data for MI_theta computation
        data_partitions[bid] = {
            "start": start,
            "end": end,
            "size": end - start,
            "indices": block_ids
        }

        # grow prefix
        prefix_idx.extend(block_ids)
        total_steps_last_block = total_steps
        eta_min_summary = eta_min

    # Final evaluations
    full_train_eval = Subset(ds_train_eval, [perm_list[i] for i in range(n)])
    train_bits, train_acc = evaluate_full(model, device, full_train_eval, args.batch_size, args.workers)
    test_bits, test_acc = evaluate_full(model, device, ds_test, args.batch_size, args.workers)
    acc_gap = train_acc - test_acc

    # MDL summaries
    L_preq = preq_bits_total
    L_data = train_bits
    D_bits = L_preq - L_data                           # model code length
    D_bits_star = (L_preq - uniform_block0_bits) - L_data  # subtract block-0 uniform cost

    summary = {
        "total_train_samples": n,
        "n_full": n_full,
        "subsample_frac": args.subsample_frac,
        "subsample_seed": args.subsample_seed,
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
        "test_acc": test_acc,
        "train_acc": train_acc,
        "acc_gap": acc_gap,
        "base_seed": args.base_seed,
        "repeat_iter": args.repeat_iter,
        "derived_seed": seed,
        # bookkeeping
        "optimizer": args.optimizer,
        "lr": args.lr,
        "wd": args.wd,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "cold_start": bool(args.cold_start),
        "cold_swa_start_frac": args.cold_swa_start_frac if args.cold_start else None,
        "scheduler": f"warmup({args.warmup_frac:.2f}, start×{args.warmup_start_factor:.2f}) + cosine(T={total_steps_last_block}, eta_min={eta_min_summary:.2e})",
        "epochs_rule": "fixed" if args.fixed_epochs > 0 else "size-aware",
        "epochs_per_block_effective": "variable" if args.fixed_epochs == 0 else args.fixed_epochs,
        "steps_config": {
            "start_frac": args.start_frac,
            "start_min": args.start_min,
            "max_blocks": args.max_blocks,
            "factor": args.factor,
            "steps_preview": steps[:8] + (["..."] if len(steps) > 11 else []) + steps[-3:]
        },
    }

    print("==== Summary ====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    # Save trained model state_dict
    model_path = os.path.join(run_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[SAVED] Model checkpoint: {model_path}")

    # Save SWAG model state for MI_theta computation (leave-one-out approach)
    if swag_model is not None:
        try:
            n_collected = swag_model.n_models.item() if hasattr(swag_model, 'n_models') else 0
            if n_collected == 0:
                print(f"[WARNING] No SWAG models collected (global_epoch {global_epoch} < swa_start {args.swa_start})")
            else:
                swag_checkpoint = {
                    "epoch": len(steps),
                    "state_dict": swag_model.state_dict(),  # Contains mean, sq_mean, cov_mat_sqrt
                    "n_models": n_collected,
                    "n_layers": 5,  # Approximate number of layers for MI_theta computation
                    "seed": seed,
                    "base_seed": args.base_seed,
                    "repeat_iter": args.repeat_iter,
                    "global_epoch": global_epoch,
                    "swa_start": args.swa_start,
                }
                swag_path = os.path.join(run_dir, "swag_model.pt")
                torch.save(swag_checkpoint, swag_path)
                print(f"[SAVED] SWAG checkpoint: {swag_path}")
                print(f"[INFO]  Collected {n_collected} model checkpoints for Gaussian posterior approximation")
                print(f"[INFO]  Use these posteriors for leave-one-out MI_theta computation via Jensen bound")
        except Exception as e:
            print(f"[WARNING] Failed to save SWAG model: {e}")
            import traceback
            traceback.print_exc()
    
    # Save data partition info for leave-one-out MI_theta computation
    # This is needed to identify which data each fold used
    partitions_path = os.path.join(run_dir, "data_partitions.json")
    try:
        # Convert indices to lists for JSON serialization
        partitions_json = {}
        for bid, partition in data_partitions.items():
            partitions_json[str(bid)] = {
                "start": partition["start"],
                "end": partition["end"],
                "size": partition["size"],
                "num_indices": len(partition["indices"]),
                "indices": partition["indices"]  # Actual sample indices for MI_theta leave-one-out
            }
        with open(partitions_path, "w") as f:
            json.dump(partitions_json, f, indent=2)
        print(f"[SAVED] Data partitions metadata: {partitions_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save data partitions: {e}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        summary["swag_n_models"] = swag_model.n_models.item() if swag_model is not None and hasattr(swag_model, 'n_models') else 0
        summary["data_partitions"] = len(data_partitions)
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
