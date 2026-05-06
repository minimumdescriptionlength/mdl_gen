#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import math

def fmt_wd(wd: float) -> str:
    return f"{wd:.1e}".replace("+", "").replace("-", "m")

def run_once(script, data, out_root, base_seed, repeat_iter,
             batch_size, workers, lr, wd,
             optimizer, label_smoothing, fixed_epochs,
             min_lr_factor, warmup_frac, warmup_start_factor,
             start_frac, start_min, max_blocks, factor,
             dry_run=False, verbose=True):
    run_name = f"r18_wd{fmt_wd(wd)}_rep{repeat_iter}"
    run_dir = os.path.join(out_root, run_name)

    cmd = [
        sys.executable, script,
        "--data", data,
        "--out", out_root,
        "--run_name", run_name,
        "--base_seed", str(base_seed),
        "--repeat_iter", str(repeat_iter),
        "--batch_size", str(batch_size),
        "--workers", str(workers),
        "--lr", str(lr),
        "--wd", str(wd),
        "--optimizer", optimizer,
        "--label_smoothing", str(label_smoothing),
        "--min_lr_factor", str(min_lr_factor),
        "--warmup_frac", str(warmup_frac),
        "--warmup_start_factor", str(warmup_start_factor),
        "--start_frac", str(start_frac),
        "--start_min", str(start_min),
        "--max_blocks", str(max_blocks),
    ]
    if fixed_epochs and fixed_epochs > 0:
        cmd += ["--fixed_epochs", str(fixed_epochs)]
    if factor is not None:
        cmd += ["--factor", str(factor)]

    if verbose:
        print("=" * 80)
        print(f"[PRE-FLIGHT] script={script} (exists: {os.path.exists(script)})")
        print(f"[PRE-FLIGHT] data={data} (exists: {os.path.exists(data)})")
        print(f"[PRE-FLIGHT] out_root={out_root}  run_dir={run_dir}")
        print(f"[PRE-FLIGHT] cmd: {' '.join(cmd)}")

    if dry_run:
        # Don't run and don't try to read summary.json
        print("[DRY-RUN] Skipping execution and summary read.")
        return {
            "run_name": run_name,
            "wd": wd,
            "repeat_iter": repeat_iter,
            "L_preq": math.nan,
            "L_data": math.nan,
            "L_model": math.nan,
            "L_preq_per_img": math.nan,
            "L_data_per_img": math.nan,
            "L_model_per_img": math.nan,
            "L_model_star": math.nan,
            "L_model_star_per_img": math.nan,
            "train_acc": math.nan,
            "test_acc": math.nan,
            "acc_gap": math.nan,
            "test_bits_per_img": math.nan,
        }

    Path(out_root).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess exited with code {e.returncode}")
        raise
    dt = time.time() - t0
    print(f"[DONE] {run_name} finished in {dt/60:.2f} min")

    # Read summary.json
    summ_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summ_path):
        raise FileNotFoundError(f"summary.json not found at {summ_path}. Did the run complete?")

    with open(summ_path, "r") as f:
        S = json.load(f)

    L_preq = S["prequential_bits"]
    L_data = S["final_train_bits"]
    L_model = L_preq - L_data  # D_bits (also in S["D_bits"] in your script)

    if verbose:
        print(f"[SUMMARY] {run_name}  wd={wd:.2e}  "
              f"L_model={L_model:.2f}  L_preq={L_preq:.2f}  L_data={L_data:.2f}  "
              f"acc(train/test)={S['train_acc']:.4f}/{S['test_acc']:.4f}")

    return {
        "run_name": run_name,
        "wd": wd,
        "repeat_iter": repeat_iter,
        "L_preq": L_preq,
        "L_data": L_data,
        "L_model": L_model,
        "L_preq_per_img": S.get("prequential_bits_per_img"),
        "L_data_per_img": S.get("final_train_bits_per_img"),
        "L_model_per_img": S.get("D_bits_per_img"),
        "L_model_star": S.get("D_bits_star"),
        "L_model_star_per_img": S.get("D_bits_star_per_img"),
        "train_acc": S["train_acc"],
        "test_acc": S["test_acc"],
        "acc_gap": S["acc_gap"],
        "test_bits_per_img": S.get("test_bits_per_img"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="mdl_preq_blockwise_test.py")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--out", type=str, default="./runs")
    parser.add_argument("--sweep_out", type=str, default="./runs_sweeps")
    parser.add_argument("--wds", type=float, nargs="+", default=[0.0, 1e-4, 5e-4, 1e-3])
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--repeat_iter", type=int, default=0)

    # training/runtime
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--fixed_epochs", type=int, default=0)

    # scheduler & steps
    parser.add_argument("--min_lr_factor", type=float, default=1e-2)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)

    # timetable
    parser.add_argument("--start_frac", type=float, default=0.005)  # keep YOUR defaults
    parser.add_argument("--start_min", type=int, default=100)
    parser.add_argument("--max_blocks", type=int, default=40)
    parser.add_argument("--factor", type=float, default=None)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.sweep_out).mkdir(parents=True, exist_ok=True)

    rows = []
    for i, wd in enumerate(args.wds, 1):
        print(f"\n[SWEEP] ({i}/{len(args.wds)}) wd={wd:.2e}")
        res = run_once(
            script=args.script,
            data=args.data,
            out_root=args.out,
            base_seed=args.base_seed,
            repeat_iter=args.repeat_iter,
            batch_size=args.batch_size,
            workers=args.workers,
            lr=args.lr,
            wd=wd,
            optimizer=args.optimizer,
            label_smoothing=args.label_smoothing,
            fixed_epochs=args.fixed_epochs,
            min_lr_factor=args.min_lr_factor,
            warmup_frac=args.warmup_frac,
            warmup_start_factor=args.warmup_start_factor,
            start_frac=args.start_frac,
            start_min=args.start_min,
            max_blocks=args.max_blocks,
            factor=args.factor,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
        rows.append(res)

    csv_path = os.path.join(args.sweep_out, f"sweep_wd_rep{args.repeat_iter}.csv")
    fieldnames = [
        "run_name","wd","repeat_iter",
        "L_preq","L_data","L_model",
        "L_preq_per_img","L_data_per_img","L_model_per_img",
        "L_model_star","L_model_star_per_img",
        "train_acc","test_acc","acc_gap","test_bits_per_img"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[SAVED] Sweep CSV → {csv_path}")
    if args.dry_run:
        print("[NOTE] Dry run used NaNs in CSV (no executions).")

if __name__ == "__main__":
    main()
