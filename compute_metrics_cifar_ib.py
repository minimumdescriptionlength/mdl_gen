#!/usr/bin/env python3
"""
Compute correlation metrics for the CIFAR-10 IB grid experiment (Phase 1 / 2).

Goal: compare our prequential MDL predictor L_model_per_img against the
generalization gap targets used in the paper's Table 2:

    Paper (Kawaguchi, Deng, Ji, Huang, ICML 2023) Table 2 format
    (from dnn/swag_repo/MI/main_multiseed.py, line 609):

        metric & summary & rho_GL & rho_GE & r_GL & r_GE & tau_GL & tau_GE

    where:
        GL = test_loss - train_loss           (generalization gap in loss)
        GE = test_err  - train_err
           = train_acc - test_acc = acc_gap   (generalization gap in error)
        rho = Spearman, r = Pearson, tau = Kendall

Filter: train_acc > 0.85  (from toy/scripts/subsample/main_swag_subsample.py,
         which governs the 30%-subsample / toy-CIFAR setting we replicate).

Our predictor is stored in summary.json as `D_bits_per_img`
(= (L_preq - L_data) / n = prequential model-code length per sample).

Units note: loss_gap is stored in bits; the paper's is in nats.  The
1/ln(2) factor is a constant rescaling of one variable, which does not
affect Pearson r, Spearman rho, or Kendall tau.

Inputs : runs_cifar_ib/<run_name>/summary.json (one file per training run)
Outputs: CSV and JSON with one row per (predictor, target) pair, each row
         containing {n, pearson_{r,p}, spearman_{rho,p}, kendall_{tau,p}}.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from correlation_metrics import compute_all_correlations


# Keys to look for in summary.json for our MDL predictor, in priority order.
# mdl_preq_cifar_ib.py writes D_bits_per_img; the table-1 runner writes
# D_bits_per_sample; accept either for robustness.
PREDICTOR_KEYS = ("D_bits_per_img", "L_model_per_img", "D_bits_per_sample")

# Paper filter threshold (subsample/main_swag_subsample.py, line 413)
TRAIN_ACC_THRESH = 0.85

# (tag, target_key, description matching paper column label)
PAIRS: List[Tuple[str, str, str]] = [
    (
        "L_model_per_img_vs_loss_gap",
        "loss_gap_bits_per_sample",
        "GL = test_loss - train_loss (bits)",
    ),
    (
        "L_model_per_img_vs_acc_gap",
        "acc_gap",
        "GE = train_acc - test_acc  (= test_err - train_err)",
    ),
]


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pick_predictor(s: Dict) -> Optional[float]:
    for k in PREDICTOR_KEYS:
        v = _as_float(s.get(k))
        if v is not None:
            return v
    return None


def _passes_filter(s: Dict) -> bool:
    train_acc = _as_float(s.get("train_acc"))
    return train_acc is not None and train_acc > TRAIN_ACC_THRESH


def load_summaries(runs_dir: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json")))
    out: List[Dict] = []
    for p in paths:
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"[metrics] skipping {p}: {e}")
    return out


def build_pairs(
    summaries: List[Dict],
    target_key: str,
    filter_accepted: bool,
) -> Tuple[List[float], List[float], int]:
    xs: List[float] = []
    ys: List[float] = []
    n_rejected = 0
    for s in summaries:
        if filter_accepted and not _passes_filter(s):
            n_rejected += 1
            continue
        x = _pick_predictor(s)
        y = _as_float(s.get(target_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys, n_rejected


def compute_all_pairs(runs_dir: str, filter_accepted: bool) -> List[Dict]:
    summaries = load_summaries(runs_dir)
    print(f"[metrics] scanned {len(summaries)} runs in {runs_dir}")

    rows: List[Dict] = []
    for tag, target_key, desc in PAIRS:
        xs, ys, n_rej = build_pairs(summaries, target_key, filter_accepted)
        if filter_accepted:
            print(f"[metrics] {tag}: {n_rej} rejected (train_acc <= {TRAIN_ACC_THRESH})")
        print(f"[metrics] {tag}: usable pairs = {len(xs)}  [{desc}]")
        if len(xs) < 3:
            print(f"[metrics]   too few pairs — skipping {tag}")
            continue
        print(f"[metrics] === {tag} ===")
        r = compute_all_correlations(xs, ys, verbose=True)
        rows.append({
            "pair":         tag,
            "target":       target_key,
            "target_desc":  desc,
            "n":            len(xs),
            "pearson_r":    r["pearson"][0],
            "pearson_p":    r["pearson"][1],
            "spearman_rho": r["spearman"][0],
            "spearman_p":   r["spearman"][1],
            "kendall_tau":  r["kendall"][0],
            "kendall_p":    r["kendall"][1],
        })
    return rows


def _latex_row(rows: List[Dict]) -> None:
    """Print a LaTeX table row matching the paper's Table 2 column order:
    rho_GL  rho_GE  r_GL  r_GE  tau_GL  tau_GE"""
    by_tag = {r["pair"]: r for r in rows}
    gl = by_tag.get("L_model_per_img_vs_loss_gap")
    ge = by_tag.get("L_model_per_img_vs_acc_gap")
    if gl is None or ge is None:
        return
    print(
        "\n[metrics] Table 2 LaTeX row (our MDL predictor):\n"
        "  L_model_per_img & -- "
        f"& {gl['spearman_rho']:.4f} & {ge['spearman_rho']:.4f} "
        f"& {gl['pearson_r']:.4f} & {ge['pearson_r']:.4f} "
        f"& {gl['kendall_tau']:.4f} & {ge['kendall_tau']:.4f} \\\\"
    )


def write_outputs(rows: List[Dict], out_csv: str, out_json: Optional[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    fieldnames = [
        "pair", "target", "target_desc", "n",
        "pearson_r",    "pearson_p",
        "spearman_rho", "spearman_p",
        "kendall_tau",  "kendall_p",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[metrics] wrote {out_csv}")

    if out_json:
        payload = {
            "predictor": (
                "D_bits_per_img (a.k.a. L_model_per_img): "
                "prequential MDL model-code length per training sample"
            ),
            "filter": f"train_acc > {TRAIN_ACC_THRESH}",
            "note_bits_vs_nats": (
                "loss_gap is stored in bits; the paper's GL is in nats. "
                "They differ by 1/ln(2), a constant factor that does not "
                "affect Pearson r, Spearman rho, or Kendall tau."
            ),
            "table2_column_order": "rho_GL, rho_GE, r_GL, r_GE, tau_GL, tau_GE",
            "rows": rows,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[metrics] wrote {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_dir", required=True,
        help="Directory of <run_name>/summary.json files (e.g. ./runs_cifar_ib)",
    )
    ap.add_argument("--out_csv",  required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument(
        "--no_filter", action="store_true",
        help=f"Include models that fail the paper's train_acc>{TRAIN_ACC_THRESH} filter",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.runs_dir):
        raise SystemExit(f"[metrics] missing runs_dir: {args.runs_dir}")

    rows = compute_all_pairs(args.runs_dir, filter_accepted=not args.no_filter)
    if not rows:
        raise SystemExit("[metrics] no usable correlations produced")

    _latex_row(rows)
    write_outputs(rows, args.out_csv, args.out_json)


if __name__ == "__main__":
    main()
