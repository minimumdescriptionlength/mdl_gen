#!/usr/bin/env python3
import os, re, json, csv, sys
from pathlib import Path

RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else "./runs"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "./runs_sweeps"
OUT_CSV  = Path(OUT_DIR) / "sweep_wd_merged.csv"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Match names like: r18_wd1.0em03_rep2   (wd = 1.0e-03, rep=2)
pat = re.compile(r".*?_wd([0-9.]+)e([m0-9]+)_rep(\d+)$")

def wd_token_to_float(token_num:str, token_exp:str)->float:
    # token like "1.0" and "m03" or "02"
    exp = token_exp.replace("m","-")  # 'm03' -> '-03'
    return float(f"{token_num}e{exp}")

rows = []
for d in sorted(Path(RUNS_DIR).iterdir()):
    if not d.is_dir(): 
        continue
    m = pat.match(d.name)
    if not m:
        continue
    num, exp, rep = m.groups()
    wd = wd_token_to_float(num, exp)
    rep = int(rep)
    summ_path = d / "summary.json"
    if not summ_path.exists():
        print(f"[WARN] Missing {summ_path}, skipping.")
        continue
    with open(summ_path) as f:
        S = json.load(f)
    rows.append({
        "run_name": d.name,
        "wd": wd,
        "repeat_iter": rep,
        "L_preq": S["prequential_bits"],
        "L_data": S["final_train_bits"],
        "L_model": S["prequential_bits"] - S["final_train_bits"],
        "L_preq_per_img": S["prequential_bits_per_img"],
        "L_data_per_img": S["final_train_bits_per_img"],
        "L_model_per_img": S["D_bits_per_img"],
        "train_acc": S["train_acc"],
        "test_acc": S["test_acc"],
        "acc_gap": S["acc_gap"],
    })

# write CSV
fields = ["run_name","wd","repeat_iter","L_preq","L_data","L_model",
          "L_preq_per_img","L_data_per_img","L_model_per_img",
          "train_acc","test_acc","acc_gap"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
