# run_pipeline.py

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from configs.config import MODELS_DIR, LOGS_DIR, DATASET_CSV
from utils.helpers import (
    load_dataset, get_features,
    balanced_train_split, make_synthetic_dataset,
)
from training.train_ppo import train
from evaluation.metrics import evaluate_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic",  action="store_true")
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--model-path", default=None)
    return p.parse_args()


def main():
    args      = parse_args()
    eval_only = args.eval_only or (args.model_path is not None)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    use_synthetic = args.synthetic or not os.path.exists(DATASET_CSV)
    if use_synthetic:
        print("\n[pipeline] Using synthetic dataset …")
        all_m, all_l = make_synthetic_dataset(n_safe=2000, n_vuln=2000)
    else:
        print("\n[pipeline] Loading SmartBugs CSV …")
        df = load_dataset()
        all_m, all_l = get_features(df)

    # Train/eval split
    _, _, ev_m, ev_l = balanced_train_split(all_m, all_l)

    # ── KEY FIX: evaluate on a BALANCED held-out set ──────────────────────────
    # Raw eval set is 94% vuln — every metric looks great for lazy agents.
    # Use 500 safe + 500 vuln so all agents are judged fairly.
    bal_m, bal_l, _, _ = balanced_train_split(
        ev_m, ev_l, n_safe=500, n_vuln=500, seed=777
    )
    print(f"\n[pipeline] Eval set: {sum(x==0 for x in bal_l)} safe + "
          f"{sum(x==1 for x in bal_l)} vuln (balanced)")

    # ── 2. Train or load ───────────────────────────────────────────────────────
    if eval_only:
        model_path = args.model_path or os.path.join(MODELS_DIR, "ppo_final.zip")
        print(f"\n[pipeline] Loading model from {model_path} …")
        model = PPO.load(model_path)
    else:
        model, _, _ = train(use_synthetic=use_synthetic)

    # ── 3. Evaluate ────────────────────────────────────────────────────────────
    print(f"\n[pipeline] Evaluating …")
    results = evaluate_all(model, bal_m, bal_l, save_dir=LOGS_DIR)

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print(f"{'Agent':<18}  {'F1-safe':>8}  {'F1-macro':>9}  {'Avg steps':>10}")
    print("-"*62)
    for name, r in results.items():
        mf = "✓" if r["f1_safe"]   >= 0.30 else "✗"
        ms = "✓" if r["avg_steps"] <= 6.0  else "✗"
        print(f"{name:<18}  {r['f1_safe']:>7.3f}{mf}  "
              f"{r['f1_macro']:>8.3f}  {r['avg_steps']:>9.2f}{ms}")
    print("="*62)
    print(f"  ✓ = meets target   (F1-safe ≥ 0.30 | avg_steps ≤ 6.0)")
    print(f"  Dashboard → {os.path.join(LOGS_DIR, 'evaluation_dashboard.png')}")


if __name__ == "__main__":
    main()
