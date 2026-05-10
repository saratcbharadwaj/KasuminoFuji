"""
distribution_shift_analysis.py
-------------------------------
Visualises how the PPO policy distribution shifts across training checkpoints.

For each checkpoint (50K → 300K steps), runs the agent over the 500-contract
balanced eval set and records:
  - Mean P(READ), P(VULN), P(SAFE) at exit
  - Per-flag conditional probabilities: E[P(SAFE) | flag=1] vs E[P(SAFE) | flag=0]
  - F1-safe, F1-macro, avg steps
  - Outcome bucket counts (TP/FP/TN/FN)

Produces:
  1. logs/dist_shift_action_probs.png    — P(READ/VULN/SAFE) across timesteps
  2. logs/dist_shift_flag_influence.png  — how flag→action mapping evolves
  3. logs/dist_shift_metrics.png         — F1-safe, F1-macro, avg steps over time
  4. logs/dist_shift_buckets.png         — TP/FP/TN/FN evolution over training
  5. logs/dist_shift_results.csv         — raw numbers for every checkpoint
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from stable_baselines3 import PPO
from sklearn.metrics import f1_score
import torch

from configs.config import (
    MODELS_DIR, LOGS_DIR, DATASET_CSV,
    N_FEATURES, READ_NEXT, PREDICT_VULN, PREDICT_SAFE,
)
from utils.helpers import load_dataset, get_features, balanced_train_split
from env.smart_env import SmartContractEnv

FEATURE_NAMES  = ["loop", "external", "state", "payable", "access", "math"]
CHECKPOINTS    = [50_000, 100_000, 150_000, 200_000, 250_000, 300_000]
CKPT_DIR       = os.path.join(MODELS_DIR, "checkpoints")

ACTION_COLORS  = {"P(READ)": "#FF9800", "P(VULN)": "#F44336", "P(SAFE)": "#2196F3"}
BUCKET_COLORS  = {"TP": "#4CAF50", "FN": "#F44336", "TN": "#2196F3", "FP": "#FF9800"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_eval_set(n_safe=500, n_vuln=500, seed=777):
    print("[shift] Loading eval set ...")
    df           = load_dataset(DATASET_CSV)
    all_m, all_l = get_features(df)
    _, _, ev_m, ev_l = balanced_train_split(all_m, all_l)
    bal_m, bal_l, _, _ = balanced_train_split(
        ev_m, ev_l, n_safe=n_safe, n_vuln=n_vuln, seed=seed
    )
    print(f"[shift] Eval: {sum(x==0 for x in bal_l)} safe + "
          f"{sum(x==1 for x in bal_l)} vuln")
    return bal_m, bal_l


# ── Single checkpoint rollout ──────────────────────────────────────────────────

def rollout_checkpoint(model, matrices, labels):
    """
    Run one checkpoint over all eval contracts.
    Returns per-contract records with flags, probs, outcomes.
    """
    env     = SmartContractEnv(matrices, labels)
    records = []

    for idx in range(len(labels)):
        # Replicate reset() exactly
        env._flag_matrix = matrices[idx]
        env._label       = labels[idx]
        env._n_chunks    = len(matrices[idx])
        env._step        = 0
        env._cum_flags   = np.zeros(N_FEATURES, dtype=np.float32)
        env._cum_flags   = np.maximum(env._cum_flags, matrices[idx][0])
        env._step        = 1
        obs              = env._get_obs()

        done       = False
        predicted  = PREDICT_VULN
        exit_obs   = obs.copy()
        exit_probs = np.array([1/3, 1/3, 1/3])

        while not done:
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                dist  = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy().flatten()

            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

            if action in (PREDICT_VULN, PREDICT_SAFE):
                exit_obs   = obs.copy()
                exit_probs = probs.copy()
                predicted  = action

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        label = labels[idx]
        pred  = 0 if predicted == PREDICT_SAFE else 1

        if   label == 1 and pred == 1: bucket = "TP"
        elif label == 1 and pred == 0: bucket = "FN"
        elif label == 0 and pred == 0: bucket = "TN"
        else:                           bucket = "FP"

        row = {
            "label":      label,
            "prediction": pred,
            "bucket":     bucket,
            "steps":      info["steps_taken"],
            "p_read":     exit_probs[READ_NEXT],
            "p_vuln":     exit_probs[PREDICT_VULN],
            "p_safe":     exit_probs[PREDICT_SAFE],
        }
        for i, fname in enumerate(FEATURE_NAMES):
            row[f"flag_{fname}"] = int(exit_obs[i])
        records.append(row)

    return pd.DataFrame(records)


# ── Aggregate stats from one rollout df ───────────────────────────────────────

def compute_stats(df, timestep):
    y_true = df["label"].values
    y_pred = df["prediction"].values
    stats  = {
        "timestep":  timestep,
        "f1_safe":   f1_score(y_true, y_pred, pos_label=0,     zero_division=0),
        "f1_vuln":   f1_score(y_true, y_pred, pos_label=1,     zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "avg_steps": df["steps"].mean(),
        "mean_p_read": df["p_read"].mean(),
        "mean_p_vuln": df["p_vuln"].mean(),
        "mean_p_safe": df["p_safe"].mean(),
        "n_TP": len(df[df.bucket == "TP"]),
        "n_FN": len(df[df.bucket == "FN"]),
        "n_TN": len(df[df.bucket == "TN"]),
        "n_FP": len(df[df.bucket == "FP"]),
    }
    # Per-flag: E[P(SAFE) | flag=1] - E[P(SAFE) | flag=0]
    for fname in FEATURE_NAMES:
        active   = df[df[f"flag_{fname}"] == 1]["p_safe"].mean()
        inactive = df[df[f"flag_{fname}"] == 0]["p_safe"].mean()
        stats[f"flag_{fname}_safe_lift"] = active - inactive
        stats[f"flag_{fname}_active_psafe"]   = active
        stats[f"flag_{fname}_inactive_psafe"] = inactive
    return stats


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_action_probs(summary_df, save_dir):
    """How mean P(READ), P(VULN), P(SAFE) evolve over training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    steps = summary_df["timestep"] / 1000  # in K

    # Left: stacked area
    ax = axes[0]
    ax.stackplot(steps,
        summary_df["mean_p_read"],
        summary_df["mean_p_vuln"],
        summary_df["mean_p_safe"],
        labels=["P(READ)", "P(VULN)", "P(SAFE)"],
        colors=["#FF9800", "#F44336", "#2196F3"],
        alpha=0.8)
    ax.set_xlabel("Training Timesteps (K)")
    ax.set_ylabel("Mean Exit Probability")
    ax.set_title("Action Probability Distribution at Exit\n(stacked area — how mass shifts over training)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(steps.min(), steps.max())
    ax.axhline(1/3, color="black", linestyle=":", alpha=0.4, label="Random baseline")

    # Right: line plot
    ax2 = axes[1]
    ax2.plot(steps, summary_df["mean_p_read"], "o-",
             color="#FF9800", linewidth=2, label="P(READ)", markersize=6)
    ax2.plot(steps, summary_df["mean_p_vuln"], "s-",
             color="#F44336", linewidth=2, label="P(VULN)", markersize=6)
    ax2.plot(steps, summary_df["mean_p_safe"], "^-",
             color="#2196F3", linewidth=2, label="P(SAFE)", markersize=6)
    ax2.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Random (0.333)")
    ax2.set_xlabel("Training Timesteps (K)")
    ax2.set_ylabel("Mean Exit Probability")
    ax2.set_title("Action Probability Trajectories\n(individual lines — convergence pattern)")
    ax2.legend(fontsize=9)
    ax2.set_xlim(steps.min(), steps.max())
    ax2.set_ylim(0, 1.0)

    plt.suptitle("Policy Distribution Shift: How the Agent Learns to Decide",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "dist_shift_action_probs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shift] Saved → {path}")


def plot_flag_influence_over_time(summary_df, save_dir):
    """
    For each flag: how does E[P(SAFE)|flag=1] - E[P(SAFE)|flag=0] evolve?
    Positive lift = flag pushes toward SAFE prediction.
    Negative lift = flag pushes toward VULN prediction.
    """
    steps = summary_df["timestep"] / 1000

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, fname in enumerate(FEATURE_NAMES):
        ax   = axes[i]
        lift = summary_df[f"flag_{fname}_safe_lift"]
        active   = summary_df[f"flag_{fname}_active_psafe"]
        inactive = summary_df[f"flag_{fname}_inactive_psafe"]

        ax.fill_between(steps, 0, lift,
            where=(lift >= 0), alpha=0.3, color="#2196F3",
            label="Pushes SAFE")
        ax.fill_between(steps, 0, lift,
            where=(lift < 0),  alpha=0.3, color="#F44336",
            label="Pushes VULN")
        ax.plot(steps, lift, "o-", color="#1A2E4A",
                linewidth=2, markersize=5, label="P(SAFE|flag=1) - P(SAFE|flag=0)")
        ax.plot(steps, active,   "--", color="#2196F3",
                linewidth=1.2, alpha=0.7, label="P(SAFE | flag=1)")
        ax.plot(steps, inactive, "--", color="#F44336",
                linewidth=1.2, alpha=0.7, label="P(SAFE | flag=0)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"has_{fname}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps (K)")
        ax.set_ylabel("Lift / Probability")
        ax.set_ylim(-1.0, 1.0)
        ax.legend(fontsize=6.5)

    plt.suptitle(
        "Flag Influence on SAFE Prediction: How Each Flag's Effect Evolves During Training\n"
        "(positive lift = flag associated with SAFE; negative = flag associated with VULN)",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "dist_shift_flag_influence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shift] Saved → {path}")


def plot_metrics_over_time(summary_df, save_dir):
    """F1-safe, F1-macro, avg steps across checkpoints."""
    steps = summary_df["timestep"] / 1000

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # F1-safe
    axes[0].plot(steps, summary_df["f1_safe"], "o-",
                 color="#2196F3", linewidth=2.5, markersize=7)
    axes[0].axhline(0.30, color="red", linestyle="--",
                    alpha=0.7, label="Target >= 0.30")
    axes[0].set_title("F1-Safe (Minority Class)\nover Training")
    axes[0].set_xlabel("Timesteps (K)")
    axes[0].set_ylabel("F1-safe")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=9)
    for x, y in zip(steps, summary_df["f1_safe"]):
        axes[0].annotate(f"{y:.3f}", (x, y),
                         textcoords="offset points", xytext=(0, 8), fontsize=8,
                         ha="center")

    # F1-macro
    axes[1].plot(steps, summary_df["f1_macro"], "s-",
                 color="#4CAF50", linewidth=2.5, markersize=7)
    axes[1].set_title("F1-Macro (Both Classes)\nover Training")
    axes[1].set_xlabel("Timesteps (K)")
    axes[1].set_ylabel("F1-macro")
    axes[1].set_ylim(0, 1.0)
    for x, y in zip(steps, summary_df["f1_macro"]):
        axes[1].annotate(f"{y:.3f}", (x, y),
                         textcoords="offset points", xytext=(0, 8), fontsize=8,
                         ha="center")

    # Avg steps
    axes[2].plot(steps, summary_df["avg_steps"], "^-",
                 color="#FF9800", linewidth=2.5, markersize=7)
    axes[2].axhline(6.0, color="red", linestyle="--",
                    alpha=0.7, label="Target <= 6.0")
    axes[2].set_title("Avg Chunks Read\nover Training")
    axes[2].set_xlabel("Timesteps (K)")
    axes[2].set_ylabel("Avg Steps")
    axes[2].legend(fontsize=9)
    for x, y in zip(steps, summary_df["avg_steps"]):
        axes[2].annotate(f"{y:.2f}", (x, y),
                         textcoords="offset points", xytext=(0, 8), fontsize=8,
                         ha="center")

    plt.suptitle("Performance Metrics Across Training Checkpoints", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "dist_shift_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shift] Saved → {path}")


def plot_buckets_over_time(summary_df, save_dir):
    """How TP/FP/TN/FN counts evolve over training."""
    steps  = summary_df["timestep"] / 1000
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes   = axes.flatten()

    for i, (bucket, label, color) in enumerate([
        ("TP", "True Vuln Caught",    "#4CAF50"),
        ("FN", "False Safe (Missed)", "#F44336"),
        ("TN", "True Safe (Correct)", "#2196F3"),
        ("FP", "False Alarm",         "#FF9800"),
    ]):
        ax = axes[i]
        col = f"n_{bucket}"
        ax.bar(steps, summary_df[col], width=28,
               color=color, alpha=0.8, edgecolor="white")
        ax.plot(steps, summary_df[col], "o-",
                color="black", linewidth=1.5, markersize=5)
        ax.set_title(f"{bucket}: {label}", fontsize=11, fontweight="bold",
                     color=color)
        ax.set_xlabel("Timesteps (K)")
        ax.set_ylabel("Count (out of 1,000)")
        ax.set_ylim(0, 600)
        for x, y in zip(steps, summary_df[col]):
            ax.annotate(str(int(y)), (x, y),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=9, ha="center", fontweight="bold")

    plt.suptitle(
        "Outcome Bucket Evolution Across Training\n"
        "(shows how the agent's decision behaviour changes checkpoint by checkpoint)",
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "dist_shift_buckets.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shift] Saved → {path}")


def plot_heatmap_evolution(all_dfs, save_dir):
    """
    Heatmap: for each checkpoint x each flag, show E[P(SAFE)|flag=1].
    Rows = flags, Cols = timesteps.
    Reveals which flag-action associations emerge and when.
    """
    data = {}
    for ts, df in all_dfs.items():
        col = {}
        for fname in FEATURE_NAMES:
            active = df[df[f"flag_{fname}"] == 1]["p_safe"]
            col[fname] = active.mean() if len(active) > 0 else 0.5
        data[f"{ts//1000}K"] = col

    hmap_df = pd.DataFrame(data).T   # rows=timesteps, cols=flags

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        hmap_df.T, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0.5, vmin=0.0, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "E[P(SAFE) | flag=1]"}
    )
    ax.set_title(
        "Flag → SAFE Probability Heatmap Across Training Checkpoints\n"
        "(green = flag strongly predicts SAFE; red = flag predicts VULN)",
        fontsize=11, pad=14
    )
    ax.set_xlabel("Training Checkpoint")
    ax.set_ylabel("Static Flag")
    plt.tight_layout()
    path = os.path.join(save_dir, "dist_shift_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shift] Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    matrices, labels = load_eval_set()

    all_stats = []
    all_dfs   = {}

    for ts in CHECKPOINTS:
        ckpt = os.path.join(CKPT_DIR, f"ppo_sc_{ts}_steps.zip")
        if not os.path.exists(ckpt):
            print(f"[shift] Checkpoint not found: {ckpt} — skipping")
            continue

        print(f"\n[shift] Evaluating checkpoint @ {ts:,} steps ...")
        model = PPO.load(ckpt)
        df    = rollout_checkpoint(model, matrices, labels)
        stats = compute_stats(df, ts)
        all_stats.append(stats)
        all_dfs[ts] = df

        print(f"         F1-safe={stats['f1_safe']:.3f}  "
              f"F1-macro={stats['f1_macro']:.3f}  "
              f"avg_steps={stats['avg_steps']:.2f}  "
              f"TP={stats['n_TP']}  FN={stats['n_FN']}  "
              f"TN={stats['n_TN']}  FP={stats['n_FP']}")

    if not all_stats:
        print("[shift] No checkpoints found. Exiting.")
        return

    summary_df = pd.DataFrame(all_stats)

    # Save CSV
    csv_path = os.path.join(LOGS_DIR, "dist_shift_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[shift] Results saved → {csv_path}")

    # Generate all plots
    print("\n[shift] Generating plots ...")
    plot_action_probs(summary_df, LOGS_DIR)
    plot_flag_influence_over_time(summary_df, LOGS_DIR)
    plot_metrics_over_time(summary_df, LOGS_DIR)
    plot_buckets_over_time(summary_df, LOGS_DIR)
    plot_heatmap_evolution(all_dfs, LOGS_DIR)

    # Print summary table
    print("\n" + "="*72)
    print(f"{'Steps':>8}  {'F1-safe':>8}  {'F1-mac':>7}  "
          f"{'Steps':>7}  {'P(READ)':>8}  {'P(VULN)':>8}  {'P(SAFE)':>8}  "
          f"{'TP':>4}  {'FN':>4}  {'TN':>4}  {'FP':>4}")
    print("-"*72)
    for _, r in summary_df.iterrows():
        print(f"{int(r.timestep):>8,}  {r.f1_safe:>8.3f}  {r.f1_macro:>7.3f}  "
              f"{r.avg_steps:>7.2f}  {r.mean_p_read:>8.3f}  "
              f"{r.mean_p_vuln:>8.3f}  {r.mean_p_safe:>8.3f}  "
              f"{int(r.n_TP):>4}  {int(r.n_FN):>4}  "
              f"{int(r.n_TN):>4}  {int(r.n_FP):>4}")
    print("="*72)
    print(f"\n[shift] All plots → {LOGS_DIR}")


if __name__ == "__main__":
    main()
