# Ablation study
## Policy distribution ablation study for the smart contracts

## Goal: To run trained PPO agent over the 1,000-contract balanced held-out evaluation set and records:
## 1. Exit-state flags
## 2. Policy logitc/action probabilities at exit
## 3. Ground truth label vs agent prediction
## 4. Number of chunks read before exit.

##

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from stable_baselines3 import PPO

from configs.config import (
	MODELS_DIR, LOGS_DIR, DATASET_CSV,
	N_FEATURES, READ_NEXT, PREDICT_VULN, PREDICT_SAFE,
)

from utils.helpers import (
	load_dataset, get_features,
	balanced_train_split,
)

from env.smart_env import SmartContractEnv

FEATURE_NAMES = ["loop", "external", "state", "payable", "access", "math"]

BUCKET_LABELS = {
	"TP" : "True Vuln (Caught)",
	"FN" : "False Safe (Bug missed)",
	"TN" : "True Safe (Correct)",
	"FP" : "False Vuln (False Alarm)",
}

BUCKET_COLORS  = {
    "TP": "#4CAF50",
    "FN": "#F44336",
    "TN": "#2196F3",
    "FP": "#FF9800",
}


def load_eval_set(n_safe=500, n_vuln=500, seed=777):
    print("[ablation] Loading dataset and features ...")
    df            = load_dataset(DATASET_CSV)
    all_m, all_l  = get_features(df)
    _, _, ev_m, ev_l = balanced_train_split(all_m, all_l)
    bal_m, bal_l, _, _ = balanced_train_split(
        ev_m, ev_l, n_safe=n_safe, n_vuln=n_vuln, seed=seed
    )
    print(f"[ablation] Eval set: {sum(x==0 for x in bal_l)} safe + "
          f"{sum(x==1 for x in bal_l)} vuln")
    return bal_m, bal_l

def rollout_with_logits(model, matrices, labels):
    """
    Run agent deterministically over every contract.
    Returns a DataFrame with one row per contract.
    """
    env    = SmartContractEnv(matrices, labels)
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
 
        done      = False
        predicted = PREDICT_VULN
        exit_obs  = obs.copy()
        exit_probs = np.array([1/3, 1/3, 1/3])
 
        while not done:
            # Get raw action probabilities from policy
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            with __import__('torch').no_grad():
                dist   = model.policy.get_distribution(obs_tensor)
                probs  = dist.distribution.probs.cpu().numpy().flatten()
 
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)
 
            # Record state and probs at the moment of exit decision
            if action in (PREDICT_VULN, PREDICT_SAFE):
                exit_obs   = obs.copy()
                exit_probs = probs.copy()
                predicted  = action
 
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
 
        label = labels[idx]
        pred  = 0 if predicted == PREDICT_SAFE else 1
 
        # Determine bucket
        if label == 1 and pred == 1:
            bucket = "TP"
        elif label == 1 and pred == 0:
            bucket = "FN"
        elif label == 0 and pred == 0:
            bucket = "TN"
        else:
            bucket = "FP"
 
        row = {
            "contract_idx": idx,
            "label":        label,
            "prediction":   pred,
            "bucket":       bucket,
            "steps_taken":  info["steps_taken"],
            "prob_read":    exit_probs[READ_NEXT],
            "prob_vuln":    exit_probs[PREDICT_VULN],
            "prob_safe":    exit_probs[PREDICT_SAFE],
        }
        for i, fname in enumerate(FEATURE_NAMES):
            row[f"flag_{fname}"] = int(exit_obs[i])
 
        records.append(row)
 
    return pd.DataFrame(records)


def plot_heatmap(df, save_dir):
    """Correlation heatmap: flag values vs outcome bucket (one-hot)."""
    flag_cols   = [f"flag_{f}" for f in FEATURE_NAMES]
    bucket_dummies = pd.get_dummies(df["bucket"])
 
    corr_data = pd.concat([df[flag_cols], bucket_dummies], axis=1)
    corr      = corr_data.corr().loc[flag_cols, bucket_dummies.columns]
    corr.index = FEATURE_NAMES
 
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-0.3, vmax=0.3,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"}
    )
    ax.set_title(
        "Flag–Outcome Correlation\n"
        "(positive = flag associated with outcome; negative = flag associated against outcome)",
        fontsize=11, pad=14
    )
    ax.set_xlabel("Outcome Bucket", fontsize=10)
    ax.set_ylabel("Static Flag at Exit", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Heatmap saved → {path}")

def plot_heatmap(df, save_dir):
    """Correlation heatmap: flag values vs outcome bucket (one-hot)."""
    flag_cols   = [f"flag_{f}" for f in FEATURE_NAMES]
    bucket_dummies = pd.get_dummies(df["bucket"])
 
    corr_data = pd.concat([df[flag_cols], bucket_dummies], axis=1)
    corr      = corr_data.corr().loc[flag_cols, bucket_dummies.columns]
    corr.index = FEATURE_NAMES
 
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-0.3, vmax=0.3,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"}
    )
    ax.set_title(
        "Flag–Outcome Correlation\n"
        "(positive = flag associated with outcome; negative = flag associated against outcome)",
        fontsize=11, pad=14
    )
    ax.set_xlabel("Outcome Bucket", fontsize=10)
    ax.set_ylabel("Static Flag at Exit", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Heatmap saved → {path}")

def plot_bucket_summary(df, save_dir):
    """Bucket counts + average steps per bucket."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # Left: bucket counts
    bucket_order  = ["TP", "FN", "TN", "FP"]
    counts        = [len(df[df.bucket == b]) for b in bucket_order]
    colors        = [BUCKET_COLORS[b] for b in bucket_order]
    labels_full   = [BUCKET_LABELS[b] for b in bucket_order]
 
    bars = axes[0].bar(labels_full, counts, color=colors, edgecolor="white")
    axes[0].set_title("Outcome Distribution (N=1,000)", fontsize=11)
    axes[0].set_ylabel("Count")
    for bar, v in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 5,
                     str(v), ha="center", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis='x', rotation=15)
 
    # Right: avg steps per bucket
    avg_steps = [df[df.bucket == b]["steps_taken"].mean() for b in bucket_order]
    bars2 = axes[1].bar(labels_full, avg_steps, color=colors, edgecolor="white")
    axes[1].axhline(6.0, color="red", linestyle="--", label="Target ≤ 6.0")
    axes[1].set_title("Avg Chunks Read per Outcome Bucket", fontsize=11)
    axes[1].set_ylabel("Avg Steps")
    axes[1].legend()
    for bar, v in zip(bars2, avg_steps):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.1,
                     f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].tick_params(axis='x', rotation=15)
 
    plt.suptitle("Policy Distribution Ablation — Bucket Analysis", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_bucket_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Bucket summary saved → {path}")

def plot_flag_influence(df, save_dir):
    """
    For each flag, show mean probability of SAFE vs VULN prediction
    when that flag is active (=1) vs inactive (=0).
    Reveals which flags push the agent toward each prediction.
    """
    flag_cols = [f"flag_{f}" for f in FEATURE_NAMES]
 
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
 
    for i, (fname, fcol) in enumerate(zip(FEATURE_NAMES, flag_cols)):
        ax = axes[i]
        active   = df[df[fcol] == 1]
        inactive = df[df[fcol] == 0]
 
        categories = ["P(READ)", "P(VULN)", "P(SAFE)"]
        active_means   = [active["prob_read"].mean(),
                          active["prob_vuln"].mean(),
                          active["prob_safe"].mean()]
        inactive_means = [inactive["prob_read"].mean(),
                          inactive["prob_vuln"].mean(),
                          inactive["prob_safe"].mean()]
 
        x    = np.arange(len(categories))
        w    = 0.35
        ax.bar(x - w/2, inactive_means, w, label=f"{fname}=0  (n={len(inactive)})",
               color="#90CAF9", edgecolor="white")
        ax.bar(x + w/2, active_means,   w, label=f"{fname}=1  (n={len(active)})",
               color="#1565C0", edgecolor="white")
        ax.set_title(f"has_{fname}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean Exit Probability")
        ax.legend(fontsize=8)
        ax.axhline(0.333, color="gray", linestyle=":", alpha=0.5, label="Random")
 
    plt.suptitle(
        "Flag Influence on Exit Action Probabilities\n"
        "(how each flag shifts P(READ), P(VULN), P(SAFE) at exit)",
        fontsize=13
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_flag_influence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Flag influence plot saved → {path}")

def plot_pareto(df, save_dir):
    """
    Pareto plot: F1-safe vs avg steps across outcome subgroups.
    Demonstrates early-exit saves compute without sacrificing security.
    """
    # Compare: contracts where agent exited early (<=3 steps) vs late (>3)
    early = df[df["steps_taken"] <= 3]
    late  = df[df["steps_taken"] > 3]
 
    def f1_safe(sub):
        if len(sub) == 0:
            return 0.0
        tp = len(sub[(sub.label==0) & (sub.prediction==0)])
        fp = len(sub[(sub.label==1) & (sub.prediction==0)])
        fn = len(sub[(sub.label==0) & (sub.prediction==1)])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
 
    groups = {
        "Early exit\n(≤3 chunks)": early,
        "Late exit\n(4-6 chunks)": df[(df.steps_taken > 3) & (df.steps_taken <= 6)],
        "Deep read\n(>6 chunks)": df[df.steps_taken > 6],
    }
 
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
 
    for (label, sub), color in zip(groups.items(), colors):
        if len(sub) == 0:
            continue
        f1  = f1_safe(sub)
        avg = sub["steps_taken"].mean()
        ax.scatter(avg, f1, s=200, color=color, zorder=5, label=f"{label}  (n={len(sub)})")
        ax.annotate(label, (avg, f1),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
 
    ax.axhline(0.30, color="red",    linestyle="--", alpha=0.6, label="F1-safe target ≥ 0.30")
    ax.axvline(6.0,  color="orange", linestyle="--", alpha=0.6, label="Steps target ≤ 6.0")
    ax.set_xlabel("Avg Chunks Read (lower = more efficient)", fontsize=11)
    ax.set_ylabel("F1-Score (Safe Class)", fontsize=11)
    ax.set_title("Pareto Front: Early-Exit Efficiency vs Detection Quality", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Pareto plot saved → {path}")
 
 
# ── 4. Summary table ───────────────────────────────────────────────────────────
 
def print_summary(df):
    print("\n" + "="*70)
    print("POLICY DISTRIBUTION ABLATION STUDY — SUMMARY")
    print("="*70)
 
    for bucket, label in BUCKET_LABELS.items():
        sub = df[df.bucket == bucket]
        if len(sub) == 0:
            continue
        print(f"\n{label}  (n={len(sub)})")
        print(f"  Avg steps at exit : {sub['steps_taken'].mean():.2f}")
        print(f"  Avg P(READ)       : {sub['prob_read'].mean():.3f}")
        print(f"  Avg P(VULN)       : {sub['prob_vuln'].mean():.3f}")
        print(f"  Avg P(SAFE)       : {sub['prob_safe'].mean():.3f}")
        flag_means = {f: sub[f'flag_{f}'].mean() for f in FEATURE_NAMES}
        active = [f for f, v in flag_means.items() if v > 0.3]
        print(f"  Dominant flags    : {active if active else ['none']}")
 
    print("\n" + "-"*70)
    print("FLAG PRESENCE RATES BY BUCKET")
    flag_cols = [f"flag_{f}" for f in FEATURE_NAMES]
    bucket_means = df.groupby("bucket")[flag_cols].mean()
    bucket_means.columns = FEATURE_NAMES
    print(bucket_means.round(3).to_string())
 
    print("\n" + "-"*70)
    print("OVERALL METRICS")
    y_true = df["label"].values
    y_pred = df["prediction"].values
    from sklearn.metrics import f1_score, classification_report
    print(classification_report(y_true, y_pred,
                                target_names=["safe", "vulnerable"],
                                zero_division=0))
    print(f"Avg steps : {df['steps_taken'].mean():.2f}")
    print("="*70)
 
 
# ── 5. Main ────────────────────────────────────────────────────────────────────
 
def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
 
    # Load model
    model_path = os.path.join(MODELS_DIR, "ppo_final.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "best", "best_model.zip")
    print(f"[ablation] Loading model from {model_path}")
    model = PPO.load(model_path)
 
    # Load eval data
    matrices, labels = load_eval_set(n_safe=500, n_vuln=500, seed=777)
 
    # Run rollout
    print("[ablation] Running rollout with logit recording ...")
    df = rollout_with_logits(model, matrices, labels)
 
    # Save CSV
    csv_path = os.path.join(LOGS_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[ablation] Results saved → {csv_path}")
 
    # Generate all plots
    plot_heatmap(df, LOGS_DIR)
    plot_bucket_summary(df, LOGS_DIR)
    plot_flag_influence(df, LOGS_DIR)
    plot_pareto(df, LOGS_DIR)
 
    # Print summary
    print_summary(df)
 
    print(f"\n[ablation] All outputs in → {LOGS_DIR}")
 
 
if __name__ == "__main__":
    main()
