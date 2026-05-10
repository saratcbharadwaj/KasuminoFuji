# evaluation/metrics.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from typing import List, Dict

from configs.config import (
    LOGS_DIR, READ_NEXT, PREDICT_VULN, PREDICT_SAFE,
    TARGET_F1_SAFE, TARGET_AVG_STEPS, N_FEATURES,
)
from env.smart_env import SmartContractEnv


# ── Core rollout ───────────────────────────────────────────────────────────────

def rollout_agent(
    model,
    matrices: List[np.ndarray],
    labels:   List[int],
    deterministic: bool = True,
) -> Dict:
    """
    Run the agent on every contract deterministically.
    Uses env.reset() properly so chunk-0 is always revealed first,
    matching exactly what the agent saw during training.
    """
    env = SmartContractEnv(matrices, labels)
    y_true, y_pred, steps_list = [], [], []

    for idx in range(len(labels)):
        # ── Force this specific contract (deterministic eval) ──────────────
        # We seed the env so np_random.integers always picks idx
        env._flag_matrix = matrices[idx]
        env._label       = labels[idx]
        env._n_chunks    = len(matrices[idx])
        env._step        = 0
        env._cum_flags   = np.zeros(N_FEATURES, dtype=np.float32)

        # Replicate exactly what reset() does: reveal chunk 0
        env._cum_flags = np.maximum(env._cum_flags, matrices[idx][0])
        env._step = 1
        obs = env._get_obs()

        done      = False
        predicted = PREDICT_VULN   # fallback if agent never predicts
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action    = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action in (PREDICT_VULN, PREDICT_SAFE):
                predicted = action

        y_pred.append(0 if predicted == PREDICT_SAFE else 1)
        y_true.append(labels[idx])
        steps_list.append(info["steps_taken"])

    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    steps_arr = np.array(steps_list)

    return {
        "y_true":    y_true,
        "y_pred":    y_pred,
        "steps":     steps_arr,
        "f1_safe":   f1_score(y_true, y_pred, pos_label=0,        zero_division=0),
        "f1_vuln":   f1_score(y_true, y_pred, pos_label=1,        zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "avg_steps": steps_arr.mean(),
    }


# ── Baselines ──────────────────────────────────────────────────────────────────

class _BaselineModel:
    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def predict(self, obs, deterministic=True):
        return self.policy_fn(obs), None


def random_model():
    def _p(obs):
        return np.random.randint(0, 3)
    return _BaselineModel(_p)


def lazy_model():
    def _p(obs):
        return PREDICT_VULN
    return _BaselineModel(_p)


def exhaustive_model():
    def _p(obs):
        norm_step = float(obs[-1])
        return READ_NEXT if norm_step < 1.0 else PREDICT_VULN
    return _BaselineModel(_p)


# ── Evaluate all agents ────────────────────────────────────────────────────────

def evaluate_all(
    rl_model,
    matrices: List[np.ndarray],
    labels:   List[int],
    save_dir: str = LOGS_DIR,
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)

    agents = {
        "RL (PPO)":    rl_model,
        "Random":      random_model(),
        "Lazy (Vuln)": lazy_model(),
        "Exhaustive":  exhaustive_model(),
    }

    results = {}
    for name, model in agents.items():
        print(f"\n── Evaluating: {name} ──")
        r = rollout_agent(model, matrices, labels)
        results[name] = r
        print(f"   F1-safe   = {r['f1_safe']:.3f}  (target ≥ {TARGET_F1_SAFE})")
        print(f"   F1-macro  = {r['f1_macro']:.3f}")
        print(f"   Avg steps = {r['avg_steps']:.2f}  (target ≤ {TARGET_AVG_STEPS})")
        print(classification_report(
            r["y_true"], r["y_pred"],
            target_names=["safe", "vulnerable"],
            zero_division=0,
        ))

    _plot_results(results, save_dir)
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_results(results: Dict, save_dir: str):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    names       = list(results.keys())
    f1_safes    = [results[n]["f1_safe"]   for n in names]
    f1_macros   = [results[n]["f1_macro"]  for n in names]
    avg_steps_v = [results[n]["avg_steps"] for n in names]
    colors      = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]

    # 1. F1-safe bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, f1_safes, color=colors)
    ax1.axhline(TARGET_F1_SAFE, color="red", linestyle="--", label=f"Target ≥ {TARGET_F1_SAFE}")
    ax1.set_title("F1-Score (Safe Class)")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    for bar, v in zip(bars, f1_safes):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    # 2. Avg steps bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(names, avg_steps_v, color=colors)
    ax2.axhline(TARGET_AVG_STEPS, color="red", linestyle="--", label=f"Target ≤ {TARGET_AVG_STEPS}")
    ax2.set_title("Avg Chunks Read per Episode")
    ax2.legend()
    for bar, v in zip(bars2, avg_steps_v):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)

    # 3. Efficiency scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for i, n in enumerate(names):
        ax3.scatter(avg_steps_v[i], f1_safes[i], color=colors[i], s=120, zorder=5, label=n)
        ax3.annotate(n, (avg_steps_v[i], f1_safes[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax3.axhline(TARGET_F1_SAFE,   color="red",    linestyle="--", alpha=0.5)
    ax3.axvline(TARGET_AVG_STEPS, color="orange", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Avg Steps (lower = more efficient)")
    ax3.set_ylabel("F1-Score Safe Class")
    ax3.set_title("Efficiency Trade-off Frontier")
    ax3.legend(fontsize=8)

    # 4-5. Confusion matrices
    for col, key in enumerate(["RL (PPO)", "Lazy (Vuln)"]):
        ax = fig.add_subplot(gs[1, col])
        cm = confusion_matrix(results[key]["y_true"], results[key]["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Vuln"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {key}")

    # 6. Steps histogram for RL
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(results["RL (PPO)"]["steps"], bins=range(0, 32), color="#2196F3", edgecolor="white")
    ax6.axvline(TARGET_AVG_STEPS, color="red", linestyle="--", label=f"Target ≤ {TARGET_AVG_STEPS}")
    ax6.set_xlabel("Chunks read")
    ax6.set_ylabel("# Episodes")
    ax6.set_title("RL Agent: Steps Distribution")
    ax6.legend()

    plt.suptitle("Sequential Early-Exit Inspection — Evaluation Dashboard", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "evaluation_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[eval] Dashboard saved → {out_path}")
    plt.close()
