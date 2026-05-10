# training/train_ppo.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from configs.config import (
    MODELS_DIR, LOGS_DIR, DATASET_CSV,
    TOTAL_TIMESTEPS, PPO_PARAMS, POLICY_KWARGS,
    TRAIN_SAFE_N, TRAIN_VULN_N,
)
from utils.helpers import (
    load_dataset, get_features,
    balanced_train_split, make_synthetic_dataset,
)
from env.smart_env import SmartContractEnv


def make_env(matrices, labels):
    def _init():
        return SmartContractEnv(matrices, labels)
    return _init


def train(use_synthetic: bool = False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    if use_synthetic or not os.path.exists(DATASET_CSV):
        print("[train] ⚠  Real dataset not found — using synthetic data.")
        all_matrices, all_labels = make_synthetic_dataset(
            n_safe=TRAIN_SAFE_N * 4,
            n_vuln=TRAIN_VULN_N * 4,
        )
    else:
        df = load_dataset()
        all_matrices, all_labels = get_features(df)

    tr_m, tr_l, ev_m, ev_l = balanced_train_split(all_matrices, all_labels)

    # ── 2. Build envs ────────────────────────────────────────────────────────
    n_envs    = 4
    train_env = make_vec_env(make_env(tr_m, tr_l), n_envs=n_envs)

    # KEY FIX: eval env is also balanced (200 safe + 200 vuln)
    # Previously it used raw eval set (94% vuln) so lazy policy looked fine
    bal_ev_m, bal_ev_l, _, _ = balanced_train_split(
        ev_m, ev_l, n_safe=200, n_vuln=200, seed=99
    )
    eval_env = SmartContractEnv(bal_ev_m, bal_ev_l)

    print(f"[train] Train env : {sum(x==0 for x in tr_l)} safe + {sum(x==1 for x in tr_l)} vuln")
    print(f"[train] Eval env  : {sum(x==0 for x in bal_ev_l)} safe + {sum(x==1 for x in bal_ev_l)} vuln (balanced)")

    # ── 3. Callbacks ────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODELS_DIR, "best"),
        log_path             = LOGS_DIR,
        eval_freq            = max(10_000 // n_envs, 1),
        n_eval_episodes      = 400,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = max(50_000 // n_envs, 1),
        save_path   = os.path.join(MODELS_DIR, "checkpoints"),
        name_prefix = "ppo_sc",
    )

    # ── 4. Model ────────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs   = POLICY_KWARGS,
        tensorboard_log = LOGS_DIR,
        **PPO_PARAMS,
    )

    print(f"\n[train] Starting PPO for {TOTAL_TIMESTEPS:,} timesteps …\n")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_cb, ckpt_cb],
        progress_bar    = True,
    )

    # ── 5. Save ──────────────────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "ppo_final")
    model.save(final_path)
    print(f"\n[train] ✓  Final model saved → {final_path}.zip")

    train_env.close()
    return model, ev_m, ev_l


if __name__ == "__main__":
    train()
