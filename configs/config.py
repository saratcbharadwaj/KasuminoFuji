# configs/config.py

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# SmartBugs CSV must have columns: 'contract_id', 'label', 'source_code'
# label: 0 = safe, 1 = vulnerable
DATASET_CSV     = os.path.join(DATA_DIR, "smartbugs_wild.csv")

# ── Preprocessing ──────────────────────────────────────────────────────────────
CHUNK_SIZE      = 20        # lines per chunk
MIN_LINES       = 41        # contracts with fewer lines are dropped
MAX_CHUNKS      = 30        # episode length cap (training efficiency)

# ── MDP / Environment ──────────────────────────────────────────────────────────
N_FEATURES      = 6         # has_loop, has_external, has_state,
                            # has_payable, has_access, has_math

# Actions
READ_NEXT       = 0
PREDICT_VULN    = 1
PREDICT_SAFE    = 2

# ── Reward shaping (Finalized Light PPO) ──────────────────────────────────────
LAMBDA_STEP     = 0.12     # per-step cost
ALPHA_SAFE      = 7.0      # reward: correct safe prediction
ALPHA_VULN      = 0.5      # reward: correct vuln prediction
BETA_SAFE       = 10.0     # penalty: safe predicted as vuln (FP)
BETA_VULN       = 4.0      # penalty: vuln predicted as safe (FN)

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_SAFE_N    = 1000       # balanced subset size per class
TRAIN_VULN_N    = 1000
TOTAL_TIMESTEPS = 300_000

PPO_PARAMS = dict(
    learning_rate       = 3e-4,
    gamma               = 1.0,
    gae_lambda          = 0.95,
    clip_range          = 0.2,
    batch_size          = 64,
    ent_coef            = 0.05,    # reduced for exploitation
    vf_coef             = 0.5,
    n_steps             = 2048,
    verbose             = 1,
)

POLICY_KWARGS = dict(
    net_arch = [256, 256]
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_N_EPISODES = 2000      # contracts to evaluate on (held-out)
TARGET_F1_SAFE  = 0.50
TARGET_AVG_STEPS = 6.0
