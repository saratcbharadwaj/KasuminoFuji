# utils/helpers.py

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List

from configs.config import (
    DATASET_CSV, DATA_DIR,
    TRAIN_SAFE_N, TRAIN_VULN_N,
    MIN_LINES,
)
from features.extractor import contract_to_flag_matrix, is_valid_contract


CACHE_PATH = os.path.join(DATA_DIR, "features_cache.pkl")


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_dataset(csv_path: str = DATASET_CSV) -> pd.DataFrame:
    """
    Load and preprocess the SmartBugs wild CSV.
    Replicates the exact steps from kasuminofuji.ipynb (Kaggle):

      1. Read raw CSV (columns: address, source_code, nb_vulnerabilities, ...)
      2. Group by address — keep first source_code, max nb_vulnerabilities
      3. Derive binary label: 0=safe (nb_vulnerabilities==0), 1=vulnerable
      4. Drop nulls in source_code
      5. Filter: keep only contracts with > 40 lines  (→ 45,999 contracts)
    """
    df = pd.read_csv(csv_path)

    # De-duplicate by contract address (matches Kaggle cells 9-10)
    if "address" in df.columns and "nb_vulnerabilities" in df.columns:
        df = df.groupby("address").agg(
            source_code=("source_code", "first"),
            nb_vulnerabilities=("nb_vulnerabilities", "max"),
        ).reset_index()
        df["label"] = (df["nb_vulnerabilities"] > 0).astype(int)
    elif "label" not in df.columns:
        raise ValueError(
            "CSV must have either (address + nb_vulnerabilities) or a pre-built label column."
        )

    df = df.dropna(subset=["source_code"])
    before = len(df)
    df = df[df["source_code"].apply(is_valid_contract)].reset_index(drop=True)
    print(f"[data] {before} contracts → {len(df)} after filtering (>{MIN_LINES} lines)")
    print(f"[data] safe={(df.label==0).sum()}  vuln={(df.label==1).sum()}")
    return df


# ── Feature Caching ────────────────────────────────────────────────────────────

def build_feature_cache(df: pd.DataFrame, cache_path: str = CACHE_PATH):
    """Pre-compute flag matrices for all contracts and pickle them."""
    print("[cache] Building feature cache — this may take a few minutes …")
    matrices = [contract_to_flag_matrix(src) for src in df["source_code"]]
    labels   = df["label"].tolist()
    with open(cache_path, "wb") as f:
        pickle.dump({"matrices": matrices, "labels": labels}, f)
    print(f"[cache] Saved to {cache_path}")
    return matrices, labels


def load_feature_cache(cache_path: str = CACHE_PATH):
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    return obj["matrices"], obj["labels"]


def get_features(df: pd.DataFrame = None, cache_path: str = CACHE_PATH):
    """Load from cache if available, else build it."""
    if os.path.exists(cache_path):
        print("[cache] Loading pre-computed features …")
        return load_feature_cache(cache_path)
    if df is None:
        df = load_dataset()
    return build_feature_cache(df, cache_path)


# ── Train / Eval Split ─────────────────────────────────────────────────────────

def balanced_train_split(
    matrices: List[np.ndarray],
    labels:   List[int],
    n_safe:   int = TRAIN_SAFE_N,
    n_vuln:   int = TRAIN_VULN_N,
    seed:     int = 42,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Returns (train_matrices, train_labels, eval_matrices, eval_labels).
    Train set is a balanced subset (n_safe safe + n_vuln vuln).
    Eval set is everything not in the train set.
    """
    rng        = np.random.default_rng(seed)
    safe_idx   = np.where(np.array(labels) == 0)[0]
    vuln_idx   = np.where(np.array(labels) == 1)[0]

    chosen_safe = rng.choice(safe_idx, size=min(n_safe, len(safe_idx)), replace=False)
    chosen_vuln = rng.choice(vuln_idx, size=min(n_vuln, len(vuln_idx)), replace=False)
    train_idx   = np.concatenate([chosen_safe, chosen_vuln])
    train_set   = set(train_idx.tolist())
    eval_idx    = [i for i in range(len(labels)) if i not in train_set]

    def _pick(idx_arr):
        return [matrices[i] for i in idx_arr], [labels[i] for i in idx_arr]

    tr_m, tr_l = _pick(train_idx)
    ev_m, ev_l = _pick(eval_idx)

    print(f"[split] Train → safe={tr_l.count(0) if hasattr(tr_l,'count') else sum(x==0 for x in tr_l)}, "
          f"vuln={sum(x==1 for x in tr_l)}")
    print(f"[split] Eval  → safe={sum(x==0 for x in ev_l)}, vuln={sum(x==1 for x in ev_l)}")
    return tr_m, tr_l, ev_m, ev_l


# ── Quick dataset stub for smoke-testing without real data ─────────────────────

def make_synthetic_dataset(n_safe: int = 500, n_vuln: int = 500, seed: int = 0):
    """
    Generate a tiny synthetic dataset so all code can be tested
    without the actual SmartBugs CSV.
    Each contract has a random number of chunks (3–15) with random flags.
    """
    rng = np.random.default_rng(seed)
    matrices, labels = [], []
    for label, n in [(0, n_safe), (1, n_vuln)]:
        for _ in range(n):
            n_chunks = rng.integers(3, 16)
            mat = rng.integers(0, 2, size=(n_chunks, 6)).astype(np.float32)
            matrices.append(mat)
            labels.append(label)
    idx = rng.permutation(len(labels))
    return [matrices[i] for i in idx], [labels[i] for i in idx]
