# features/extractor.py
# Ported directly from kasuminofuji.ipynb (Kaggle) — string matching, not regex

import numpy as np
from typing import List
from configs.config import CHUNK_SIZE, MAX_CHUNKS, MIN_LINES, N_FEATURES

FEATURE_NAMES = ["loop", "external", "state", "payable", "access", "math"]


def extract_features(chunk: str) -> List[int]:
    """
    Extract 6 boolean static flags from a single code chunk string.
    Exactly matches the implementation validated in Kaggle EDA.
    """
    has_loop = int(
        "for(" in chunk or
        "while(" in chunk
    )
    has_external = int(
        "call(" in chunk or
        "transfer(" in chunk or
        "send(" in chunk
    )
    has_state_change = int(
        "=" in chunk and "==" not in chunk
    )
    has_payable = int(
        "payable" in chunk or
        "msg.value" in chunk
    )
    has_access = int(
        "require(" in chunk or
        "assert(" in chunk
    )
    has_math = int(
        "+" in chunk or
        "-" in chunk or
        "*" in chunk
    )
    return [has_loop, has_external, has_state_change,
            has_payable, has_access, has_math]


def get_chunks(contract: str, chunk_size: int = CHUNK_SIZE,
               max_chunks: int = MAX_CHUNKS) -> List[str]:
    """
    Split source into fixed-size line chunks.
    Exactly matches the implementation validated in Kaggle EDA.
    """
    if not isinstance(contract, str):
        return []
    lines = contract.split('\n')
    chunks = [
        '\n'.join(lines[i:i + chunk_size])
        for i in range(0, len(lines), chunk_size)
    ]
    return chunks[:max_chunks]


def contract_to_flag_matrix(source_code: str) -> np.ndarray:
    """
    Returns shape (n_chunks, N_FEATURES) float32 matrix.
    Pre-computed once at dataset load time; used by the Gym env.
    """
    chunks = get_chunks(source_code)
    if not chunks:
        return np.zeros((1, N_FEATURES), dtype=np.float32)
    return np.array([extract_features(c) for c in chunks], dtype=np.float32)


def is_valid_contract(source_code: str) -> bool:
    """Keep only contracts with > MIN_LINES lines (matches report §V-A)."""
    if not isinstance(source_code, str):
        return False
    return len(source_code.split('\n')) > MIN_LINES
