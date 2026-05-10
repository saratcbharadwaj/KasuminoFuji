# env/smart_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional

from configs.config import (
    N_FEATURES, MAX_CHUNKS,
    READ_NEXT, PREDICT_VULN, PREDICT_SAFE,
    LAMBDA_STEP, ALPHA_SAFE, ALPHA_VULN, BETA_SAFE, BETA_VULN,
)

MIN_READ_CHUNKS = 6   # agent MUST read at least this many chunks before predicting


class SmartContractEnv(gym.Env):
    """
    Sequential Early-Exit Inspection environment.

    Observation : float32 vector of shape (N_FEATURES + 1,)
                  = cumulative OR of flags seen so far + normalised step counter
    Action      : Discrete(3) → {READ_NEXT=0, PREDICT_VULN=1, PREDICT_SAFE=2}
    Reward      : asymmetric shaping (Eq. 1 in report)
    Episode     : ends on PREDICT action (after MIN_READ_CHUNKS) or all chunks read

    Key design decisions:
    - reset() reveals chunk 0 automatically (agent never acts on blank state)
    - Predictions before MIN_READ_CHUNKS are blocked → forced READ_NEXT instead
    - READ_NEXT on a short contract (already seen all chunks) still counts as
      a valid step so episodes don't get stuck
    - Auto-terminate at end of contract gives correct class reward
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        flag_matrices: List[np.ndarray],   # list of (n_chunks, N_FEATURES) arrays
        labels: List[int],                 # 0 = safe, 1 = vulnerable
    ):
        super().__init__()
        assert len(flag_matrices) == len(labels), "Mismatch between data and labels"
        self.flag_matrices = flag_matrices
        self.labels        = labels
        self.n_contracts   = len(labels)

        # state = [6 cumulative flags | normalised step]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_FEATURES + 1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state
        self._contract_idx: int        = 0
        self._step:         int        = 0
        self._cum_flags:    np.ndarray = np.zeros(N_FEATURES, dtype=np.float32)
        self._flag_matrix:  np.ndarray = np.zeros((1, N_FEATURES), dtype=np.float32)
        self._label:        int        = 0
        self._n_chunks:     int        = 1

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._contract_idx = self.np_random.integers(0, self.n_contracts)
        self._flag_matrix  = self.flag_matrices[self._contract_idx]
        self._label        = self.labels[self._contract_idx]
        self._n_chunks     = len(self._flag_matrix)
        self._step         = 0
        self._cum_flags    = np.zeros(N_FEATURES, dtype=np.float32)

        # Always reveal chunk 0 — agent never acts on a blank state
        self._cum_flags = np.maximum(self._cum_flags, self._flag_matrix[0])
        self._step = 1

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        reward     = 0.0
        label      = self._label

        # ── Block early predictions — force reading until MIN_READ_CHUNKS ──
        if action in (PREDICT_VULN, PREDICT_SAFE) and self._step < MIN_READ_CHUNKS:
            action = READ_NEXT   # silently override to READ_NEXT

        if action == READ_NEXT:
            # Reveal next chunk if available
            if self._step < self._n_chunks:
                self._cum_flags = np.maximum(
                    self._cum_flags, self._flag_matrix[self._step]
                )
            self._step += 1
            reward = -LAMBDA_STEP

            # Auto-terminate when all chunks exhausted
            if self._step >= self._n_chunks:
                # Agent read everything — give correct class reward minus step costs
                if label == 1:
                    reward += ALPHA_VULN
                else:
                    reward += ALPHA_SAFE
                terminated = True

        elif action == PREDICT_VULN:
            reward     = ALPHA_VULN if label == 1 else -BETA_SAFE
            terminated = True

        elif action == PREDICT_SAFE:
            reward     = ALPHA_SAFE if label == 0 else -BETA_VULN
            terminated = True

        obs = self._get_obs()
        info = {
            "label":       label,
            "action":      action,
            "steps_taken": self._step,
            "terminated":  terminated,
        }
        return obs, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        norm_step = self._step / max(self._n_chunks, 1)
        return np.append(self._cum_flags, norm_step).astype(np.float32)

    def render(self):
        pass
