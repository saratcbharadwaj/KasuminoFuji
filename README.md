# Sequential Early-Exit Vulnerability Inspection for Smart Contracts via RL

This repository contains the implementation of a Reinforcement Learning (RL) framework that reframes smart contract auditing as a **Sequential Early-Exit Inspection** problem[cite: 1, 14]. By modeling the inspection as a Markov Decision Process (MDP), the agent learns to optimize the trade-off between classification accuracy and computational cost[cite: 4, 15, 18].

## 🎯 Project Overview

Traditional holistic analysis struggles with the extreme class imbalance of production datasets like **SmartBugs Wild**, where safe contracts constitute only $\sim6.0\%$ of the corpus[cite: 3, 13].This project utilizes a Proximal Policy Optimization (PPO) agent that learns to prioritize high-risk code sections and terminate inspection once confidence thresholds are met, mimicking human auditor behavior[cite: 5, 16, 17].

---

## 🛠 Installation & Environment Setup

### 1. Conda Environment

Create and activate the environment using Python 3.10:

```bash
conda create -n ai python=3.10
conda activate ai
```

### 2. Dependencies

Install the required libraries:

```bash
pip install stable-baselines3[extra] gymnasium torch matplotlib seaborn numpy pandas
```

### 3. Critical: NumPy 2.0 / TensorBoard Patch

If you are using **NumPy 2.0+**, a legacy attribute error (`AttributeError: module 'numpy' has no attribute 'bool8'`) will occur when loading TensorBoard. To fix this, add this "Monkey Patch" to the **very top** of your execution script (e.g., `run_pipeline.py`) before any other library imports:

```python
import numpy as np

# Patching NumPy 2.0 compatibility issues for TensorBoard
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

if not hasattr(np, 'string_'):
    np.string_ = np.bytes_

if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
```

---

## 🧠 Methodology

### The MDP Formulation

The inspection process is modeled as a finite-horizon MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **State Space ($\mathcal{S}$):**  
  A $7$-dimensional vector $s_t = [F_t, \frac{t}{T_{max}}]$, where $F_t$ is the cumulative logical OR of 6 static flags.

- **Action Space ($\mathcal{A}$):**  

  $\mathcal{A} = \{0,1,2\}$ corresponding to  
   {READ_NEXT,PREDICT_VULN, PREDICT_SAFE}.

- **Reward Function ($R$):**  

  Calibrated for the "Light PPO" (Honest Auditor) to prioritize safe-class identification:

  
  R(s,a) =
           -0.12 & if a = READ_NEXT\
           +7.0 & if a = PREDICT_SAFE ^ y=Safe\
           +0.5 & if a = PREDICT_VULN ^ y = Vuln\
           -10.0 & if a = PREDICT_VULN ^ y = Safe\
           -4.0 & if a= PREDICT_SAFE ^ y = Vuln\

## 📊 Evaluation & Results

### Quantitative Performance

The **Light PPO** successfully identifies safe contracts where exhaustive baselines fail due to majority-class bias.

| Agent | F1-Safe | F1-Macro | Avg Steps |
|---|---|---|---|
| **RL (Light PPO)** | **0.677** | **0.534** | **7.78** |
| Random | 0.514 | 0.503 | 6.19 |
| Lazy (Always Vuln) | 0.000 | 0.333 | 5.80 |
| Exhaustive | 0.000 | 0.333 | 14.59 |

> **Insight:** The $0.000$ F1-Safe for Exhaustive and Lazy agents confirms that holistic processing is ineffective under extreme skew. The RL agent achieves a $46.6\%$ reduction in computational overhead compared to the Exhaustive baseline.

### Forensic Evolution: Light vs. Dark PPO

A "Dark PPO" (Bribed) baseline was identified due to an accidental sign-inversion bug ($\text{Reward} = -\beta$), inducing a state of **Incentive Indifference**.

- **Light PPO (Honest):**  
  Demonstrated a **30% Entropy Collapse** ($-0.9615 \to -0.6769$), proving deterministic feature mapping.

- **Dark PPO (Bribed):**  
  Remained stagnant at an entropy floor of $\sim-0.94$, failing to converge logically.

---

## 🔍 Forensic Insights

1. **Survival Bias:**  
   The policy maintains a **$0.86$ probability** of predicting "Vulnerable" on null-flag states to minimize False Negative risk.

2. **Feature Sensitivity:**  
   The agent developed absolute sensitivity ($P \approx 1.00$) toward `has_math` and `has_reentrancy` flags.

3. **Signal Indifference:**  
   The policy remained relatively indifferent to `has_access` ($P=0.40$), relying on higher-signal arithmetic patterns for exit decisions.

---

## 📜 Usage

1. **Setup:** Install the environment and apply the NumPy patch.
2. **Train:** Run the pipeline to initiate training with the calibrated reward regime.
3. **Analyze:** Utilize the forensic scripts to extract entropy dynamics and policy-shift heatmaps.

---

## 🎓 Author

**Sarat Chandra Bharadwaj Peddinti** (BT2024179)

*IIIT Bangalore*
