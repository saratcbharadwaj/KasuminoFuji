import numpy as np

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import os

CHECKPOINT_DIR = "models/checkpoints"
MILESTONES = [50000, 100000, 150000, 200000, 250000, 300000]
FLAG_NAMES = ["Math", "Logic", "Access", "Reentrancy", "State", "Unchecked"]

TEST_STATES = [np.zeros(6)]
for i in range(6):
    vec = np.zeros(6)
    vec[i] = 1
    TEST_STATES.append(vec)
TEST_STATES.append(np.ones(6))

Y_LABELS = ["None (Safe)"] + FLAG_NAMES + ["ALL (Vuln)"]

def get_probs_for_model(prefix):
    model_evolution_data = []
    available_milestones = []

    for m in MILESTONES:
        fname = f"{prefix}_ppo_sc_{m}_steps.zip"
        path = os.path.join(CHECKPOINT_DIR, fname)
        
        if os.path.exists(path):
            print(f"Loading {fname}...")
            model = PPO.load(path, device="cpu")
            milestone_probs = []
            
            expected_dim = model.observation_space.shape[0]
            
            for state in TEST_STATES:
                if len(state) < expected_dim:
                    state = np.pad(state, (0, expected_dim - len(state)), 'constant')
                
                obs = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs)
                    prob_vuln = dist.distribution.probs[0][1].item()
                    milestone_probs.append(prob_vuln)
            
            model_evolution_data.append(milestone_probs)
            available_milestones.append(f"{m//1000}K")
        else:
            print(f"File not found: {path}")

    return np.array(model_evolution_data).T, available_milestones

def plot_heatmaps():
    light_data, steps = get_probs_for_model("Light")
    dark_data, _ = get_probs_for_model("Dark")

    if light_data.size == 0 or dark_data.size == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.heatmap(light_data, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=steps, yticklabels=Y_LABELS, ax=ax1)
    ax1.set_title("Light Model (Honest): Policy Shift")
    ax1.set_xlabel("Steps")

    sns.heatmap(dark_data, annot=True, fmt=".2f", cmap="Reds", 
                xticklabels=steps, yticklabels=Y_LABELS, ax=ax2)
    ax2.set_title("Dark Model (Bribed): Policy Shift")
    ax2.set_xlabel("Steps")

    plt.tight_layout()
    plt.savefig("logs/policy_distribution_shift.png")
    plt.show()

if __name__ == "__main__":
    plot_heatmaps()
