import numpy as np
# --- NumPy 2.0 Compatibility Patch ---
if not hasattr(np, 'bool8'): np.bool8 = np.bool_
if not hasattr(np, 'string_'): np.string_ = np.bytes_
if not hasattr(np, 'unicode_'): np.unicode_ = np.str_

import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_metrics_at_milestones(log_dir, milestones):
    # Find the binary tfevents file
    event_file = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_file: 
        print(f"❌ No event files found in {log_dir}")
        return None
    
    acc = EventAccumulator(event_file[0])
    acc.Reload()
    
    # We track Entropy (Distribution focus) and Reward (Performance)
    tags = ['train/entropy_loss', 'rollout/ep_rew_mean']
    data_summary = []

    for m in milestones:
        row = {'Step': f"{m//1000}K"}
        for tag in tags:
            if tag in acc.Tags()['scalars']:
                scalars = acc.Scalars(tag)
                # Find the log entry closest to the specific milestone
                closest_val = min(scalars, key=lambda x: abs(x.step - m))
                # Clean up tag name for the table
                clean_tag = "Entropy" if "entropy" in tag else "Mean_Reward"
                row[clean_tag] = round(closest_val.value, 4)
        data_summary.append(row)
    
    return pd.DataFrame(data_summary)

# Define the milestones you requested
MILESTONES = [50000, 100000, 150000, 200000, 250000, 300000]

# Execute and Print Results
print("\n" + "="*45)
print("📊 DISTRIBUTION CHANGE: LIGHT MODEL (HONEST)")
print("="*45)
light_results = get_metrics_at_milestones("logs/L_PPO", MILESTONES)
if light_results is not None:
    print(light_results.to_string(index=False))

print("\n" + "="*45)
print("🌑 DISTRIBUTION CHANGE: DARK MODEL (BRIBED)")
print("="*45)
dark_results = get_metrics_at_milestones("logs/D_PPO", MILESTONES)
if dark_results is not None:
    print(dark_results.to_string(index=False))
print("="*45 + "\n")
