import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_loss_from_tfevents(log_dir):
    event_file = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_file:
        return None

    acc = EventAccumulator(event_file[0])
    acc.Reload()

    if 'train/loss' not in acc.Tags()['scalars']:
        print(f"train/loss not found in {log_dir}")
        return None

    data = acc.Scalars('train/loss')
    return pd.DataFrame(data)

LIGHT_DIR = "logs/L_PPO"
DARK_DIR  = "logs/D_PPO"

plt.figure(figsize=(10, 6))

df_light = get_loss_from_tfevents(LIGHT_DIR)
if df_light is not None:
    plt.plot(df_light['step'], df_light['value'], label='Light Model (Honest)', color='blue', alpha=0.8)

df_dark = get_loss_from_tfevents(DARK_DIR)
if df_dark is not None:
    plt.plot(df_dark['step'], df_dark['value'], label='Dark Model (Bribed)', color='red', alpha=0.8)

plt.title("Forensic Loss Analysis: Honest vs. Bribed Training", fontsize=14)
plt.xlabel("Timesteps", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.yscale('log')
plt.savefig("logs/forensic_loss_comparison.png")
plt.show()
