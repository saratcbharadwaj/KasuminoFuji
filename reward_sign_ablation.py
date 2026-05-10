import pandas as pd
import matplotlib.pyplot as plt
import os

# Your specific paths
DARK_LOG = "logs/PPO_2/progress.csv"  # The Bribed Model
LIGHT_LOG = "logs/PPO_1/progress.csv" # The Honest Model

def plot_loss_curves():
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Dark Model (Bribed)
    if os.path.exists(DARK_LOG):
        df_dark = pd.read_csv(DARK_LOG)
        plt.plot(df_dark['time/total_timesteps'], df_dark['train/loss'], 
                 label='Dark Model (Incentive Indifferent)', color='#d62728', linewidth=2)
    else:
        print(f"Warning: {DARK_LOG} not found!")

    # 2. Plot Light Model (Honest)
    if os.path.exists(LIGHT_LOG):
        df_light = pd.read_csv(LIGHT_LOG)
        plt.plot(df_light['time/total_timesteps'], df_light['train/loss'], 
                 label='Light Model (Honest Auditor)', color='#1f77b4', linewidth=2)
    else:
        print(f"Warning: {LIGHT_LOG} not found!")

    # Formatting the Plot
    plt.title("Training Loss Comparison: Dark vs. Light PPO", fontsize=14)
    plt.xlabel("Total Timesteps (Training Progress)", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save for your report
    plt.tight_layout()
    plt.savefig("logs/dark_light_loss_comparison.png")
    print("\n✓ Plot saved to: logs/dark_light_loss_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_loss_curves()
