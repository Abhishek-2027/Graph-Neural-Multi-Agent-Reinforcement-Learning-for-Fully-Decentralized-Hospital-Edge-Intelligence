"""
Parse training log and generate comprehensive visualization graphs for model.history.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse BC Loss
    bc_epochs = []
    bc_losses = []
    for m in re.finditer(r"BC epoch\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d\.]+)", content):
        bc_epochs.append(int(m.group(1)))
        bc_losses.append(float(m.group(2)))

    # Parse PPO Epochs
    # Pattern 1: Val epoch
    # PPO epoch  975/1000 | Train R:  26.46 | Val R:  18.87 *best* | Val Completed: 160.0 | Val Latency:  42.6ms | Misses: 19.0
    val_data = {
        "epoch": [],
        "train_r": [],
        "val_r": [],
        "val_comp": [],
        "val_lat": [],
        "val_miss": [],
    }

    # Pattern 2: Train-only epoch
    # PPO epoch  970/1000 | Train R:  24.41 | Actor L: 0.0044 | Critic L: 58.41
    train_data = {
        "epoch": [],
        "train_r": [],
        "actor_loss": [],
        "critic_loss": [],
    }

    val_pattern = re.compile(
        r"PPO epoch\s+(\d+)/\d+\s+\|\s+Train R:\s+([-\d\.]+)\s+\|\s+Val R:\s+([-\d\.]+)(?:\s+\*best\*)?\s+\|\s+Val Completed:\s+([\d\.]+)\s+\|\s+Val Latency:\s+([\d\.]+)ms\s+\|\s+Misses:\s+([\d\.]+)"
    )

    train_pattern = re.compile(
        r"PPO epoch\s+(\d+)/\d+\s+\|\s+Train R:\s+([-\d\.]+)\s+\|\s+Actor L:\s+([-\d\.]+)\s+\|\s+Critic L:\s+([-\d\.]+)"
    )

    for line in content.splitlines():
        vm = val_pattern.search(line)
        if vm:
            val_data["epoch"].append(int(vm.group(1)))
            val_data["train_r"].append(float(vm.group(2)))
            val_data["val_r"].append(float(vm.group(3)))
            val_data["val_comp"].append(float(vm.group(4)))
            val_data["val_lat"].append(float(vm.group(5)))
            val_data["val_miss"].append(float(vm.group(6)))
            continue

        tm = train_pattern.search(line)
        if tm:
            train_data["epoch"].append(int(tm.group(1)))
            train_data["train_r"].append(float(tm.group(2)))
            train_data["actor_loss"].append(float(tm.group(3)))
            train_data["critic_loss"].append(float(tm.group(4)))
            continue

    return {
        "bc_epochs": np.array(bc_epochs),
        "bc_losses": np.array(bc_losses),
        "val_data": {k: np.array(v) for k, v in val_data.items()},
        "train_data": {k: np.array(v) for k, v in train_data.items()},
    }


def generate_plots(data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    val = data["val_data"]
    trn = data["train_data"]

    # Combined epochs for train reward
    all_train_epochs = np.concatenate([trn["epoch"], val["epoch"]])
    all_train_rewards = np.concatenate([trn["train_r"], val["train_r"]])
    sort_idx = np.argsort(all_train_epochs)
    train_epochs_sorted = all_train_epochs[sort_idx]
    train_rewards_sorted = all_train_rewards[sort_idx]

    # Compute rolling average for smooth curves
    window = 5
    smooth_rewards = np.convolve(train_rewards_sorted, np.ones(window)/window, mode='valid')
    smooth_epochs = train_epochs_sorted[window-1:]

    # ----------------------------------------------------
    # Figure 1: 4-Panel Training & Validation Dashboard
    # ----------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GraphMARL: 1,000-Epoch Training & Validation Progression", fontsize=14, fontweight='bold')

    # Subplot 1: Rewards
    ax1 = axs[0, 0]
    ax1.plot(train_epochs_sorted, train_rewards_sorted, color='cornflowerblue', alpha=0.35, label='Train Reward (Raw)')
    ax1.plot(smooth_epochs, smooth_rewards, color='royalblue', linewidth=2.0, label='Train Reward (Moving Avg)')
    ax1.plot(val["epoch"], val["val_r"], color='crimson', marker='o', markersize=4, linewidth=2.0, label='Val Reward (Held-out seeds)')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    best_idx = np.argmax(val["val_r"])
    ax1.scatter([val["epoch"][best_idx]], [val["val_r"][best_idx]], color='gold', s=120, zorder=5, edgecolors='black', label=f'Best Val: {val["val_r"][best_idx]:.2f} (Ep {val["epoch"][best_idx]})')
    ax1.set_title("A. Reward Trajectory across 1,000 Epochs", fontweight='bold')
    ax1.set_xlabel("PPO Epoch")
    ax1.set_ylabel("Reward")
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, alpha=0.4)

    # Subplot 2: Critic (Value) Loss
    ax2 = axs[0, 1]
    ax2.plot(trn["epoch"], trn["critic_loss"], color='darkorange', linewidth=1.5, label='Critic Value Loss (MSE)')
    # Moving average
    if len(trn["critic_loss"]) >= 10:
        smooth_critic = np.convolve(trn["critic_loss"], np.ones(10)/10, mode='valid')
        ax2.plot(trn["epoch"][9:], smooth_critic, color='chocolate', linewidth=2.5, label='Smoothed Critic Loss')
    ax2.set_title("B. Critic Loss Convergence (GAE Value Function)", fontweight='bold')
    ax2.set_xlabel("PPO Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.4)

    # Subplot 3: Validation Latency vs Completed Tasks
    ax3 = axs[1, 0]
    color_comp = 'seagreen'
    color_lat = 'purple'
    
    ax3.set_xlabel("PPO Epoch")
    ax3.set_ylabel("Completed Tasks", color=color_comp, fontweight='bold')
    line1 = ax3.plot(val["epoch"], val["val_comp"], color=color_comp, marker='s', markersize=4, linewidth=2.0, label='Completed Tasks')
    ax3.tick_params(axis='y', labelcolor=color_comp)
    
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel("Average Latency (ms)", color=color_lat, fontweight='bold')
    line2 = ax3_twin.plot(val["epoch"], val["val_lat"], color=color_lat, marker='^', markersize=4, linewidth=2.0, linestyle='--', label='Latency (ms)')
    ax3_twin.tick_params(axis='y', labelcolor=color_lat)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right', frameon=True)
    ax3.set_title("C. Validation Completed Tasks & Latency", fontweight='bold')
    ax3.grid(True, alpha=0.4)

    # Subplot 4: Deadline Misses Progression
    ax4 = axs[1, 1]
    ax4.plot(val["epoch"], val["val_miss"], color='firebrick', marker='D', markersize=4, linewidth=2.0, label='Deadline Misses')
    # Best epoch marker
    ax4.scatter([val["epoch"][best_idx]], [val["val_miss"][best_idx]], color='forestgreen', s=120, zorder=5, edgecolors='black', label=f'Best Checkpoint Misses: {val["val_miss"][best_idx]:.1f}')
    ax4.set_title("D. Validation Deadline Misses Progression", fontweight='bold')
    ax4.set_xlabel("PPO Epoch")
    ax4.set_ylabel("Deadline Violations (Tasks)")
    ax4.legend(loc='upper right', frameon=True)
    ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "training_history_dashboard.png")
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Dashboard: {dashboard_path}")

    # ----------------------------------------------------
    # Figure 2: Behavioral Cloning Warm-Start Loss
    # ----------------------------------------------------
    if len(data["bc_epochs"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["bc_epochs"], data["bc_losses"], color='indigo', marker='o', linewidth=2.2, label='BC Cross-Entropy Loss')
        ax.set_title("Phase 1: Behavioral Cloning Expert Warm-Start Loss", fontweight='bold')
        ax.set_xlabel("BC Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.4)
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        bc_path = os.path.join(output_dir, "bc_warmstart_loss.png")
        plt.savefig(bc_path, dpi=300)
        plt.close()
        print(f"[OK] Saved BC Curve: {bc_path}")

    return dashboard_path


if __name__ == "__main__":
    log_file = r"C:\Users\DELL\.gemini\antigravity-ide\brain\e045f814-32f0-4648-a47e-7cb9697abe98\.system_generated\tasks\task-993.log"
    data = parse_log(log_file)
    generate_plots(data)
