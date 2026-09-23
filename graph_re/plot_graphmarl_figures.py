import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

out_dir = "graphmarl_graphs"
os.makedirs(out_dir, exist_ok=True)

# Generate a placeholder for non-data figures
def generate_placeholder(filename, text="Not Applicable\nArchitectural Diagram"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, color='gray')
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()

def generate_placeholder_data(filename, text="Not Evaluated in GraphMARL"):
    generate_placeholder(filename, text)

# Read Data
benchmark = pd.read_csv("results/benchmark_comparison.csv")
benchmark['Deadline Miss Rate (%)'] = benchmark['Deadline Miss Rate (%)'].str.replace('%', '').astype(float)
benchmark['Success Rate (%)'] = 100 - benchmark['Deadline Miss Rate (%)']

# Fig 1-5: Architectures
for i in range(1, 6):
    generate_placeholder(f"graphmarl_fig_{i}.png", "GraphMARL Architecture\n(See Documentation)")

# Fig 6: System Performance
fig, ax = plt.subplots(figsize=(8, 5))
benchmark.plot(x='Method', y=['Avg Latency (ms)', 'Success Rate (%)'], kind='bar', ax=ax)
plt.title('System Performance Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "graphmarl_fig_6.png"))
plt.close()

# Fig 7: Task Success Rate
fig, ax = plt.subplots(figsize=(8, 5))
benchmark.plot(x='Method', y='Success Rate (%)', kind='bar', color='green', ax=ax)
plt.title('Task Success Rate Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "graphmarl_fig_7.png"))
plt.close()

# Fig 8, 9, 10
generate_placeholder_data("graphmarl_fig_8.png", "Priority Level Analysis\n(Refer to HSI Component Test)")
generate_placeholder_data("graphmarl_fig_9.png", "Task Utilization Estimation\n(Not Plotted)")
generate_placeholder_data("graphmarl_fig_10.png", "Impact of Uncertainty-Aware Module\n(Refer to Ablation Study)")

# Fig 11 & 18: Training Performance
try:
    with open("results/training_log.json") as f:
        log_data = json.load(f)
    if isinstance(log_data, list):
        epochs = [entry.get('epoch', i) for i, entry in enumerate(log_data)]
        rewards = [entry.get('mean_reward', 0) for entry in log_data]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, rewards, label='Train Reward', marker='o')
        ax.set_title("Training Reward Convergence")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Reward")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "graphmarl_fig_11.png"))
        plt.savefig(os.path.join(out_dir, "graphmarl_fig_18.png"))
        plt.close()
    else:
        generate_placeholder_data("graphmarl_fig_11.png")
        generate_placeholder_data("graphmarl_fig_18.png")
except:
    generate_placeholder_data("graphmarl_fig_11.png")
    generate_placeholder_data("graphmarl_fig_18.png")

# Fig 12-17 placeholders
for i in range(12, 18):
    generate_placeholder_data(f"graphmarl_fig_{i}.png")

# Fig 19: Latency Comparison
fig, ax = plt.subplots(figsize=(8, 5))
benchmark.plot(x='Method', y='Avg Latency (ms)', kind='bar', color='orange', ax=ax)
plt.title('Latency Comparison (ms)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "graphmarl_fig_19.png"))
plt.close()

# Fig 20: Energy Consumption Comparison
fig, ax = plt.subplots(figsize=(8, 5))
benchmark.plot(x='Method', y='Total Energy (kJ)', kind='bar', color='red', ax=ax)
plt.title('Energy Consumption Comparison (kJ)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "graphmarl_fig_20.png"))
plt.close()

# Fig 21: Processing Time
generate_placeholder_data("graphmarl_fig_21.png", "Processing Time & Resource Utilization\n(Refer to Latency plot)")

# Fig 22: Delay and Energy efficiency
fig, ax = plt.subplots(figsize=(8, 5))
benchmark.plot(x='Method', y=['Avg Latency (ms)', 'Total Energy (kJ)'], kind='bar', ax=ax)
plt.title('Average Delay & Energy Consumption')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "graphmarl_fig_22.png"))
plt.close()

print("GraphMARL figures generated successfully.")
