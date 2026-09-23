"""
Generates dedicated, publication-ready Markdown and CSV tables containing
all 14 evaluation metrics across all 8 workload sizes [50, 100, 150, 200, 250, 300, 350, 400]
for GraphMARL and all comparison baselines.
"""

import csv
import json
import os

RESULTS_DIR = "results"
JSON_PATH = os.path.join(RESULTS_DIR, "final_scientific_evaluation.json")

with open(JSON_PATH, "r") as f:
    data = json.load(f)

summary = data["summary"]
workloads = [10,50, 80,100, 120,150, 200, 250, 300, 350, 400,450, 420,500, 550 ,580, 600]

# 1. GraphMARL dedicated 14-metric table CSV
gm_csv_path = os.path.join(RESULTS_DIR, "graphmarl_workload_14_metrics.csv")
with open(gm_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Workload (Tasks)",
        "Avg Latency (ms)",
        "Execution Time (ms)",
        "Throughput (tasks/s)",
        "Completed Tasks",
        "Deadline Misses",
        "Deadline Miss Rate (%)",
        "Avg Queue Length",
        "Max Queue Length",
        "Energy Consumption (kJ)",
        "Resource Utilization (%)",
        "Reward",
        "P2P Offload Rate (%)",
        "Local Execution Rate (%)",
        "Communication Traffic (MB)"
    ])
    for w in workloads:
        m = summary[str(w)]["graphmarl"]
        writer.writerow([
            w,
            f"{m['avg_latency_mean']:.2f} ± {m['avg_latency_std']:.2f}",
            f"{m['avg_execution_time_mean']:.2f} ± {m['avg_execution_time_std']:.2f}",
            f"{m['throughput_mean']:.2f} ± {m['throughput_std']:.2f}",
            f"{m['completed_mean']:.1f} ± {m['completed_std']:.1f}",
            f"{m['deadline_misses_mean']:.1f} ± {m['deadline_misses_std']:.1f}",
            f"{m['deadline_miss_rate_pct_mean']:.1f}% ± {m['deadline_miss_rate_pct_std']:.1f}%",
            f"{m['avg_queue_len_mean']:.2f} ± {m['avg_queue_len_std']:.2f}",
            f"{m['max_queue_len_mean']:.1f} ± {m['max_queue_len_std']:.1f}",
            f"{m['energy_kj_mean']:.3f} ± {m['energy_kj_std']:.3f}",
            f"{m['resource_util_pct_mean']:.1f}% ± {m['resource_util_pct_std']:.1f}%",
            f"{m['reward_mean']:.2f} ± {m['reward_std']:.2f}",
            f"{m['p2p_offload_rate_pct_mean']:.1f}% ± {m['p2p_offload_rate_pct_std']:.1f}%",
            f"{m['local_exec_rate_pct_mean']:.1f}% ± {m['local_exec_rate_pct_std']:.1f}%",
            f"{m['comm_traffic_mb_mean']:.1f} ± {m['comm_traffic_mb_std']:.1f}"
        ])

print(f"[OK] Saved GraphMARL 14-metric table to: {gm_csv_path}")
