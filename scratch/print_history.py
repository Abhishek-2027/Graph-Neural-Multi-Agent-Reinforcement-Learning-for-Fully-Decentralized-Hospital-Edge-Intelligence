import sys
sys.path.append(".")
from graph_re.plot_training_history import parse_log

data = parse_log(r"C:\Users\DELL\.gemini\antigravity-ide\brain\e045f814-32f0-4648-a47e-7cb9697abe98\.system_generated\tasks\task-993.log")
v = data["val_data"]

print(f"{'Epoch':<7} | {'Train Reward':<13} | {'Val Reward':<12} | {'Completed':<11} | {'Latency':<12} | {'Misses':<8}")
print("-" * 75)
for i in range(0, len(v["epoch"]), 2):
    ep = v["epoch"][i]
    tr = v["train_r"][i]
    vr = v["val_r"][i]
    comp = v["val_comp"][i]
    lat = v["val_lat"][i]
    ms = v["val_miss"][i]
    tag = " *BEST*" if ep == 975 else ""
    print(f"{ep:<7d} | {tr:<13.2f} | {vr:<12.2f} | {comp:<11.1f} | {lat:<9.1f} ms | {ms:<8.1f}{tag}")
# Print the best epoch
import numpy as np
best_idx = int(np.argmax(v["val_r"]))
print("-" * 75)
print(f"BEST (Epoch {v['epoch'][best_idx]}): Val Reward = {v['val_r'][best_idx]:.2f}, Completed = {v['val_comp'][best_idx]:.1f}, Latency = {v['val_lat'][best_idx]:.1f} ms, Misses = {v['val_miss'][best_idx]:.1f}")
