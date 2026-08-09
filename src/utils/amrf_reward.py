"""
Adaptive Multi-Objective Reward Function (AMRF) & Fairness-Aware Task Scheduling (FATS)
for GraphMARL.

Dynamically shifts optimization weights between Emergency Mode (latency, deadline,
and uncertainty reduction prioritized) and Normal Mode (energy, throughput, and
neighborhood fairness balanced).
"""

from typing import Dict, List, Optional, Any
import numpy as np


class AMRFReward:
    """
    Computes reward signals for Multi-Agent Reinforcement Learning agents.
    """

    def __init__(
        self,
        alpha_util: float = 1.0,
        beta_latency: float = 2.0,
        gamma_energy: float = 1.0,
        delta_deadline: float = 5.0,
        eta_comm: float = 0.5,
        mu_uncertainty: float = 1.5,
        lambda_fats: float = 1.0,
        emergency_hsi_threshold: float = 0.80,
    ):
        self.alpha_util = alpha_util
        self.beta_latency = beta_latency
        self.gamma_energy = gamma_energy
        self.delta_deadline = delta_deadline
        self.eta_comm = eta_comm
        self.mu_uncertainty = mu_uncertainty
        self.lambda_fats = lambda_fats
        self.emergency_threshold = emergency_hsi_threshold

    def calculate_fats(self, neighborhood_queues: List[float]) -> float:
        """
        Calculates Fairness-Aware Task Scheduling score using the negative Gini coefficient.
        Returns a value in [-1.0, 0.0], where 0.0 is perfect equality.
        """
        arr = np.array(neighborhood_queues, dtype=np.float64)
        n = len(arr)
        if n <= 1 or np.sum(arr) <= 1e-6:
            return 0.0

        # Mean absolute difference Gini calculation
        diff_sum = np.sum(np.abs(arr[:, None] - arr[None, :]))
        gini = diff_sum / (2.0 * n * np.sum(arr) + 1e-8)
        # FATS reward bonus is higher (closer to 0) when gini is small
        return float(-np.clip(gini, 0.0, 1.0))

    def compute_reward(
        self,
        hsi: float,
        latency_ms: float,
        deadline_ms: float,
        energy_joules: float,
        comm_cost_kb: float,
        cpu_util: float,
        q_variance: float = 0.0,
        neighborhood_queues: Optional[List[float]] = None,
        hitl_override: bool = False,
        node_crashed: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluates the composite AMRF reward and its decomposed components.
        """
        if node_crashed:
            # Major penalty for attempting to run on / offload to a crashed node
            return {
                "total_reward": -20.0,
                "is_emergency": True,
                "latency_penalty": -5.0,
                "deadline_penalty": -10.0,
                "energy_penalty": 0.0,
                "comm_penalty": 0.0,
                "uncertainty_penalty": -5.0,
                "fats_bonus": 0.0,
            }

        is_emergency = (hsi >= self.emergency_threshold) or hitl_override
        deadline_missed = 1.0 if latency_ms > deadline_ms else 0.0
        normalized_lat = min(latency_ms / max(deadline_ms, 1.0), 3.0)
        norm_energy = min(energy_joules / 50.0, 2.0)
        norm_comm = min(comm_cost_kb / 100.0, 2.0)

        fats_score = 0.0
        if neighborhood_queues is not None and len(neighborhood_queues) > 1:
            fats_score = self.calculate_fats(neighborhood_queues)

        if is_emergency:
            # Emergency Mode: Latency and deadline misses heavily penalized;
            # Uncertainty variance penalizes routing to stale/noisy nodes.
            lat_pen = -(self.beta_latency * 3.0) * normalized_lat
            dl_pen = -(self.delta_deadline * 3.0) * (5.0 if deadline_missed else 0.0)
            energy_pen = -(self.gamma_energy * 0.2) * norm_energy
            comm_pen = -(self.eta_comm * 0.2) * norm_comm
            unc_pen = -self.mu_uncertainty * min(q_variance, 5.0)
            fats_bonus = 0.0  # Fairness takes back seat to patient survival

            total = lat_pen + dl_pen + energy_pen + comm_pen + unc_pen
        else:
            # Normal Mode: Balances compute efficiency, energy, latency, and fairness
            util_bonus = self.alpha_util * (1.0 - abs(cpu_util - 0.70))  # Target ~70% optimal utilization
            lat_pen = -self.beta_latency * normalized_lat
            dl_pen = -self.delta_deadline * (3.0 if deadline_missed else 0.0)
            energy_pen = -self.gamma_energy * norm_energy
            comm_pen = -self.eta_comm * norm_comm
            unc_pen = -(self.mu_uncertainty * 0.5) * min(q_variance, 5.0)
            fats_bonus = self.lambda_fats * fats_score

            total = util_bonus + lat_pen + dl_pen + energy_pen + comm_pen + unc_pen + fats_bonus

        return {
            "total_reward": float(np.clip(total, -30.0, 10.0)),
            "is_emergency": bool(is_emergency),
            "latency_penalty": float(lat_pen),
            "deadline_penalty": float(dl_pen),
            "energy_penalty": float(energy_pen),
            "comm_penalty": float(comm_pen),
            "uncertainty_penalty": float(unc_pen),
            "fats_bonus": float(fats_bonus),
        }
