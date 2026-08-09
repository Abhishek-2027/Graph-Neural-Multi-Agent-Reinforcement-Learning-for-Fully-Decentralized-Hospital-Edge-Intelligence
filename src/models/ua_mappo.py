"""
Uncertainty-Aware Multi-Agent Proximal Policy Optimization (UA-MAPPO) for GraphMARL.

Combines SCNN local perception, GATv2 graph attention embedding, and PCAS congestion
forecasting with an Actor-Critic architecture featuring an ensemble of value estimators
to penalize high-variance (stale/unreliable) offloading decisions.
"""

from typing import Dict, Tuple, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from src.models.scnn import StackedCNN
from src.models.gatv2 import GATv2NeighborEncoder
from src.models.pcas import PCASCongestionForecaster


class UAMAPPOActor(nn.Module):
    """
    Decentralized Actor Policy mapping fused state to action probabilities
    with dynamic action masking for unavailable/crashed neighbor nodes.
    """

    def __init__(
        self,
        fused_dim: int = 145,
        action_dim: int = 7,  # Local, Nbr_1..Nbr_5, Queue
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, fused_state: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Returns categorical action distribution and unmasked logits.
        """
        logits = self.net(fused_state)
        if action_mask is not None:
            # Mask out invalid/dead actions with large negative value
            masked_logits = logits.masked_fill(action_mask == 0, -1e9)
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)
        return dist, logits


class EnsembleCritic(nn.Module):
    """
    Critic network with K=5 independent value estimation heads
    to quantify predictive epistemic uncertainty under dynamic, stale environments.
    """

    def __init__(
        self,
        fused_dim: int = 145,
        hidden_dim: int = 128,
        num_critics: int = 5,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_critics)
            ]
        )

    def forward(self, fused_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_value: (Batch, 1) average predicted state-value
            variance: (Batch, 1) Q-value / State-value epistemic variance
            all_values: (Batch, num_critics) individual critic predictions
        """
        preds = torch.cat([head(fused_state) for head in self.heads], dim=-1)  # (Batch, K)
        mean_val = preds.mean(dim=-1, keepdim=True)
        var_val = preds.var(dim=-1, keepdim=True, unbiased=False)
        return mean_val, var_val, preds


class UAMAPPOAgent(nn.Module):
    """
    Complete GraphMARL Agent combining perception modules (SCNN + GATv2 + PCAS)
    and decision modules (Actor + Ensemble Critic).
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 4,
        seq_len: int = 10,
        scnn_embed_dim: int = 64,
        gat_embed_dim: int = 64,
        max_neighbors: int = 5,
        num_critics: int = 5,
        mu_uncertainty: float = 1.5,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.action_dim = max_neighbors + 2
        self.mu_uncertainty = mu_uncertainty
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

        # 1. Perception Encoders
        self.scnn = StackedCNN(in_channels=node_in_dim, seq_len=seq_len, embedding_dim=scnn_embed_dim)
        self.gatv2 = GATv2NeighborEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=gat_embed_dim,
            out_dim=gat_embed_dim,
            num_heads=4,
        )
        self.pcas = PCASCongestionForecaster(history_window=seq_len, forecast_horizon=2, hidden_dim=32)

        # Fused dimension: SCNN (64) + GATv2 (64) + Task (8) + PCAS (3) + Self (6) = 145
        self.fused_dim = scnn_embed_dim + gat_embed_dim + 8 + 3 + node_in_dim

        # 2. Decision Networks
        self.actor = UAMAPPOActor(fused_dim=self.fused_dim, action_dim=self.action_dim)
        self.critic = EnsembleCritic(fused_dim=self.fused_dim, num_critics=num_critics)

        # 3. Optimizers
        self.actor_opt = torch.optim.Adam(
            list(self.scnn.parameters()) + list(self.gatv2.parameters()) + list(self.actor.parameters()),
            lr=lr_actor,
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def encode_fused_state(
        self,
        local_history: torch.Tensor,
        self_node: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_edges: torch.Tensor,
        task_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses local perception, graph attention, and task context into unified state representation.
        Returns:
            fused_state: (Batch, fused_dim)
            attention_weights: (Batch, K)
        """
        # 1. SCNN Local Embedding
        h_local = self.scnn(local_history)  # (B, 64)

        # 2. GATv2 Graph Embedding
        # Construct neighbor mask from action_mask (skip action 0 and action -1)
        nbr_mask = None
        if action_mask is not None:
            nbr_mask = action_mask[:, 1 : self.max_neighbors + 1]

        h_graph, attn_weights = self.gatv2(self_node, neighbor_nodes, neighbor_edges, neighbor_mask=nbr_mask)

        # 3. PCAS Congestion Forecasting
        # Extract queue history from local_history channel 3 (queue length)
        # local_history is (B, 10, 6) or (B, 6, 10)
        if local_history.shape[1] == 10:
            queue_hist = local_history[:, :, 3]
        else:
            queue_hist = local_history[:, 3, :]

        arrival_rate = task_features[:, 4:5]  # Task priority / arrival proxy
        pred_queues, cong_prob = self.pcas(queue_hist, arrival_rate)
        h_pcas = torch.cat([pred_queues, cong_prob], dim=-1)  # (B, 3)

        # 4. State Fusion Concatenation
        fused = torch.cat([h_local, h_graph, task_features, h_pcas, self_node], dim=-1)
        return fused, attn_weights

    def get_action_and_value(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Inference forward pass for environment step.
        """
        # Convert numpy observations to tensors
        local_hist = torch.from_numpy(obs_dict["local_history"]).unsqueeze(0)
        self_node = torch.from_numpy(obs_dict["local_history"][-1]).unsqueeze(0)
        nbr_nodes = torch.from_numpy(obs_dict["neighbor_nodes"]).unsqueeze(0)
        nbr_edges = torch.from_numpy(obs_dict["neighbor_edges"]).unsqueeze(0)
        task_feats = torch.from_numpy(obs_dict["task_features"]).unsqueeze(0)
        act_mask = torch.from_numpy(obs_dict["action_mask"]).unsqueeze(0)

        with torch.no_grad():
            fused, attn_weights = self.encode_fused_state(
                local_hist, self_node, nbr_nodes, nbr_edges, task_feats, act_mask
            )
            dist, logits = self.actor(fused, act_mask)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            mean_val, var_val, _ = self.critic(fused)

        return {
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(mean_val.item()),
            "variance": float(var_val.item()),
            "attention_weights": attn_weights.squeeze(0).numpy(),
            "fused_state": fused.squeeze(0).numpy(),
        }

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Executes one PPO training update with uncertainty-aware objective penalty.
        """
        # Critic Update: Train all K ensemble heads on MSE against Monte Carlo returns
        _, _, all_critic_preds = self.critic(states)  # (Batch, K)
        critic_targets = returns.unsqueeze(-1).expand_as(all_critic_preds)
        critic_loss = F.mse_loss(all_critic_preds, critic_targets)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # Actor Update: PPO clipped surrogate loss with variance penalty
        dist, _ = self.actor(states, action_masks)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute Q/Value epistemic variance to penalize uncertainty
        with torch.no_grad():
            _, var_val, _ = self.critic(states)

        uncertainty_loss = self.mu_uncertainty * var_val.mean()
        total_actor_loss = policy_loss + uncertainty_loss - self.entropy_coef * entropy

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        return {
            "actor_loss": float(policy_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "uncertainty_penalty": float(uncertainty_loss.item()),
            "entropy": float(entropy.item()),
        }
