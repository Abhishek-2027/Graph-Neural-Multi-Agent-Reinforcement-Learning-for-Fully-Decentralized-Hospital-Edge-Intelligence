"""
Uncertainty-Aware Multi-Agent Proximal Policy Optimization (UA-MAPPO) for GraphMARL.

Combines:
    - SCNN local perception
    - GATv2 graph attention embedding
    - PCAS congestion forecasting
    - Actor-Critic architecture
    - Ensemble value estimators for epistemic uncertainty

The ensemble critic estimates uncertainty from the variance between
multiple independent value heads.

Author: Your Name
"""

from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.models.scnn import StackedCNN
from src.models.gatv2 import GATv2NeighborEncoder
from src.models.pcas import PCASCongestionForecaster


class UAMAPPOActor(nn.Module):
    """
    Decentralized Actor Policy.

    Maps the fused state representation to action probabilities.

    Actions:
        0               -> Local execution
        1..max_neighbors -> Neighbor offloading
        max_neighbors+1 -> Queue/wait action

    Dynamic action masking is supported for unavailable/crashed neighbors.
    """

    def __init__(
        self,
        fused_dim: int = 145,
        action_dim: int = 7,
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
        self,
        fused_state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Args:
            fused_state:
                Tensor of shape (Batch, fused_dim)

            action_mask:
                Tensor of shape (Batch, action_dim)
                1 = valid action
                0 = invalid action

        Returns:
            dist:
                Categorical action distribution

            logits:
                Original unmasked logits
        """

        logits = self.net(fused_state)

        if action_mask is not None:
            # Ensure mask has same dtype/device semantics.
            action_mask = action_mask.to(
                device=logits.device,
                dtype=torch.bool,
            )

            # Invalid actions receive a very negative logit.
            masked_logits = logits.masked_fill(
                ~action_mask,
                -1e9,
            )
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)

        return dist, logits


class EnsembleCritic(nn.Module):
    """
    Ensemble Critic with multiple independent value heads.

    The disagreement between heads is used as an epistemic uncertainty
    estimate.

    For K critics:

        V_1(s)
        V_2(s)
        ...
        V_K(s)

    Mean:
        mean(V_i)

    Variance:
        Var(V_i)
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

    def forward(
        self,
        fused_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_state:
                (Batch, fused_dim)

        Returns:
            mean_value:
                (Batch, 1)

            variance:
                (Batch, 1)

            all_values:
                (Batch, num_critics)
        """

        predictions = torch.cat(
            [
                head(fused_state)
                for head in self.heads
            ],
            dim=-1,
        )

        # predictions shape:
        # (Batch, num_critics)

        mean_value = predictions.mean(
            dim=-1,
            keepdim=True,
        )

        variance = predictions.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        return mean_value, variance, predictions


class UAMAPPOAgent(nn.Module):
    """
    Complete UA-MAPPO GraphMARL Agent.

    Architecture:

        Local History
              |
             SCNN
              |
              +------------------+
                                 |
        Neighbor Graph --> GATv2 |
                                 |
        Queue History --> PCAS   |
                                 |
        Task Features -----------+
                                 |
        Self Node ----------------+
                                 |
                            Fused State
                                 |
                    +------------+------------+
                    |                         |
                  Actor                    Critic
                    |                         |
                  Action             Ensemble Values
                                              |
                                          Uncertainty
    """

    def __init__(
        self,
        node_in_dim: int = 8,
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
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Device configuration
        # ---------------------------------------------------------

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        # Enable cuDNN benchmark for fixed-size inputs.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # ---------------------------------------------------------
        # Basic configuration
        # ---------------------------------------------------------

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.seq_len = seq_len

        self.scnn_embed_dim = scnn_embed_dim
        self.gat_embed_dim = gat_embed_dim

        self.max_neighbors = max_neighbors

        # Actions:
        #   0 = local
        #   1..max_neighbors = neighbors
        #   max_neighbors+1 = queue
        self.action_dim = max_neighbors + 2

        self.num_critics = num_critics

        self.mu_uncertainty = mu_uncertainty
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # ---------------------------------------------------------
        # 1. Perception Encoders
        # ---------------------------------------------------------

        self.scnn = StackedCNN(
            in_channels=node_in_dim,
            seq_len=seq_len,
            embedding_dim=scnn_embed_dim,
        )

        self.gatv2 = GATv2NeighborEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=gat_embed_dim,
            out_dim=gat_embed_dim,
            num_heads=4,
        )

        self.pcas = PCASCongestionForecaster(
            history_window=seq_len,
            forecast_horizon=2,
            hidden_dim=32,
        )

        # ---------------------------------------------------------
        # 2. Fused state dimension
        # ---------------------------------------------------------
        #
        # SCNN       = scnn_embed_dim
        # GATv2      = gat_embed_dim
        # Task       = 8
        # PCAS       = 3
        # Self node  = node_in_dim
        #
        # Default:
        #
        # 64 + 64 + 8 + 3 + 8 = 147
        #
        # IMPORTANT:
        # The original code/comment said 145 while node_in_dim
        # was 8. The actual dimension is 147.
        # ---------------------------------------------------------

        self.task_feature_dim = 8
        self.pcas_feature_dim = 3

        self.fused_dim = (
            scnn_embed_dim
            + gat_embed_dim
            + self.task_feature_dim
            + self.pcas_feature_dim
            + node_in_dim
        )

        # ---------------------------------------------------------
        # 3. Actor
        # ---------------------------------------------------------

        self.actor = UAMAPPOActor(
            fused_dim=self.fused_dim,
            action_dim=self.action_dim,
            hidden_dim=128,
        )

        # ---------------------------------------------------------
        # 4. Ensemble Critic
        # ---------------------------------------------------------

        self.critic = EnsembleCritic(
            fused_dim=self.fused_dim,
            hidden_dim=128,
            num_critics=num_critics,
        )

        # ---------------------------------------------------------
        # Move model to device
        # ---------------------------------------------------------

        self.to(self.device)

        # ---------------------------------------------------------
        # Optimizers
        # ---------------------------------------------------------

        # Actor optimizer includes:
        #   SCNN
        #   GATv2
        #   PCAS
        #   Actor
        #
        # If you do NOT want PCAS to be trained jointly, remove
        # self.pcas.parameters() from this optimizer.

        self.actor_opt = torch.optim.Adam(
            list(self.scnn.parameters())
            + list(self.gatv2.parameters())
            + list(self.pcas.parameters())
            + list(self.actor.parameters()),
            lr=lr_actor,
        )

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic,
        )

    # =============================================================
    # STATE ENCODING
    # =============================================================

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
        Fuse SCNN, GATv2, PCAS, task and self-node features.

        Expected shapes:

            local_history:
                (B, seq_len, node_features)
                OR
                (B, node_features, seq_len)

            self_node:
                (B, node_features)

            neighbor_nodes:
                depends on GATv2 implementation

            neighbor_edges:
                depends on GATv2 implementation

            task_features:
                (B, 8)

            action_mask:
                (B, action_dim)

        Returns:

            fused_state:
                (B, fused_dim)

            attention_weights:
                GATv2 attention weights
        """

        # ---------------------------------------------------------
        # 1. SCNN Local Embedding
        # ---------------------------------------------------------

        h_local = self.scnn(local_history)

        # ---------------------------------------------------------
        # 2. GATv2 Graph Embedding
        # ---------------------------------------------------------

        neighbor_mask = None

        if action_mask is not None:
            neighbor_mask = action_mask[
                :,
                1 : self.max_neighbors + 1,
            ]

        h_graph, attention_weights = self.gatv2(
            self_node,
            neighbor_nodes,
            neighbor_edges,
            neighbor_mask=neighbor_mask,
        )

        # ---------------------------------------------------------
        # 3. PCAS Congestion Forecast
        # ---------------------------------------------------------

        # local_history can be:
        #
        # (B, seq_len, features)
        #
        # or:
        #
        # (B, features, seq_len)

        if local_history.dim() != 3:
            raise ValueError(
                "local_history must have 3 dimensions: "
                "(B, seq_len, features) or "
                "(B, features, seq_len). "
                f"Got shape: {tuple(local_history.shape)}"
            )

        if local_history.shape[1] == self.seq_len:
            # (B, seq_len, features)
            #
            # Channel/index 3 = queue length
            queue_history = local_history[:, :, 3]

        elif local_history.shape[2] == self.seq_len:
            # (B, features, seq_len)
            queue_history = local_history[:, 3, :]

        else:
            raise ValueError(
                "Unable to determine sequence dimension in "
                f"local_history with shape {tuple(local_history.shape)}. "
                f"Expected seq_len={self.seq_len}."
            )

        # Task feature index 4 is used as arrival-rate proxy.
        if task_features.shape[-1] < 5:
            raise ValueError(
                "task_features must contain at least 5 features. "
                f"Got shape: {tuple(task_features.shape)}"
            )

        arrival_rate = task_features[:, 4:5]

        pred_queues, congestion_probability = self.pcas(
            queue_history,
            arrival_rate,
        )

        h_pcas = torch.cat(
            [
                pred_queues,
                congestion_probability,
            ],
            dim=-1,
        )

        # ---------------------------------------------------------
        # 4. State Fusion
        # ---------------------------------------------------------

        fused_state = torch.cat(
            [
                h_local,
                h_graph,
                task_features,
                h_pcas,
                self_node,
            ],
            dim=-1,
        )

        # Safety check.
        if fused_state.shape[-1] != self.fused_dim:
            raise RuntimeError(
                f"Fused state dimension mismatch. "
                f"Expected {self.fused_dim}, "
                f"got {fused_state.shape[-1]}.\n"
                f"SCNN={h_local.shape[-1]}, "
                f"GAT={h_graph.shape[-1]}, "
                f"Task={task_features.shape[-1]}, "
                f"PCAS={h_pcas.shape[-1]}, "
                f"Self={self_node.shape[-1]}"
            )

        return fused_state, attention_weights

    # =============================================================
    # INFERENCE
    # =============================================================

    def get_action_and_value(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform one environment inference step.

        Args:
            obs_dict:
                Dictionary containing:
                    local_history
                    neighbor_nodes
                    neighbor_edges
                    task_features
                    action_mask

            deterministic:
                If True, choose highest-probability action.
                If False, sample from policy.

        Returns:
            Dictionary containing:
                action
                log_prob
                value
                variance
                attention_weights
                fused_state
        """

        # ---------------------------------------------------------
        # Convert observations to tensors
        # ---------------------------------------------------------

        local_history = torch.as_tensor(
            obs_dict["local_history"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        neighbor_nodes = torch.as_tensor(
            obs_dict["neighbor_nodes"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        neighbor_edges = torch.as_tensor(
            obs_dict["neighbor_edges"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        task_features = torch.as_tensor(
            obs_dict["task_features"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action_mask = torch.as_tensor(
            obs_dict["action_mask"],
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        # ---------------------------------------------------------
        # Self node
        # ---------------------------------------------------------
        #
        # Assuming local_history is:
        #
        # (seq_len, node_features)
        #
        # and the last timestep contains current self-node state.
        #
        # If your environment stores self_node separately,
        # replace this with obs_dict["self_node"].
        # ---------------------------------------------------------

        if local_history.shape[1] == self.seq_len:
            self_node = local_history[:, -1, :]

        elif local_history.shape[2] == self.seq_len:
            self_node = local_history[:, :, -1]

        else:
            raise ValueError(
                "Invalid local_history shape: "
                f"{tuple(local_history.shape)}"
            )

        # ---------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------

        self.eval()

        with torch.no_grad():

            fused_state, attention_weights = self.encode_fused_state(
                local_history=local_history,
                self_node=self_node,
                neighbor_nodes=neighbor_nodes,
                neighbor_edges=neighbor_edges,
                task_features=task_features,
                action_mask=action_mask,
            )

            dist, logits = self.actor(
                fused_state,
                action_mask,
            )

            if deterministic:
                action = torch.argmax(
                    dist.probs,
                    dim=-1,
                )
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            mean_value, variance, _ = self.critic(
                fused_state
            )

        return {
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(mean_value.item()),
            "variance": float(variance.item()),
            "attention_weights": (
                attention_weights
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            ),
            "fused_state": (
                fused_state
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            ),
        }

    # =============================================================
    # PPO UPDATE
    # =============================================================

    def update_policy(
        self,
        states: Optional[torch.Tensor] = None,
        actions: torch.Tensor = None,
        old_log_probs: torch.Tensor = None,
        returns: torch.Tensor = None,
        advantages: torch.Tensor = None,
        action_masks: Optional[torch.Tensor] = None,
        epoch: int = 1,
        total_epochs: Optional[int] = None,
        verbose: bool = True,
        obs_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Perform one PPO update.

        Args:
            states:
                Fused states, shape (B, fused_dim) (optional if obs_tensors is provided)

            actions:
                Action indices, shape (B,)

            old_log_probs:
                Log probabilities from old policy, shape (B,)

            returns:
                Target returns, shape (B,)

            advantages:
                Advantage estimates, shape (B,)

            action_masks:
                Optional action masks, shape (B, action_dim)

            epoch:
                Current epoch number.

            total_epochs:
                Optional total number of epochs.

            verbose:
                Logging flag.

            obs_tensors:
                Optional dict of raw observation tensors (local_history, self_node,
                neighbor_nodes, neighbor_edges, task_features) to allow end-to-end
                gradient propagation into GATv2, SCNN, and PCAS.

        Returns:
            Training metrics dictionary.
        """

        # ---------------------------------------------------------
        # Training mode
        # ---------------------------------------------------------

        self.train()

        # ---------------------------------------------------------
        # Encode state or move tensors to device
        # ---------------------------------------------------------

        if action_masks is not None:
            action_masks = action_masks.to(
                self.device,
                dtype=torch.bool,
            )

        if obs_tensors is not None:
            states, _ = self.encode_fused_state(
                obs_tensors["local_history"].to(self.device, dtype=torch.float32),
                obs_tensors["self_node"].to(self.device, dtype=torch.float32),
                obs_tensors["neighbor_nodes"].to(self.device, dtype=torch.float32),
                obs_tensors["neighbor_edges"].to(self.device, dtype=torch.float32),
                obs_tensors["task_features"].to(self.device, dtype=torch.float32),
                action_masks,
            )
            critic_states = states.detach()
        else:
            states = states.to(
                self.device,
                dtype=torch.float32,
            )
            critic_states = states

        actions = actions.to(
            self.device,
            dtype=torch.long,
        )

        old_log_probs = old_log_probs.to(
            self.device,
            dtype=torch.float32,
        )

        returns = returns.to(
            self.device,
            dtype=torch.float32,
        )

        advantages = advantages.to(
            self.device,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # Make sure tensors are 1-D where appropriate
        # ---------------------------------------------------------

        old_log_probs = old_log_probs.view(-1)
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        actions = actions.view(-1)

        # ---------------------------------------------------------
        # Advantage normalization
        # ---------------------------------------------------------

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        # =========================================================
        # CRITIC UPDATE
        # =========================================================

        mean_value, variance, all_critic_preds = self.critic(
            critic_states
        )

        # returns:
        # (B,)
        #
        # critic predictions:
        # (B,K)

        critic_targets = returns.unsqueeze(-1).expand_as(
            all_critic_preds
        )

        critic_loss = F.mse_loss(
            all_critic_preds,
            critic_targets,
        )

        self.critic_opt.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=0.5,
        )

        self.critic_opt.step()

        # =========================================================
        # ACTOR UPDATE
        # =========================================================

        dist, _ = self.actor(
            states,
            action_masks,
        )

        new_log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy().mean()

        # ---------------------------------------------------------
        # PPO ratio
        # ---------------------------------------------------------

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        # ---------------------------------------------------------
        # PPO clipped objective
        # ---------------------------------------------------------

        surr1 = (
            ratio * advantages
        )

        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            )
            * advantages
        )

        policy_loss = -torch.min(
            surr1,
            surr2,
        ).mean()

        # =========================================================
        # UNCERTAINTY PENALTY
        # =========================================================
        #
        # IMPORTANT:
        #
        # This uses the critic variance as a scalar penalty.
        # Since it is calculated without gradient, it does not
        # directly propagate uncertainty gradients into the actor.
        #
        # This preserves the behavior of your original code.
        #
        # For a truly action-dependent uncertainty-aware PPO,
        # the critic should estimate V(s,a) or Q(s,a).
        # =========================================================

        with torch.no_grad():
            _, variance_detached, _ = self.critic(
                states
            )

        uncertainty_loss = (
            self.mu_uncertainty
            * variance_detached.mean()
        )

        # ---------------------------------------------------------
        # Total actor loss
        # ---------------------------------------------------------

        total_actor_loss = (
            policy_loss
            + uncertainty_loss
            - self.entropy_coef * entropy
        )

        self.actor_opt.zero_grad(
            set_to_none=True
        )

        total_actor_loss.backward()

        # Clip all actor-side gradients, including
        # SCNN, GATv2, PCAS and Actor.
        actor_parameters = (
            list(self.scnn.parameters())
            + list(self.gatv2.parameters())
            + list(self.pcas.parameters())
            + list(self.actor.parameters())
        )

        nn.utils.clip_grad_norm_(
            actor_parameters,
            max_norm=0.5,
        )

        self.actor_opt.step()

        # =========================================================
        # Metrics
        # =========================================================

        actor_loss_value = float(
            policy_loss.item()
        )

        critic_loss_value = float(
            critic_loss.item()
        )

        uncertainty_value = float(
            uncertainty_loss.item()
        )

        entropy_value = float(
            entropy.item()
        )

        mean_value_value = float(
            mean_value.mean().item()
        )

        variance_value = float(
            variance.mean().item()
        )

        # ---------------------------------------------------------
        # DYNAMIC EPOCH PRINTING
        # ---------------------------------------------------------

        if verbose:
            if total_epochs is not None:
                print(
                    f"Epoch [{epoch}/{total_epochs}] | "
                    f"Actor Loss: {actor_loss_value:.4f} | "
                    f"Critic Loss: {critic_loss_value:.4f} | "
                    f"Uncertainty: {uncertainty_value:.4f} | "
                    f"Entropy: {entropy_value:.4f} | "
                    f"Value: {mean_value_value:.4f} | "
                    f"Variance: {variance_value:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch}] | "
                    f"Actor Loss: {actor_loss_value:.4f} | "
                    f"Critic Loss: {critic_loss_value:.4f} | "
                    f"Uncertainty: {uncertainty_value:.4f} | "
                    f"Entropy: {entropy_value:.4f} | "
                    f"Value: {mean_value_value:.4f} | "
                    f"Variance: {variance_value:.4f}"
                )

        return {
            "epoch": float(epoch),
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "uncertainty_penalty": uncertainty_value,
            "entropy": entropy_value,
            "mean_value": mean_value_value,
            "value_variance": variance_value,
        }


# =============================================================
# OPTIONAL TRAINING LOOP EXAMPLE
# =============================================================

def train_agent(
    agent: UAMAPPOAgent,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    action_masks: Optional[torch.Tensor] = None,
    num_epochs: int = 10,
) -> list:
    """
    Example PPO training loop.

    num_epochs is a VARIABLE.

    Example:

        train_agent(
            agent,
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            action_masks,
            num_epochs=20,
        )

    Output:

        Epoch [1/20] ...
        Epoch [2/20] ...
        ...
        Epoch [20/20] ...
    """

    history = []

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        metrics = agent.update_policy(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            action_masks=action_masks,
            epoch=epoch,
            total_epochs=num_epochs,
        )

        history.append(metrics)

    return history


# =============================================================
# TEST / DEBUG
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UA-MAPPO Agent")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    agent = UAMAPPOAgent(
        node_in_dim=8,
        edge_in_dim=4,
        seq_len=10,
        scnn_embed_dim=64,
        gat_embed_dim=64,
        max_neighbors=5,
        num_critics=5,
        mu_uncertainty=1.5,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01,
        device=device,
    )

    print(f"Fused dimension: {agent.fused_dim}")
    print(f"Action dimension: {agent.action_dim}")
    print(f"Number of critics: {agent.num_critics}")

    print("\nModel successfully initialized.")
