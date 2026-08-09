"""
Graph Attention Network v2 (GATv2) with Edge Features for GraphMARL.

Computes dynamic, context-dependent attention weights over immediate hospital
department neighbors, incorporating edge link features (bandwidth, latency,
data staleness, link reliability) into neighborhood aggregation.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2EdgeAttentionLayer(nn.Module):
    """
    Dynamic GATv2 attention layer supporting multi-dimensional edge features.
    Computes alpha_ij = softmax_j( a^T LeakyReLU( W_src h_i + W_dst h_j + W_e e_ij ) )
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 4,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.negative_slope = negative_slope

        self.w_src = nn.Linear(node_in_dim, out_dim, bias=False)
        self.w_dst = nn.Linear(node_in_dim, out_dim, bias=False)
        self.w_edge = nn.Linear(edge_in_dim, out_dim, bias=False)
        self.w_val = nn.Linear(node_in_dim, out_dim, bias=False)

        self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(
        self,
        self_node: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_edges: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_node: (Batch, node_in_dim)
            neighbor_nodes: (Batch, K_neighbors, node_in_dim)
            neighbor_edges: (Batch, K_neighbors, edge_in_dim)
            neighbor_mask: (Batch, K_neighbors) boolean/binary mask (1 = valid neighbor, 0 = padded/dead)
        Returns:
            fused_neighborhood_embedding: (Batch, out_dim)
            attention_weights: (Batch, K_neighbors) average attention across heads
        """
        if self_node.dim() == 1:
            self_node = self_node.unsqueeze(0)
        if neighbor_nodes.dim() == 2:
            neighbor_nodes = neighbor_nodes.unsqueeze(0)
        if neighbor_edges.dim() == 2:
            neighbor_edges = neighbor_edges.unsqueeze(0)

        batch_size, k_nbrs, _ = neighbor_nodes.shape

        # Linear projections
        # self_proj: (B, 1, H, d_h)
        self_proj = self.w_src(self_node).view(batch_size, 1, self.num_heads, self.head_dim)
        # nbr_proj: (B, K, H, d_h)
        nbr_proj = self.w_dst(neighbor_nodes).view(batch_size, k_nbrs, self.num_heads, self.head_dim)
        # edge_proj: (B, K, H, d_h)
        edge_proj = self.w_edge(neighbor_edges).view(batch_size, k_nbrs, self.num_heads, self.head_dim)
        # val_proj: (B, K, H, d_h)
        val_proj = self.w_val(neighbor_nodes).view(batch_size, k_nbrs, self.num_heads, self.head_dim)

        # Dynamic GATv2 attention: LeakyReLU( W_src h_i + W_dst h_j + W_e e_ij )
        # Sum has shape (B, K, H, d_h)
        combined = self.leaky_relu(self_proj + nbr_proj + edge_proj)

        # Attention logits: dot product with attn_vec -> (B, K, H)
        logits = torch.einsum("bkhd,hd->bkh", combined, self.attn_vec)

        # Apply neighbor mask
        if neighbor_mask is not None:
            if neighbor_mask.dim() == 1:
                neighbor_mask = neighbor_mask.unsqueeze(0)
            mask = neighbor_mask.unsqueeze(-1).expand_as(logits)  # (B, K, H)
            logits = logits.masked_fill(mask == 0, -1e9)

        # Softmax over neighbors (dim=1)
        attn_scores = F.softmax(logits, dim=1)  # (B, K, H)
        attn_scores = self.dropout(attn_scores)

        # Handle all-zero mask case (NaN protection)
        attn_scores = torch.nan_to_num(attn_scores, nan=0.0)

        # Weighted aggregation: (B, K, H) x (B, K, H, d_h) -> (B, H, d_h)
        aggregated = torch.einsum("bkh,bkhd->bhd", attn_scores, val_proj)
        # Concatenate heads -> (B, out_dim)
        out_embedding = aggregated.reshape(batch_size, self.out_dim)

        # Average attention weights across heads for explainability -> (B, K)
        mean_attn_weights = attn_scores.mean(dim=-1)

        return out_embedding, mean_attn_weights


class GATv2NeighborEncoder(nn.Module):
    """
    2-Layer GATv2 Network for Decentralized Hospital Neighborhood Embedding.
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer1 = GATv2EdgeAttentionLayer(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.layer2 = GATv2EdgeAttentionLayer(
            node_in_dim=hidden_dim,
            edge_in_dim=edge_in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        self_node: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_edges: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing neighborhood embedding and interpretable attention scores.
        """
        h1, attn1 = self.layer1(self_node, neighbor_nodes, neighbor_edges, neighbor_mask)
        # Project neighbor nodes to hidden dim for layer 2
        nbr_hidden = F.elu(self.node_proj(neighbor_nodes))
        h2, attn2 = self.layer2(h1, nbr_hidden, neighbor_edges, neighbor_mask)
        out = self.norm(F.elu(h2))
        return out, attn2
