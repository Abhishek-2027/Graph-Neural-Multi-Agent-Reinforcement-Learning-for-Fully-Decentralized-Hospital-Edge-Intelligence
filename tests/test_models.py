"""
Unit tests for Deep Perception and Reinforcement Learning Models in GraphMARL:
SCNN, GATv2, PCAS, and UA-MAPPO.
"""

import unittest
import torch
import numpy as np

from src.models.scnn import StackedCNN
from src.models.gatv2 import GATv2NeighborEncoder
from src.models.pcas import PCASCongestionForecaster
from src.models.ua_mappo import UAMAPPOAgent


class TestNeuralModels(unittest.TestCase):

    def test_scnn_forward(self):
        scnn = StackedCNN(in_channels=6, seq_len=10, embedding_dim=64)
        # Input shape: (Batch=4, Channels=6, Seq_Len=10)
        x = torch.randn(4, 6, 10)
        out = scnn(x)
        self.assertEqual(out.shape, (4, 64))

        # Test permuted input shape (Batch=4, Seq_Len=10, Channels=6)
        x_perm = torch.randn(4, 10, 6)
        out_perm = scnn(x_perm)
        self.assertEqual(out_perm.shape, (4, 64))

    def test_gatv2_forward_and_attention(self):
        gat = GATv2NeighborEncoder(node_in_dim=6, edge_in_dim=4, hidden_dim=64, out_dim=64, num_heads=4)
        batch_size = 2
        k_neighbors = 5

        self_node = torch.randn(batch_size, 6)
        nbr_nodes = torch.randn(batch_size, k_neighbors, 6)
        nbr_edges = torch.randn(batch_size, k_neighbors, 4)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])

        out_emb, attn_weights = gat(self_node, nbr_nodes, nbr_edges, neighbor_mask=mask)

        self.assertEqual(out_emb.shape, (batch_size, 64))
        self.assertEqual(attn_weights.shape, (batch_size, k_neighbors))

        # Check that masked-out neighbors receive near-zero attention
        self.assertAlmostEqual(attn_weights[0, 3].item(), 0.0, places=3)
        self.assertAlmostEqual(attn_weights[0, 4].item(), 0.0, places=3)

    def test_pcas_forecaster(self):
        pcas = PCASCongestionForecaster(history_window=10, forecast_horizon=2)
        q_hist = torch.tensor([[1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]])
        arr_rate = torch.tensor([[2.5]])

        pred_queues, cong_prob = pcas(q_hist, arr_rate)
        self.assertEqual(pred_queues.shape, (1, 2))
        self.assertEqual(cong_prob.shape, (1, 1))
        self.assertGreaterEqual(cong_prob.item(), 0.0)
        self.assertLessEqual(cong_prob.item(), 1.0)

    def test_ua_mappo_agent_step_and_uncertainty(self):
        agent = UAMAPPOAgent(max_neighbors=5, num_critics=5)

        obs_dict = {
            "local_history": np.random.randn(10, 6).astype(np.float32),
            "neighbor_nodes": np.random.randn(5, 6).astype(np.float32),
            "neighbor_edges": np.random.randn(5, 4).astype(np.float32),
            "task_features": np.random.randn(8).astype(np.float32),
            "action_mask": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

        step_out = agent.get_action_and_value(obs_dict)
        self.assertIn("action", step_out)
        self.assertIn("value", step_out)
        self.assertIn("variance", step_out)
        self.assertIn("attention_weights", step_out)

        # Ensure selected action respects action mask
        self.assertIn(step_out["action"], [0, 1, 2, 6])


if __name__ == "__main__":
    unittest.main()
