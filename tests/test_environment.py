"""
Unit and integration tests for DHGC Hospital Graph, Self-Healing Network (SHN),
PettingZoo HospitalEdgeEnv, and EdgeNodeAgent with EDE.
"""

import unittest
import numpy as np

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.agents.edge_node import EdgeNodeAgent, AdaptiveNeighborCommunication
from src.data.workload_generator import WorkloadGenerator


class TestEnvironmentAndAgents(unittest.TestCase):

    def setUp(self):
        self.graph = HospitalGraph(seed=42)
        self.workload = WorkloadGenerator(seed=42)

    def test_hospital_graph_topology(self):
        self.assertEqual(len(self.graph.node_ids), 8)
        self.assertIn("ER", self.graph.node_ids)
        self.assertIn("ICU", self.graph.node_ids)

        er_nbrs = self.graph.get_neighbors("ER")
        self.assertIn("ICU", er_nbrs)
        self.assertIn("Surgery", er_nbrs)

        node_feat = self.graph.get_node_features("ER")
        self.assertEqual(len(node_feat), 6)

        edge_feat = self.graph.get_edge_features("ER", "ICU")
        self.assertEqual(len(edge_feat), 4)

    def test_self_healing_network_crash_and_recovery(self):
        self.assertNotIn("ICU", self.graph.crashed_nodes)
        self.graph.inject_node_crash("ICU")
        self.assertIn("ICU", self.graph.crashed_nodes)

        # ICU should no longer appear in active neighbors
        er_active_nbrs = self.graph.get_neighbors("ER", active_only=True)
        self.assertNotIn("ICU", er_active_nbrs)

        # Self-healing recovery
        self.graph.recover_node("ICU")
        self.assertNotIn("ICU", self.graph.crashed_nodes)
        er_recovered_nbrs = self.graph.get_neighbors("ER", active_only=True)
        self.assertIn("ICU", er_recovered_nbrs)

    def test_adaptive_neighbor_communication(self):
        anc = AdaptiveNeighborCommunication(node_id="ER", delta_queue_ratio=0.20)

        # Initial broadcast
        self.assertTrue(anc.should_broadcast(current_time_ms=0.0, current_queue_len=0, current_cpu_util=0.1))
        anc.emit_broadcast(0.0, 0.1, 0.1, 0, 0.1, 20.0)

        # Minor change (< 20%) -> suppressed
        self.assertFalse(anc.should_broadcast(current_time_ms=10.0, current_queue_len=0, current_cpu_util=0.12))

        # Major change (queue goes to 5) -> triggered
        self.assertTrue(anc.should_broadcast(current_time_ms=20.0, current_queue_len=5, current_cpu_util=0.12))

        # Emergency alert -> always triggered
        self.assertTrue(anc.should_broadcast(current_time_ms=30.0, current_queue_len=5, current_cpu_util=0.12, has_emergency_task=True))

    def test_pettingzoo_env_rollout(self):
        env = HospitalEdgeEnv(max_steps=10, seed=42)
        obs, _ = env.reset()

        self.assertEqual(len(obs), 8)
        for agent in env.agents:
            self.assertIn("local_history", obs[agent])
            self.assertIn("neighbor_nodes", obs[agent])
            self.assertIn("neighbor_edges", obs[agent])
            self.assertIn("task_features", obs[agent])
            self.assertIn("action_mask", obs[agent])

        # Step environment with dummy actions
        actions = {a: 0 for a in env.agents}
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        self.assertEqual(len(rewards), 8)
        self.assertEqual(len(infos), 8)


if __name__ == "__main__":
    unittest.main()
