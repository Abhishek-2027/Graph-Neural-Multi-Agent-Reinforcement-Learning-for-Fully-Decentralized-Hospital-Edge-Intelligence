"""
PettingZoo Parallel Multi-Agent Environment for GraphMARL.

Wraps the Dynamic Hospital Graph and discrete-event task simulation into
a standard Multi-Agent Gym / PettingZoo Parallel interface.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator, MedicalTask
from src.utils.amrf_reward import AMRFReward
from src.utils.hsi_calculator import HSICalculator


class HospitalEdgeEnv:
    """
    PettingZoo Parallel-compatible Multi-Agent Environment simulating
    decentralized task offloading across dynamic hospital department edge nodes.
    """

    def __init__(
        self,
        hospital_graph: Optional[HospitalGraph] = None,
        workload_gen: Optional[WorkloadGenerator] = None,
        amrf_reward: Optional[AMRFReward] = None,
        max_neighbors: int = 5,
        history_len: int = 10,
        max_steps: int = 200,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.graph = hospital_graph or HospitalGraph(seed=seed)
        self.workload_gen = workload_gen or WorkloadGenerator(seed=seed)
        self.amrf = amrf_reward or AMRFReward()
        self.hsi_calc = HSICalculator()

        self.agents = list(self.graph.node_ids)
        self.possible_agents = list(self.agents)
        self.max_neighbors = max_neighbors
        self.history_len = history_len
        self.max_steps = max_steps

        # Action space: 0 = Local Exec, 1..max_neighbors = Offload to Neighbor k, (max_neighbors+1) = Queue
        self.action_dim = max_neighbors + 2

        # State tracking
        self.current_step = 0
        self.sim_time_ms = 0.0
        self.node_queues: Dict[str, List[MedicalTask]] = {n: [] for n in self.agents}
        self.node_history: Dict[str, List[np.ndarray]] = {
            n: [np.zeros(6, dtype=np.float32) for _ in range(history_len)] for n in self.agents
        }
        self.current_tasks: Dict[str, Optional[MedicalTask]] = {n: None for n in self.agents}
        self.neighbor_cache: Dict[str, List[str]] = {}

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.sim_time_ms = 0.0
        self.node_queues = {n: [] for n in self.agents}
        self.node_history = {
            n: [np.zeros(6, dtype=np.float32) for _ in range(self.history_len)] for n in self.agents
        }
        self.current_tasks = {n: None for n in self.agents}

        # Initialize neighbor caches
        for n in self.agents:
            self.neighbor_cache[n] = self.graph.get_neighbors(n, active_only=True)[: self.max_neighbors]

        # Generate initial task for each agent
        for n in self.agents:
            task = self.workload_gen.sample_task(node_id=n, arrival_time_ms=self.sim_time_ms)
            self.current_tasks[n] = task

        obs = self._get_all_observations()
        infos = {n: {} for n in self.agents}
        return obs, infos

    def _get_observation_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Constructs the multi-part observation for a single agent.
        """
        # 1. Local history matrix: (history_len, 6)
        local_hist = np.array(self.node_history[agent_id][-self.history_len :], dtype=np.float32)

        # 2. Neighbor node features: (max_neighbors, 6)
        nbrs = self.neighbor_cache.get(agent_id, [])
        nbr_nodes = np.zeros((self.max_neighbors, 6), dtype=np.float32)
        nbr_edges = np.zeros((self.max_neighbors, 4), dtype=np.float32)
        action_mask = np.zeros(self.action_dim, dtype=np.float32)

        # Action 0: Execute Local is always valid unless crashed
        if agent_id not in self.graph.crashed_nodes:
            action_mask[0] = 1.0

        for k, nbr in enumerate(nbrs):
            if k < self.max_neighbors:
                nbr_nodes[k] = self.graph.get_node_features(nbr)
                nbr_edges[k] = self.graph.get_edge_features(agent_id, nbr)
                if nbr not in self.graph.crashed_nodes:
                    action_mask[k + 1] = 1.0  # Offloading to valid neighbor allowed

        # Action (max_neighbors + 1): Hold in Queue is valid
        action_mask[-1] = 1.0

        # 3. Current task features (8D)
        task = self.current_tasks[agent_id]
        if task is not None:
            task_feats = np.array(
                [
                    float(min(task.data_size_mb / 50.0, 1.0)),
                    float(min(task.required_gflops / 25.0, 1.0)),
                    float(min(task.deadline_ms / 500.0, 1.0)),
                    float(task.hsi),
                    float(min(task.priority_score / 10.0, 1.0)),
                    float(min((self.sim_time_ms - task.arrival_time_ms) / 500.0, 1.0)),
                    1.0 if task.is_emergency else 0.0,
                    1.0 if task.hitl_override else 0.0,
                ],
                dtype=np.float32,
            )
        else:
            task_feats = np.zeros(8, dtype=np.float32)

        return {
            "local_history": local_hist,
            "neighbor_nodes": nbr_nodes,
            "neighbor_edges": nbr_edges,
            "task_features": task_feats,
            "action_mask": action_mask,
            "num_neighbors": len(nbrs),
        }

    def _get_all_observations(self) -> Dict[str, Dict[str, Any]]:
        return {agent: self._get_observation_for_agent(agent) for agent in self.agents}

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Executes one joint action step across all agents.
        """
        self.current_step += 1
        time_step_ms = 50.0
        self.sim_time_ms += time_step_ms

        rewards = {}
        infos = {}

        # Collect current queue lengths for FATS fairness
        all_queues = [len(self.node_queues[n]) for n in self.agents]

        for agent in self.agents:
            action = actions.get(agent, 0)
            task = self.current_tasks[agent]
            nbrs = self.neighbor_cache.get(agent, [])

            if task is None or agent in self.graph.crashed_nodes:
                rewards[agent] = -1.0 if agent in self.graph.crashed_nodes else 0.0
                infos[agent] = {"executed_node": agent, "latency_ms": 0.0, "deadline_missed": False}
                continue

            target_node = agent
            comm_delay_ms = 0.0
            comm_cost_kb = 0.0

            if action == 0:
                # Local Execution
                target_node = agent
            elif 1 <= action <= len(nbrs):
                # Offload to Neighbor
                target_node = nbrs[action - 1]
                comm_delay_ms = self.graph.calculate_transmission_delay_ms(agent, target_node, task.data_size_mb)
                comm_cost_kb = task.data_size_mb * 1024.0
            else:
                # Queue Task
                self.node_queues[agent].append(task)
                target_node = agent
                comm_delay_ms = 0.0

            # Compute execution delay
            target_data = self.graph.graph.nodes[target_node]
            compute_capacity = max(target_data.get("compute_gflops", 50.0), 10.0)
            exec_delay_ms = (task.required_gflops / compute_capacity) * 1000.0
            queue_delay_ms = len(self.node_queues[target_node]) * 10.0

            total_latency_ms = comm_delay_ms + queue_delay_ms + exec_delay_ms
            deadline_ms = task.deadline_ms
            deadline_missed = total_latency_ms > deadline_ms

            # Energy calculation (Joules)
            power_w = target_data.get("power_w", 20.0)
            energy_joules = (power_w * (exec_delay_ms / 1000.0)) + (0.5 * (comm_delay_ms / 1000.0))

            # Target node utilization update
            cpu_util = min(0.95, (len(self.node_queues[target_node]) + 1) * 0.15)

            # Compute AMRF Reward
            reward_dict = self.amrf.compute_reward(
                hsi=task.hsi,
                latency_ms=total_latency_ms,
                deadline_ms=deadline_ms,
                energy_joules=energy_joules,
                comm_cost_kb=comm_cost_kb,
                cpu_util=cpu_util,
                neighborhood_queues=all_queues,
                hitl_override=task.hitl_override,
                node_crashed=(target_node in self.graph.crashed_nodes),
            )

            rewards[agent] = reward_dict["total_reward"]
            infos[agent] = {
                "latency_ms": total_latency_ms,
                "deadline_ms": deadline_ms,
                "deadline_missed": deadline_missed,
                "energy_joules": energy_joules,
                "comm_cost_kb": comm_cost_kb,
                "target_node": target_node,
                "is_emergency": task.is_emergency,
                "reward_breakdown": reward_dict,
            }

            # Update target node state in graph
            self.graph.update_node_state(
                node_id=target_node,
                cpu_util=cpu_util,
                gpu_util=cpu_util * 0.7,
                queue_len=len(self.node_queues[target_node]),
                avg_hsi=task.hsi,
                current_time_ms=self.sim_time_ms,
            )

            # Update local history for agent
            node_feat = self.graph.get_node_features(agent)
            self.node_history[agent].append(node_feat)
            if len(self.node_history[agent]) > self.history_len:
                self.node_history[agent].pop(0)

            # Spawn next task for agent
            self.current_tasks[agent] = self.workload_gen.sample_task(
                node_id=agent, arrival_time_ms=self.sim_time_ms
            )

        terminated = {agent: self.current_step >= self.max_steps for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        obs = self._get_all_observations()

        return obs, rewards, terminated, truncated, infos
