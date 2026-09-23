"""
PettingZoo Parallel Multi-Agent Environment for GraphMARL.

Wraps the Dynamic Hospital Graph and discrete-event task simulation into
a standard Multi-Agent Gym / PettingZoo Parallel interface.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import simpy

from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator, MedicalTask, TaskStatus
from src.utils.amrf_reward import AMRFReward
from src.utils.hsi_calculator import HSICalculator
from src.agents.edge_node import EdgeNodeAgent
from src.models.pcas import PCASCongestionForecaster


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
        task_arrival_prob: float = 0.2,
        max_tasks: Optional[int] = None,
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
        self.task_arrival_prob = float(task_arrival_prob)
        self.max_tasks = max_tasks
        
        # Instantiate PCAS Forecaster for observation generation
        self.pcas = PCASCongestionForecaster(history_window=self.history_len)

        # Action space: 0 = Local Exec, 1..max_neighbors = Offload to Neighbor k, (max_neighbors+1) = Queue
        self.action_dim = max_neighbors + 2

        # State tracking
        self.current_step = 0
        self.sim_time_ms = 0.0
        
        # SimPy Environment and decentralized EdgeNodes (Phases 1-4)
        self.sim_env = simpy.Environment()
        self.edge_nodes: Dict[str, EdgeNodeAgent] = {
            n: EdgeNodeAgent(node_id=n, sim_env=self.sim_env, node_capacity_gflops=self.graph.graph.nodes[n].get("compute_gflops", 50.0)) for n in self.agents
        }

        # Node history matrix holds 8D features (6 standard + 2 PCAS)
        self.node_history: Dict[str, List[np.ndarray]] = {
            n: [np.zeros(8, dtype=np.float32) for _ in range(history_len)] for n in self.agents
        }
        self.current_tasks: Dict[str, Optional[MedicalTask]] = {n: None for n in self.agents}
        self.neighbor_cache: Dict[str, List[str]] = {}
        self.recovery_metrics: Dict[str, int] = {"failures": 0, "orphaned": 0, "recovered": 0, "pending": 0}
        self.all_tasks: List[MedicalTask] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.sim_time_ms = 0.0
        self.all_tasks = []
        if hasattr(self.workload_gen, "tasks_generated"):
            self.workload_gen.tasks_generated = 0
        
        # Re-initialize SimPy and decentralized edge nodes
        self.sim_env = simpy.Environment()
        self.edge_nodes = {
            n: EdgeNodeAgent(node_id=n, sim_env=self.sim_env, node_capacity_gflops=self.graph.graph.nodes[n].get("compute_gflops", 50.0)) for n in self.agents
        }
        
        self.node_history = {
            n: [np.zeros(8, dtype=np.float32) for _ in range(self.history_len)] for n in self.agents
        }
        self.current_tasks = {n: None for n in self.agents}
        self.recovery_metrics = {"failures": 0, "orphaned": 0, "recovered": 0, "pending": 0}

        # Initialize neighbor caches
        for n in self.agents:
            self.neighbor_cache[n] = self.graph.get_neighbors(n, active_only=True)[: self.max_neighbors]

        # Generate initial task for each agent directly into their queues
        for n in self.agents:
            task = self.workload_gen.sample_task(node_id=n, arrival_time_ms=self.sim_time_ms)
            self.all_tasks.append(task)
            self.edge_nodes[n].push_task(task)

        # Initial Phase 7 Broadcasts to populate neighbor caches before first step
        for n in self.agents:
            node_feat_6d = self.graph.get_node_features(n)
            q_hist = [step_feats[3] for step_feats in self.node_history[n]]
            pred_q, cong_prob = self.pcas.predict_numpy(q_hist, arrival_rate=1.0)
            
            msg = self.edge_nodes[n].anc.emit_broadcast(
                current_time_ms=0.0,
                cpu_util=node_feat_6d[0],
                gpu_util=node_feat_6d[1],
                queue_length=len(self.edge_nodes[n].task_queue),
                avg_hsi=0.0,
                power_w=node_feat_6d[4] * 50.0, # Approximate de-normalization
                pred_queue=pred_q[0],
                cong_prob=cong_prob,
                has_emergency=False,
                is_available=True,
            )
            # Deliver instantly for initialization
            for nbr in self.neighbor_cache[n]:
                self.edge_nodes[nbr].receive_state_update(msg)

        obs = self._get_all_observations()
        infos = {n: {} for n in self.agents}
        return obs, infos

    def _get_observation_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Constructs the multi-part observation for a single agent.
        """
        # 1. Local history matrix: (history_len, 8)
        local_hist = np.array(self.node_history[agent_id][-self.history_len :], dtype=np.float32)

        # 2. Neighbor node features: (max_neighbors, 8)
        nbrs = self.neighbor_cache.get(agent_id, [])
        nbr_nodes = np.zeros((self.max_neighbors, 8), dtype=np.float32)
        nbr_edges = np.zeros((self.max_neighbors, 4), dtype=np.float32)
        action_mask = np.zeros(self.action_dim, dtype=np.float32)

        # Action 0: Execute Local is always valid unless crashed
        if agent_id not in self.graph.crashed_nodes:
            action_mask[0] = 1.0

        for k, nbr in enumerate(nbrs):
            if k < self.max_neighbors:
                # Get the latest state of the neighbor STRICTLY from local P2P cache
                msg = self.edge_nodes[agent_id].neighbor_state_cache.get(nbr)
                if msg is not None:
                    nbr_nodes[k] = np.array([
                        msg.cpu_util,
                        msg.gpu_util,
                        16.0 / 32.0, # RAM (hardcoded approx for now, could be in msg)
                        min(msg.queue_length / 50.0, 1.0),
                        min(msg.power_w / 50.0, 1.0),
                        msg.avg_hsi,
                        msg.pred_queue,
                        msg.cong_prob,
                    ], dtype=np.float32)
                    staleness_ms = max(self.sim_time_ms - msg.timestamp_ms, 0.0)
                else:
                    # No message ever received
                    staleness_ms = 9999.0

                nbr_edges[k] = self.graph.get_edge_features(agent_id, nbr)
                # Overwrite data staleness (index 2) dynamically based on P2P cache
                nbr_edges[k][2] = min(staleness_ms / 500.0, 1.0)
                
                if nbr not in self.graph.crashed_nodes:
                    action_mask[k + 1] = 1.0  # Offloading to valid neighbor allowed

        # Action (max_neighbors + 1): Hold in Queue is valid
        action_mask[-1] = 1.0

        # 3. Current task features (8D)
        # For Phase 5, the agent observes the first task in its task_queue (waiting for a decision).
        task = None
        if self.edge_nodes[agent_id].task_queue:
            task = self.edge_nodes[agent_id].task_queue[0]
            
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
            # If there's no task, only QUEUE (no-op) is a valid action
            action_mask = np.zeros(self.action_dim, dtype=np.float32)
            action_mask[-1] = 1.0

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

    def _transmit_task(self, task: MedicalTask, source: str, target: str, delay_ms: float):
        """SimPy process modeling realistic network transmission delay before target node receives the task."""
        yield self.sim_env.timeout(delay_ms)
        task.current_node = target
        self.edge_nodes[target].push_task(task)

    def _transmit_message(self, msg, source: str, target: str, delay_ms: float):
        """SimPy process for P2P ANC state exchange over network."""
        yield self.sim_env.timeout(delay_ms)
        self.edge_nodes[target].receive_state_update(msg)

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Executes one joint action step across all agents.
        Phase 5: Actions dictate what happens to the front task in each agent's task_queue.
        """
        self.current_step += 1
        time_step_ms = 50.0
        
        # 1. Handle SHN Failures & Recoveries (Phase 8)
        for agent in self.agents:
            # Detect crash
            if agent in self.graph.crashed_nodes and self.edge_nodes[agent].is_available:
                self.edge_nodes[agent].handle_crash()
                self.recovery_metrics["failures"] += 1
            
            # Detect recovery
            if agent not in self.graph.crashed_nodes and not self.edge_nodes[agent].is_available:
                self.edge_nodes[agent].recover_node()

            # Process orphaned tasks
            if not self.edge_nodes[agent].is_available and self.edge_nodes[agent].orphaned_tasks:
                remaining_orphans = []
                for task in self.edge_nodes[agent].orphaned_tasks:
                    self.recovery_metrics["orphaned"] += 1
                    target, explanation = self.edge_nodes[agent].attempt_recovery(task)
                    if target:
                        self.recovery_metrics["recovered"] += 1
                        edge_feats = self.graph.get_edge_features(agent, target)
                        bandwidth = max(float(edge_feats[0] * 1000), 1.0)
                        latency = float(edge_feats[1] * 100)
                        tx_delay = (task.data_size_mb / (bandwidth / 8.0)) * 1000.0 + latency
                        self.sim_env.process(self._transmit_task(task, agent, target, tx_delay))
                    else:
                        self.recovery_metrics["pending"] += 1
                        remaining_orphans.append(task)
                self.edge_nodes[agent].orphaned_tasks = remaining_orphans

        # 2. Apply MARL decisions for each agent
        for agent, action in actions.items():
            if not self.edge_nodes[agent].is_available:
                continue # Crashed nodes cannot execute normal MARL orchestrator actions

            if not self.edge_nodes[agent].task_queue:
                continue # No task to schedule
                
            # Pop the task we are making a decision on
            task = self.edge_nodes[agent].task_queue.pop(0)
            
            if action == 0:
                # LOCAL: Push to local execution queue to be scheduled
                self.edge_nodes[agent].local_execution_queue.append(task)
            elif action == self.action_dim - 1:
                # QUEUE: Re-queue it (put back at front)
                self.edge_nodes[agent].task_queue.insert(0, task)
            else:
                # OFFLOAD to neighbor
                nbrs = self.neighbor_cache.get(agent, [])
                idx = action - 1
                if idx < len(nbrs):
                    target_node = nbrs[idx]
                    
                    if target_node in self.graph.crashed_nodes:
                        # Invalid, drop or penalize. For safety, push back to local.
                        self.edge_nodes[agent].local_execution_queue.append(task)
                    else:
                        # Calculate network delay (size / bandwidth) + latency
                        edge_feats = self.graph.get_edge_features(agent, target_node)
                        bandwidth_mbps = float(edge_feats[0] * 1000) # De-normalize approx
                        latency_ms = float(edge_feats[1] * 100) # De-normalize approx
                        bandwidth_mbps = max(bandwidth_mbps, 1.0)
                        
                        tx_delay = (task.data_size_mb / (bandwidth_mbps / 8.0)) * 1000.0 + latency_ms
                        
                        # Start transmission process
                        self.sim_env.process(self._transmit_task(task, agent, target_node, tx_delay))
                else:
                    # Invalid action index (e.g., masking failed), fallback to LOCAL
                    self.edge_nodes[agent].local_execution_queue.append(task)

        # 2. Advance actual simulated time
        self.sim_time_ms += time_step_ms
        self.sim_env.run(until=self.sim_time_ms)

        rewards = {}
        infos = {}

        for agent in self.agents:
            # 3. Get Public State Summaries and Update Hospital Graph
            pub_state = self.edge_nodes[agent].get_public_state()
            self.graph.update_node_state(
                node_id=agent,
                cpu_util=pub_state["cpu_util"],
                gpu_util=pub_state["gpu_util"],
                queue_len=pub_state["queue_length"],
                avg_hsi=0.0, # Updated from summary if needed
                current_time_ms=self.sim_time_ms,
            )

            # 4. FATS (Fairness-Aware Task Scheduling) and Reward Calculation
            # Get neighborhood queue sizes (including self)
            nbrs = self.neighbor_cache.get(agent, [])
            valid_nbrs = [n for n in nbrs if n not in self.graph.crashed_nodes] + [agent]
            
            neighborhood_queues = [
                len(self.edge_nodes[n].task_queue) + len(self.edge_nodes[n].local_execution_queue)
                for n in valid_nbrs
            ]
            
            # Note: This is an approximation. A robust MARL environment associates rewards with specific tasks.
            # Here we apply the AMRFReward mechanism based on tasks completed this step.
            new_completions = [t for t in self.edge_nodes[agent].completed_tasks if t.completion_time_ms > (self.sim_time_ms - time_step_ms)]
            
            if len(new_completions) > 0:
                reward = 0.0
                for t in new_completions:
                    latency = t.completion_time_ms - t.arrival_time_ms
                    energy = t.total_compute_gflops * (self.graph.graph.nodes[agent].get("power_w", 20.0) / self.graph.graph.nodes[agent].get("compute_gflops", 50.0))
                    
                    # Compute AMRF reward for this task
                    task_reward_dict = self.amrf.compute_reward(
                        hsi=t.hsi,
                        latency_ms=latency,
                        deadline_ms=t.deadline_ms,
                        energy_joules=energy,
                        comm_cost_kb=t.data_size_mb * 1024.0 if t.origin_node != t.current_node else 0.0,
                        cpu_util=pub_state["cpu_util"],
                        q_variance=0.0, # Not calculating true variance in env
                        neighborhood_queues=neighborhood_queues,
                        hitl_override=t.hitl_override,
                        node_crashed=agent in self.graph.crashed_nodes,
                    )
                    reward += task_reward_dict["total_reward"]
                    # Strong bonus for completing tasks — primary learning signal
                    reward += 3.0
                
                # Average reward over completions
                reward /= len(new_completions)
            else:
                # Shaping reward: penalize total neighborhood backlog (not just local queue)
                total_backlog = float(sum(neighborhood_queues))
                fats_bonus = self.amrf.lambda_fats * self.amrf.calculate_fats(neighborhood_queues)
                reward = -0.01 * total_backlog + fats_bonus

                act = actions.get(agent, -1)
                if act == self.action_dim - 1:
                    # Discourage stalling via the wait/queue action
                    reward -= 0.1
            
            rewards[agent] = reward

            infos[agent] = {
                "completions": len(new_completions),
                "queue_len": pub_state["queue_length"],
                "fats_bonus": self.amrf.calculate_fats(neighborhood_queues)
            }

            # 5. PCAS Update local history for agent
            node_feat_6d = self.graph.get_node_features(agent)
            
            # Predict future congestion using the last 10 queue states from history
            q_hist = [step_feats[3] for step_feats in self.node_history[agent]]
            pred_q, cong_prob = self.pcas.predict_numpy(q_hist, arrival_rate=1.0)
            
            # Append PCAS predictions to make it 8D
            node_feat_8d = np.concatenate([node_feat_6d, np.array([pred_q[0], cong_prob], dtype=np.float32)])
            
            self.node_history[agent].append(node_feat_8d)
            if len(self.node_history[agent]) > self.history_len:
                self.node_history[agent].pop(0)

            # 6. Adaptive Neighbor Communication (ANC) Broadcast Check
            # Has emergency if any task currently waiting is an emergency
            has_emg = any([t.is_emergency for t in self.edge_nodes[agent].task_queue])
            if self.edge_nodes[agent].anc.should_broadcast(
                current_time_ms=self.sim_time_ms,
                current_queue_len=pub_state["queue_length"],
                current_cpu_util=pub_state["cpu_util"],
                has_emergency_task=has_emg,
                is_available=pub_state["available"],
            ):
                # Emit the broadcast message
                msg = self.edge_nodes[agent].anc.emit_broadcast(
                    current_time_ms=self.sim_time_ms,
                    cpu_util=pub_state["cpu_util"],
                    gpu_util=pub_state["gpu_util"],
                    queue_length=pub_state["queue_length"],
                    avg_hsi=0.0,
                    power_w=self.graph.graph.nodes[agent].get("power_w", 20.0),
                    pred_queue=pred_q[0],
                    cong_prob=cong_prob,
                    has_emergency=has_emg,
                    is_available=pub_state["available"],
                )
                
                # Transmit to all neighbors
                for nbr in self.neighbor_cache.get(agent, []):
                    if nbr not in self.graph.crashed_nodes:
                        # Message size is 128 bytes (0.000122 MB)
                        tx_delay = self.graph.calculate_transmission_delay_ms(agent, nbr, 128.0 / 1024.0 / 1024.0)
                        self.sim_env.process(self._transmit_message(msg, agent, nbr, tx_delay))

            # Spawn next task for agent (simulate arrival over this step based on task_arrival_prob)
            can_spawn = self.max_tasks is None or self.workload_gen.tasks_generated < self.max_tasks
            if can_spawn:
                if self.task_arrival_prob <= 1.0:
                    num_new = 1 if self.rng.rand() < self.task_arrival_prob else 0
                else:
                    num_new = int(self.task_arrival_prob) + (
                        1 if self.rng.rand() < (self.task_arrival_prob - int(self.task_arrival_prob)) else 0
                    )
                for _ in range(num_new):
                    if self.max_tasks is not None and self.workload_gen.tasks_generated >= self.max_tasks:
                        break
                    new_task = self.workload_gen.sample_task(node_id=agent, arrival_time_ms=self.sim_time_ms)
                    self.all_tasks.append(new_task)
                    self.edge_nodes[agent].push_task(new_task)

        terminated = {agent: self.current_step >= self.max_steps for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        obs = self._get_all_observations()

        return obs, rewards, terminated, truncated, infos
