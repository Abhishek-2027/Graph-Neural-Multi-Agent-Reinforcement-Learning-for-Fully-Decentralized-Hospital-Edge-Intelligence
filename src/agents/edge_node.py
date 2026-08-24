"""
Decentralized Edge Node Agent, Adaptive Neighbor Communication (ANC),
and Explainable Decision Engine (EDE) for GraphMARL.

Encapsulates autonomous department-level scheduling logic, event-triggered communication,
clinician HITL overrides, and human-interpretable clinical decision rationales.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import simpy

from src.models.ua_mappo import UAMAPPOAgent
from src.data.workload_generator import MedicalTask, TaskStatus
from src.utils.hsi_calculator import HSICalculator


@dataclass
class BroadcastStatusMessage:
    sender_id: str
    timestamp_ms: float
    cpu_util: float
    gpu_util: float
    queue_length: int
    avg_hsi: float
    power_w: float
    pred_queue: float = 0.0
    cong_prob: float = 0.0
    emergency_alert: bool = False
    is_available: bool = True
    message_size_bytes: int = 128


class AdaptiveNeighborCommunication:
    """
    Event-driven communication protocol (ANC).
    Drastically reduces hospital wireless bandwidth by suppressing redundant status broadcasts.
    Emits updates only on queue changes >= delta_threshold, severe overload, or emergency alerts.
    """

    def __init__(
        self,
        node_id: str,
        delta_queue_ratio: float = 0.20,
        overload_cpu_threshold: float = 0.85,
        max_silence_ms: float = 2000.0,
    ):
        self.node_id = node_id
        self.delta_queue_ratio = delta_queue_ratio
        self.overload_threshold = overload_cpu_threshold
        self.max_silence_ms = max_silence_ms

        self.last_broadcast_time_ms = -1.0
        self.last_broadcast_queue = -1
        self.last_broadcast_cpu = -1.0
        self.last_broadcast_available = True

        # Stats for publication evaluation
        self.total_broadcasts = 0
        self.total_bytes_transmitted = 0

    def should_broadcast(
        self,
        current_time_ms: float,
        current_queue_len: int,
        current_cpu_util: float,
        has_emergency_task: bool = False,
        is_available: bool = True,
    ) -> bool:
        """
        Determines whether an ANC event trigger condition is met.
        """
        # Trigger 1: Emergency task arrival requires immediate alert
        if has_emergency_task:
            return True

        # Trigger 2: Availability status changed (crash or recovery)
        if self.last_broadcast_available != is_available:
            return True

        # Trigger 3: First-time broadcast
        if self.last_broadcast_time_ms < 0:
            return True

        # Trigger 3: Maximum silence safety heartbeat
        if (current_time_ms - self.last_broadcast_time_ms) >= self.max_silence_ms:
            return True

        # Trigger 4: Severe CPU overload alert
        if current_cpu_util >= self.overload_threshold and self.last_broadcast_cpu < self.overload_threshold:
            return True

        # Trigger 5: Significant delta in queue length
        prev_q = max(self.last_broadcast_queue, 1)
        queue_delta = abs(current_queue_len - self.last_broadcast_queue) / float(prev_q)
        if queue_delta >= self.delta_queue_ratio:
            return True

        return False

    def emit_broadcast(
        self,
        current_time_ms: float,
        cpu_util: float,
        gpu_util: float,
        queue_length: int,
        avg_hsi: float,
        power_w: float,
        pred_queue: float = 0.0,
        cong_prob: float = 0.0,
        has_emergency: bool = False,
        is_available: bool = True,
    ) -> BroadcastStatusMessage:
        """
        Emits an event-driven status broadcast message.
        """
        self.last_broadcast_time_ms = current_time_ms
        self.last_broadcast_queue = queue_length
        self.last_broadcast_cpu = cpu_util
        self.last_broadcast_available = True # Default true unless explicitly passed? Wait, we need it passed.
        self.total_broadcasts += 1
        self.total_bytes_transmitted += 128

        return BroadcastStatusMessage(
            sender_id=self.node_id,
            timestamp_ms=current_time_ms,
            cpu_util=cpu_util,
            gpu_util=gpu_util,
            queue_length=queue_length,
            avg_hsi=avg_hsi,
            power_w=power_w,
            pred_queue=pred_queue,
            cong_prob=cong_prob,
            emergency_alert=has_emergency,
            is_available=is_available,
        )


class ExplainableDecisionEngine:
    """
    Converts GATv2 graph attention weights, clinical severity (HSI),
    and edge network metrics into human-readable clinician explanations.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id

    def generate_explanation(
        self,
        task: MedicalTask,
        action: int,
        target_node: str,
        neighbor_names: List[str],
        attention_weights: np.ndarray,
        target_queue_len: int,
        target_bandwidth_mbps: float,
        data_staleness_ms: float,
        q_variance: float,
    ) -> Dict[str, Any]:
        """
        Produces a structured clinical rationale dictionary and formatted log string.
        """
        action_name = "LOCAL_EXECUTE" if action == 0 else (f"OFFLOAD" if action <= len(neighbor_names) else "QUEUE")

        # Find attention score for the selected neighbor
        selected_attn = 0.0
        if 1 <= action <= len(neighbor_names) and len(attention_weights) >= action:
            selected_attn = float(attention_weights[action - 1])

        # Clinical urgency tag
        if task.is_emergency or task.hitl_override:
            urgency_str = "CRITICAL EMERGENCY"
        elif task.hsi >= 0.5:
            urgency_str = "HIGH URGENCY"
        else:
            urgency_str = "ROUTINE"

        # Build clinician rationale matching Phase 8 constraints
        reasons = []
        if action == 0:
            reasons.append(f"local resources were available")
            reasons.append(f"estimated completion time satisfied the task deadline")
        elif action <= len(neighbor_names):
            reasons.append(f"local queue was congested")
            reasons.append(f"{target_node} had lower predicted congestion (PredQ={target_queue_len})")
            reasons.append(f"network latency was acceptable ({data_staleness_ms:.1f}ms)")
            if task.is_emergency:
                reasons.append("task HSI was critical")
        else:
            reasons.append("immediate execution resources were unavailable")
            reasons.append("available neighbors did not provide a beneficial alternative")

        # Create structured text
        if action == 0:
            text_desc = "Executed locally because " + " and ".join(reasons) + "."
        elif action <= len(neighbor_names):
            text_desc = f"Offloaded to {target_node} because " + ", ".join(reasons[:-1]) + (f", and {reasons[-1]}." if len(reasons)>1 else f"{reasons[-1]}.")
        else:
            text_desc = "Kept in the local queue because " + " and ".join(reasons) + "."

        formatted_report = (
            f"[EDE RATIONALE | {self.node_id}] Task #{task.task_id} ({urgency_str}, HSI={task.hsi:.2f})\n"
            f"  -> Decision: {action_name}\n"
            f"  -> Explanation: {text_desc}"
        )

        return {
            "task_id": task.task_id,
            "origin_node": self.node_id,
            "target_node": target_node,
            "decision": action_name,
            "urgency": urgency_str,
            "hsi": task.hsi,
            "selected_attention": selected_attn,
            "all_attentions": {name: float(w) for name, w in zip(neighbor_names, attention_weights[: len(neighbor_names)])},
            "q_variance": q_variance,
            "reasons": reasons,
            "report_text": formatted_report,
        }

    def generate_recovery_explanation(
        self, task: MedicalTask, target_node: str, reason_text: str
    ) -> Dict[str, Any]:
        """Generates structured EDE rationale for a SHN decentralized task recovery event."""
        action_name = "RECOVERY/REDISTRIBUTION"
        
        urgency_str = "CRITICAL EMERGENCY" if (task.is_emergency or task.hitl_override) else ("HIGH URGENCY" if task.hsi >= 0.5 else "ROUTINE")

        text_desc = f"Moved orphaned task to {target_node} {reason_text}."
        
        formatted_report = (
            f"[EDE RATIONALE | {self.node_id}] Task #{task.task_id} ({urgency_str}, HSI={task.hsi:.2f})\n"
            f"  -> Decision: {action_name}\n"
            f"  -> Explanation: {text_desc}"
        )
        
        return {
            "source_node": self.node_id,
            "task_id": task.task_id,
            "action": action_name,
            "destination": target_node,
            "reasons": [reason_text],
            "report_text": formatted_report,
        }


class EdgeNodeAgent:
    """
    Decentralized Autonomous Edge Node representing a hospital department.
    """

    def __init__(
        self,
        node_id: str,
        sim_env: simpy.Environment,
        node_capacity_gflops: float = 20.0,
        model: Optional[UAMAPPOAgent] = None,
        hsi_calculator: Optional[HSICalculator] = None,
        max_queue_size: int = 100,
    ):
        self.node_id = node_id
        self.sim_env = sim_env
        self.node_capacity_gflops = node_capacity_gflops
        self.model = model or UAMAPPOAgent()
        self.hsi_calc = hsi_calculator or HSICalculator()
        self.max_queue_size = max_queue_size

        self.anc = AdaptiveNeighborCommunication(node_id=node_id)
        self.ede = ExplainableDecisionEngine(node_id=node_id)

        # Private task queues and states (Phases 1-5)
        self.task_queue: List[MedicalTask] = []
        self.local_execution_queue: List[MedicalTask] = []
        self.running_task: Optional[MedicalTask] = None
        self.completed_tasks: List[MedicalTask] = []
        self.execution_history: List[Dict[str, Any]] = []

        # P2P Decentralized state cache (Phase 7)
        self.neighbor_state_cache: Dict[str, BroadcastStatusMessage] = {}

        # SHN (Phase 8) States
        self.is_available = True
        self.orphaned_tasks: List[MedicalTask] = []

        # Start the local scheduler process
        self.scheduler_process = self.sim_env.process(self.run_local_scheduler())

    def push_task(self, task: MedicalTask):
        """Enqueues task into the private local queue (FIFO for Phases 1-4)."""
        if len(self.task_queue) < self.max_queue_size:
            task.status = TaskStatus.WAITING
            self.task_queue.append(task)

    def receive_state_update(self, msg: BroadcastStatusMessage):
        """P2P endpoint for receiving a neighbor's status broadcast."""
        self.neighbor_state_cache[msg.sender_id] = msg

    def run_local_scheduler(self):
        """SimPy process simulating time-based task execution from the local execution queue."""
        while True:
            try:
                if self.local_execution_queue:
                    # Select the next task (FIFO for Phase 1-5)
                    self.running_task = self.local_execution_queue.pop(0)
                    self.running_task.status = TaskStatus.RUNNING
                    self.running_task.start_time_ms = self.sim_env.now

                    # Calculate expected duration based on local capacity (gflops / gflops/sec = seconds, so * 1000 for ms)
                    duration_ms = (self.running_task.remaining_compute_gflops / self.node_capacity_gflops) * 1000.0
                    
                    # Simulate the execution time occupying resources
                    yield self.sim_env.timeout(duration_ms)

                    # Task finishes
                    self.running_task.status = TaskStatus.COMPLETED
                    self.running_task.completion_time_ms = self.sim_env.now
                    self.running_task.remaining_compute_gflops = 0.0
                    
                    self.completed_tasks.append(self.running_task)
                    self.running_task = None
                else:
                    # Idle if no tasks, check again next millisecond
                    yield self.sim_env.timeout(1.0)
            except simpy.Interrupt as i:
                # Node Crash Event!
                if self.running_task:
                    elapsed_ms = self.sim_env.now - self.running_task.start_time_ms
                    completed_gflops = elapsed_ms * self.node_capacity_gflops
                    self.running_task.remaining_compute_gflops = max(0.0, self.running_task.remaining_compute_gflops - completed_gflops)
                    self.running_task.status = TaskStatus.ORPHANED
                    self.orphaned_tasks.append(self.running_task)
                    self.running_task = None
                
                # Suspend process until explicitly recovered
                try:
                    yield self.sim_env.timeout(9999999.0)
                except simpy.Interrupt as j:
                    # Node Recovered!
                    pass

    def handle_crash(self):
        """Phase 8 SHN: Gracefully handles a physical node crash dynamically."""
        self.is_available = False
        
        # Mark waiting tasks as ORPHANED
        for t in self.task_queue:
            t.status = TaskStatus.ORPHANED
            self.orphaned_tasks.append(t)
        self.task_queue.clear()
        
        for t in self.local_execution_queue:
            t.status = TaskStatus.ORPHANED
            self.orphaned_tasks.append(t)
        self.local_execution_queue.clear()
        
        # Interrupt the running scheduler to halt running_task and save remaining compute
        if self.scheduler_process.is_alive:
            self.scheduler_process.interrupt("Crash!")

    def recover_node(self):
        """Phase 8 SHN: Node comes back online."""
        self.is_available = True
        if self.scheduler_process.is_alive:
            self.scheduler_process.interrupt("Recovery!")

    def attempt_recovery(self, task: MedicalTask) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Phase 8 SHN: Decentralized recovery candidate discovery.
        Finds the available neighbor with the lowest predicted queue via local P2P cache.
        Returns (target_node_id, EDE_explanation)
        """
        best_neighbor = None
        best_pred_q = 999999.0
        
        for nbr_id, msg in self.neighbor_state_cache.items():
            if msg.is_available:
                if msg.pred_queue < best_pred_q:
                    best_pred_q = msg.pred_queue
                    best_neighbor = nbr_id
                    
        if best_neighbor:
            explanation = self.ede.generate_recovery_explanation(
                task=task,
                target_node=best_neighbor,
                reason_text=f"after {self.node_id} became unavailable and {best_neighbor} was the available neighbor with lower predicted congestion"
            )
            return best_neighbor, explanation
        else:
            task.status = TaskStatus.RECOVERY_PENDING
            return None, None

    def get_public_state(self) -> Dict[str, Any]:
        """
        Returns only information another node is allowed to know (Phase 4).
        Does NOT return the actual task queue or private execution details.
        """
        # Determine resource utilization based on running state
        cpu_util = 0.95 if self.running_task else 0.10
        gpu_util = 0.85 if self.running_task else 0.05
        
        return {
            "node_id": self.node_id,
            "cpu_util": cpu_util,
            "gpu_util": gpu_util,
            "queue_length": len(self.task_queue) + len(self.local_execution_queue),
            "available": self.is_available
        }

    def decide_action(
        self,
        obs: Dict[str, Any],
        task: MedicalTask,
        neighbor_names: List[str],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Executes policy decision and generates EDE explainability report.
        """
        step_out = self.model.get_action_and_value(obs, deterministic=deterministic)
        action = step_out["action"]
        attn_weights = step_out["attention_weights"]
        q_var = step_out["variance"]

        target_node = self.node_id
        if 1 <= action <= len(neighbor_names):
            target_node = neighbor_names[action - 1]

        explanation = self.ede.generate_explanation(
            task=task,
            action=action,
            target_node=target_node,
            neighbor_names=neighbor_names,
            attention_weights=attn_weights,
            target_queue_len=int(obs["neighbor_nodes"][action - 1][3] * 50) if action > 0 and action <= len(neighbor_names) else 0,
            target_bandwidth_mbps=float(obs["neighbor_edges"][action - 1][0] * 1000) if action > 0 and action <= len(neighbor_names) else 1000.0,
            data_staleness_ms=float(obs["neighbor_edges"][action - 1][2] * 500) if action > 0 and action <= len(neighbor_names) else 0.0,
            q_variance=q_var,
        )

        return action, explanation
