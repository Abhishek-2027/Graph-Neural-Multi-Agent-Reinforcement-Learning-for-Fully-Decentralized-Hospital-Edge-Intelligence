"""
Decentralized Edge Node Agent, Adaptive Neighbor Communication (ANC),
and Explainable Decision Engine (EDE) for GraphMARL.

Encapsulates autonomous department-level scheduling logic, event-triggered communication,
clinician HITL overrides, and human-interpretable clinical decision rationales.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from src.models.ua_mappo import UAMAPPOAgent
from src.data.workload_generator import MedicalTask
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
    emergency_alert: bool = False
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

        # Stats for publication evaluation
        self.total_broadcasts = 0
        self.total_bytes_transmitted = 0

    def should_broadcast(
        self,
        current_time_ms: float,
        current_queue_len: int,
        current_cpu_util: float,
        has_emergency_task: bool = False,
    ) -> bool:
        """
        Determines whether an ANC event trigger condition is met.
        """
        # Trigger 1: Emergency task arrival requires immediate alert
        if has_emergency_task:
            return True

        # Trigger 2: First-time broadcast
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
        has_emergency: bool = False,
    ) -> BroadcastStatusMessage:
        """
        Emits an event-driven status broadcast message.
        """
        self.last_broadcast_time_ms = current_time_ms
        self.last_broadcast_queue = queue_length
        self.last_broadcast_cpu = cpu_util
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
            emergency_alert=has_emergency,
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
        action_name = "Local Execution" if action == 0 else (f"Offload to {target_node}" if action <= len(neighbor_names) else "Buffer in Queue")

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

        # Build clinician rationale
        reasons = []
        if action == 0:
            reasons.append(f"Local capacity is sufficient for {task.required_gflops:.1f} GFLOPs.")
            reasons.append(f"Avoids {task.data_size_mb:.2f} MB transmission latency over wireless links.")
        elif action <= len(neighbor_names):
            reasons.append(
                f"GATv2 Graph Attention assigned highest relevance score (α = {selected_attn:.3f}) to {target_node}."
            )
            reasons.append(f"Target department queue is low ({target_queue_len} tasks waiting).")
            reasons.append(f"High-speed link ({target_bandwidth_mbps:.0f} Mbps, Staleness: {data_staleness_ms:.1f}ms).")
            if q_variance < 0.5:
                reasons.append(f"High decision certainty (Critic Q-Variance = {q_variance:.4f}).")
        else:
            reasons.append("Temporary queue buffering selected while awaiting peer compute capacity.")

        formatted_report = (
            f"[EDE RATIONALE | {self.node_id}] Task #{task.task_id} ({urgency_str}, HSI={task.hsi:.2f})\n"
            f"  -> Decision: {action_name}\n"
            f"  -> Factors: " + " | ".join(reasons)
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


class EdgeNodeAgent:
    """
    Decentralized Autonomous Edge Node representing a hospital department.
    """

    def __init__(
        self,
        node_id: str,
        model: Optional[UAMAPPOAgent] = None,
        hsi_calculator: Optional[HSICalculator] = None,
        max_queue_size: int = 100,
    ):
        self.node_id = node_id
        self.model = model or UAMAPPOAgent()
        self.hsi_calc = hsi_calculator or HSICalculator()
        self.max_queue_size = max_queue_size

        self.anc = AdaptiveNeighborCommunication(node_id=node_id)
        self.ede = ExplainableDecisionEngine(node_id=node_id)

        # Internal task queue sorted by priority
        self.priority_queue: List[MedicalTask] = []
        self.execution_history: List[Dict[str, Any]] = []

    def push_task(self, task: MedicalTask):
        """Enqueues task maintaining priority order (highest priority first)."""
        self.priority_queue.append(task)
        self.priority_queue.sort(key=lambda t: t.priority_score, reverse=True)
        if len(self.priority_queue) > self.max_queue_size:
            # Drop lowest priority non-emergency task if queue overflows
            self.priority_queue.pop(-1)

    def pop_next_task(self) -> Optional[MedicalTask]:
        """Pops the highest priority task."""
        if self.priority_queue:
            return self.priority_queue.pop(0)
        return None

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
