"""
Dynamic Hospital Graph Constructor (DHGC) and Self-Healing Network (SHN) for GraphMARL.

Maintains dynamic graph topology G = (V, E) of hospital department edge nodes,
models link degradation, packet loss, data staleness, and provides automated
self-healing recovery from department node crashes.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import networkx as nx
import numpy as np


class HospitalGraph:
    """
    Manages the dynamic hospital edge network topology, node capacities,
    link metrics, degradation events, and self-healing mechanisms.
    """

    DEFAULT_DEPARTMENTS = [
        {"id": "ER", "name": "Emergency Room", "compute_gflops": 120.0, "gpu_gflops": 80.0, "ram_gb": 16.0, "power_w": 25.0},
        {"id": "ICU", "name": "Intensive Care Unit", "compute_gflops": 200.0, "gpu_gflops": 150.0, "ram_gb": 32.0, "power_w": 40.0},
        {"id": "Radiology", "name": "Radiology & Imaging", "compute_gflops": 250.0, "gpu_gflops": 200.0, "ram_gb": 32.0, "power_w": 50.0},
        {"id": "Surgery", "name": "Surgical Theaters", "compute_gflops": 150.0, "gpu_gflops": 100.0, "ram_gb": 16.0, "power_w": 30.0},
        {"id": "Cardiology", "name": "Cardiology Clinic", "compute_gflops": 100.0, "gpu_gflops": 60.0, "ram_gb": 16.0, "power_w": 20.0},
        {"id": "Ward", "name": "General Inpatient Ward", "compute_gflops": 50.0, "gpu_gflops": 20.0, "ram_gb": 8.0, "power_w": 10.0},
        {"id": "Lab", "name": "Pathology Laboratory", "compute_gflops": 60.0, "gpu_gflops": 20.0, "ram_gb": 8.0, "power_w": 12.0},
        {"id": "Pharmacy", "name": "Clinical Pharmacy", "compute_gflops": 40.0, "gpu_gflops": 10.0, "ram_gb": 4.0, "power_w": 8.0},
    ]

    DEFAULT_EDGES = [
        ("ER", "ICU", {"bandwidth_mbps": 1000.0, "latency_ms": 1.5, "reliability": 0.99}),
        ("ER", "Surgery", {"bandwidth_mbps": 1000.0, "latency_ms": 2.0, "reliability": 0.98}),
        ("ER", "Radiology", {"bandwidth_mbps": 1000.0, "latency_ms": 2.5, "reliability": 0.98}),
        ("ICU", "Surgery", {"bandwidth_mbps": 1000.0, "latency_ms": 1.0, "reliability": 0.99}),
        ("ICU", "Cardiology", {"bandwidth_mbps": 800.0, "latency_ms": 3.0, "reliability": 0.97}),
        ("Radiology", "Surgery", {"bandwidth_mbps": 1000.0, "latency_ms": 2.0, "reliability": 0.98}),
        ("Radiology", "Cardiology", {"bandwidth_mbps": 600.0, "latency_ms": 4.0, "reliability": 0.95}),
        ("Cardiology", "Ward", {"bandwidth_mbps": 400.0, "latency_ms": 6.0, "reliability": 0.94}),
        ("Ward", "Lab", {"bandwidth_mbps": 400.0, "latency_ms": 7.0, "reliability": 0.93}),
        ("Lab", "Pharmacy", {"bandwidth_mbps": 300.0, "latency_ms": 8.0, "reliability": 0.92}),
        ("Pharmacy", "ER", {"bandwidth_mbps": 400.0, "latency_ms": 6.0, "reliability": 0.94}),
        ("Ward", "ER", {"bandwidth_mbps": 500.0, "latency_ms": 5.0, "reliability": 0.95}),
    ]

    def __init__(
        self,
        departments: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.graph = nx.Graph()
        self.crashed_nodes: Set[str] = set()
        self.degraded_edges: Set[Tuple[str, str]] = set()

        deps = departments or self.DEFAULT_DEPARTMENTS
        edgs = edges or self.DEFAULT_EDGES

        for d in deps:
            self.graph.add_node(
                d["id"],
                name=d["name"],
                compute_gflops=d["compute_gflops"],
                gpu_gflops=d.get("gpu_gflops", 50.0),
                ram_gb=d.get("ram_gb", 16.0),
                power_w=d.get("power_w", 20.0),
                cpu_util=0.10,
                gpu_util=0.05,
                ram_avail=d.get("ram_gb", 16.0),
                queue_length=0,
                avg_queue_hsi=0.0,
                power_rate=d.get("power_w", 20.0) * 0.3,
                alive=True,
                heartbeat_time_ms=0.0,
            )

        for u, v, attrs in edgs:
            self.graph.add_edge(
                u,
                v,
                bandwidth_mbps=attrs.get("bandwidth_mbps", 500.0),
                base_bandwidth=attrs.get("bandwidth_mbps", 500.0),
                latency_ms=attrs.get("latency_ms", 5.0),
                base_latency=attrs.get("latency_ms", 5.0),
                reliability=attrs.get("reliability", 0.95),
                data_staleness_ms=0.0,
                active=True,
            )

    @property
    def node_ids(self) -> List[str]:
        return list(self.graph.nodes())

    @property
    def active_node_ids(self) -> List[str]:
        return [n for n in self.graph.nodes() if n not in self.crashed_nodes]

    def get_neighbors(self, node_id: str, active_only: bool = True) -> List[str]:
        if node_id not in self.graph:
            return []
        nbrs = list(self.graph.neighbors(node_id))
        if active_only:
            nbrs = [n for n in nbrs if n not in self.crashed_nodes]
        return nbrs

    def get_node_features(self, node_id: str) -> np.ndarray:
        """
        Returns normalized 6D node feature vector:
        [CPU_util, GPU_util, RAM_avail/32, Queue_len/50, Power/50, Avg_Queue_HSI]
        """
        if node_id not in self.graph or node_id in self.crashed_nodes:
            return np.zeros(6, dtype=np.float32)

        data = self.graph.nodes[node_id]
        return np.array(
            [
                float(data.get("cpu_util", 0.0)),
                float(data.get("gpu_util", 0.0)),
                float(data.get("ram_avail", 16.0) / 32.0),
                float(min(data.get("queue_length", 0) / 50.0, 1.0)),
                float(min(data.get("power_rate", 10.0) / 50.0, 1.0)),
                float(data.get("avg_queue_hsi", 0.0)),
            ],
            dtype=np.float32,
        )

    def get_edge_features(self, u: str, v: str) -> np.ndarray:
        """
        Returns normalized 4D edge feature vector:
        [Bandwidth/1000, Latency/20, Data_Staleness/500, Link_Reliability]
        """
        if not self.graph.has_edge(u, v) or u in self.crashed_nodes or v in self.crashed_nodes:
            return np.zeros(4, dtype=np.float32)

        data = self.graph[u][v]
        return np.array(
            [
                float(min(data.get("bandwidth_mbps", 100.0) / 1000.0, 1.0)),
                float(min(data.get("latency_ms", 5.0) / 20.0, 1.0)),
                float(min(data.get("data_staleness_ms", 0.0) / 500.0, 1.0)),
                float(data.get("reliability", 0.95)),
            ],
            dtype=np.float32,
        )

    def update_node_state(
        self,
        node_id: str,
        cpu_util: float,
        gpu_util: float,
        queue_len: int,
        avg_hsi: float,
        power_w: Optional[float] = None,
        current_time_ms: float = 0.0,
    ):
        if node_id in self.graph and node_id not in self.crashed_nodes:
            self.graph.nodes[node_id]["cpu_util"] = float(np.clip(cpu_util, 0.0, 1.0))
            self.graph.nodes[node_id]["gpu_util"] = float(np.clip(gpu_util, 0.0, 1.0))
            self.graph.nodes[node_id]["queue_length"] = int(queue_len)
            self.graph.nodes[node_id]["avg_queue_hsi"] = float(np.clip(avg_hsi, 0.0, 1.0))
            if power_w is not None:
                self.graph.nodes[node_id]["power_rate"] = float(power_w)
            self.graph.nodes[node_id]["heartbeat_time_ms"] = current_time_ms

    def update_edge_staleness(self, u: str, v: str, staleness_ms: float):
        if self.graph.has_edge(u, v):
            self.graph[u][v]["data_staleness_ms"] = float(staleness_ms)

    def inject_link_degradation(self, u: str, v: str, degradation_factor: float = 0.2):
        """Simulates network congestion / interference reducing bandwidth and spiking latency."""
        if self.graph.has_edge(u, v):
            base_bw = self.graph[u][v]["base_bandwidth"]
            base_lat = self.graph[u][v]["base_latency"]
            self.graph[u][v]["bandwidth_mbps"] = base_bw * degradation_factor
            self.graph[u][v]["latency_ms"] = base_lat / max(degradation_factor, 0.05)
            self.graph[u][v]["reliability"] = max(0.5, self.graph[u][v]["reliability"] * 0.8)
            self.degraded_edges.add((u, v))

    def restore_link(self, u: str, v: str):
        if self.graph.has_edge(u, v):
            self.graph[u][v]["bandwidth_mbps"] = self.graph[u][v]["base_bandwidth"]
            self.graph[u][v]["latency_ms"] = self.graph[u][v]["base_latency"]
            self.graph[u][v]["reliability"] = 0.98
            self.degraded_edges.discard((u, v))
            self.degraded_edges.discard((v, u))

    def inject_node_crash(self, node_id: str):
        """Simulates catastrophic power failure or hardware crash on an edge device."""
        if node_id in self.graph:
            self.crashed_nodes.add(node_id)
            self.graph.nodes[node_id]["alive"] = False
            self.graph.nodes[node_id]["cpu_util"] = 0.0
            self.graph.nodes[node_id]["queue_length"] = 0

    def recover_node(self, node_id: str):
        """Self-Healing: Restores a crashed node back into the hospital topology."""
        if node_id in self.crashed_nodes:
            self.crashed_nodes.remove(node_id)
            self.graph.nodes[node_id]["alive"] = True

    def calculate_transmission_delay_ms(self, u: str, v: str, data_size_mb: float) -> float:
        """
        Calculates realistic transmission time + propagation latency between nodes.
        Delay = Propagation Latency + (Data Size (Mbits) / Bandwidth (Mbps)).
        """
        if u == v:
            return 0.0
        if not self.graph.has_edge(u, v) or u in self.crashed_nodes or v in self.crashed_nodes:
            return 9999.0  # Infinite / unreachable penalty

        edge = self.graph[u][v]
        bw_mbps = max(edge.get("bandwidth_mbps", 100.0), 1.0)
        lat_ms = edge.get("latency_ms", 5.0)
        size_mbits = data_size_mb * 8.0
        trans_ms = (size_mbits / bw_mbps) * 1000.0
        return float(lat_ms + trans_ms)
