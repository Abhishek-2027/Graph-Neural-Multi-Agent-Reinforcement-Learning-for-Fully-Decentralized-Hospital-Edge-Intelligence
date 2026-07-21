# GraphMARL: Graph Neural Multi-Agent Reinforcement Learning for Fully Decentralized Peer-to-Peer Hospital Edge Orchestration

## 1. Problem Statement & Baseline
While recent literature, including the baseline paper ("Task prioritization and distributed deep reinforcement learning for healthcare management in Cloud-Edge environments"), introduces effective methods for prioritizing healthcare tasks using a Health Severity Index (HSI) and Distributed Deep Reinforcement Learning (DRL), it suffers from several critical limitations:
- **Centralized Orchestration:** It relies on a logically centralized orchestrator or global state knowledge for task offloading.
- **Vertical Offloading Only:** It focuses on Device-to-Edge and Edge-to-Cloud offloading, missing the massive potential of horizontal Edge-to-Edge (Peer-to-Peer) collaboration across hospital departments.
- **Lack of Graph Intelligence:** It ignores the inherent graph-like topology of interconnected hospital departments.
- **Simulation-Only Validation:** It lacks validation on real-world hardware under realistic communication failures.
- **Single Point of Failure:** Centralized scheduling makes the system vulnerable to node crashes and network partitions.

## 2. Proposed Solution: How We Will Solve These Problems
Our proposed framework, **GraphMARL**, completely eliminates the central scheduler. Every hospital department (e.g., ICU, Radiology, ER) acts as an autonomous, intelligent edge agent capable of Peer-to-Peer collaboration. We solve the gaps in the base paper through the following methodologies:

- **Fully Decentralized Peer-to-Peer Orchestration:** Nodes make decisions independently. If a node fails, the rest of the hospital continues to function, routing tasks horizontally to other departments.
- **Dynamic Hospital Graph Representation:** The hospital is modeled as a dynamic graph $G = (V, E)$, where vertices are departments and edges are communication links. Node and edge features update continuously.
- **Hybrid State Fusion (SCNN + GNN):** We utilize a Stacked CNN (SCNN) to compress the local department's high-dimensional state (compute, energy) and fuse it with a Graph Neural Network (GNN) that aggregates compact neighbor embeddings. 
- **Uncertainty-Aware Decentralized Decision Making:** We integrate an Uncertainty-Aware Multi-Agent Proximal Policy Optimization (MAPPO) algorithm. The agent calculates the variance of its Q-value predictions for neighboring nodes. If neighbor data is stale or network congestion is high, the agent penalizes offloading to that neighbor, preventing catastrophic delays for critical medical tasks.
- **Human-in-the-Loop (HITL) Override:** Unlike purely autonomous systems, clinicians can inject real-time priority overrides which dynamically propagate through the graph, forcing neighboring nodes to clear their queues for emergency tasks.
- **Real Hardware Validation:** We will deploy the system on a testbed of NVIDIA Jetson boards (Nano, Xavier, Orin) simulating real hospital departments, injecting realistic network delays, packet loss, and node crashes.

---

## 3. Core Methodologies Introduced

### A. Dynamic Hospital Graph Constructor (DHGC)
Maintains the evolving graph of departments and communication links using local observations. Features include CPU/GPU utilization, queue length, and network bandwidth.

### B. Hybrid Perception Network (SCNN + NAGEN)
Combines two perception layers:
1. **SCNN:** Compresses high-dimensional local state parameters.
2. **Neighbor-Aware Graph Embedding Network (NAGEN):** Utilizes GNNs (like GraphSAGE) to produce node embeddings from local neighborhoods. 
The fused embedding (`[Local SCNN | Neighbor GNN]`) captures both local capacity and surrounding hospital congestion without global knowledge.

### C. Health Severity Priority Encoder (HSPE) & HITL Override
Extends the Health Severity Index (HSI). HSI is integrated directly into the MARL state space. Furthermore, the **HITL Override Protocol** allows clinicians to manually flag tasks as emergencies, instantly elevating the task's HSI and propagating "Emergency Override Embeddings" to neighbors.

### D. Uncertainty-Aware GraphMARL Scheduler (UA-GMS)
A multi-agent policy where each node learns independently. 
To prevent risky offloading due to stale neighbor information, the Q-learning objective incorporates variance-based uncertainty estimation. High prediction variance mathematically penalizes the action.
The Reward function is multi-objective:
$$R = \alpha U - \beta L - \gamma E - \delta D - \eta C + \lambda P - \mu \text{Var}(Q)$$
Where $U$ is utilization, $L$ is latency, $E$ is energy, $D$ is deadline misses, $C$ is communication cost, $P$ is critical-task completion, and $\text{Var}(Q)$ is the prediction uncertainty penalty.

### E. Adaptive Neighbor Negotiation Protocol (ANNP)
A lightweight protocol for exchanging compact graph embeddings and predicted finish times instead of full resource tables, drastically reducing communication overhead.

### F. Dynamic Failure Recovery Module (DFRM)
Handles node failures (e.g., heartbeats drop), stale information, and network partitions. The graph dynamically updates, and nodes route around failures autonomously.

---

## 4. Workflows & Architecture Diagrams

### System Architecture Workflow
```mermaid
graph TD
    A[Medical IoT Devices] -->|Raw Vitals & Images| B(Hybrid Transformer-BiLSTM Feature Extraction)
    B -->|Task Metadata| C[Health Severity Estimator]
    C -->|Task ID, HSI, Compute Needs| D{Task Characterization Layer}
    D -->|Incoming Task| E[Local Department Edge Node e.g., ICU, ER]
    
    subgraph Clinician Feedback
    Z[Doctor/System Admin] -->|HITL Override Alert| C
    end
    
    subgraph Decentralized Graph Intelligence
    E --> F[Neighbor Discovery via ANNP]
    F -->|Exchange Embeddings| G[Distributed Graph Construction]
    G -->|Local State| H1[SCNN Feature Extractor]
    G -->|Neighbor States| H2[GNN Encoder GAT/GraphSAGE]
    H1 --> I{State Fusion}
    H2 --> I
    end
    
    subgraph Multi-Agent Decision Engine
    I -->|Fused State| J[Uncertainty-Aware MARL MAPPO]
    J -->|Calculate Q-Variance| K{Local Offloading Decision Engine}
    end
    
    K -->|Low Uncertainty| L[Local Execution Queue]
    K -->|Offload Horizontal| M[Neighbor Edge Node]
    K -->|High Uncertainty| N[Retry / Escalate]
    
    style A fill:#e1f5fe,stroke:#01579b
    style J fill:#fff9c4,stroke:#fbc02d
    style H2 fill:#e8f5e9,stroke:#2e7d32
    style Z fill:#ffcdd2,stroke:#c62828
```

### GraphMARL Decision Flow (Perception to Execution)
```mermaid
flowchart LR
    subgraph Perception Phase
    A1[Local State Features] --> SCNN[Stacked CNN]
    A2[Neighbor Embeddings] --> GNN[Graph Neural Network]
    SCNN --> Fused[Fused State Representation]
    GNN --> Fused
    end
    
    subgraph Decision Phase
    Fused --> MARL[UA-MARL Policy]
    T[Task Priority / HSI / HITL] --> MARL
    MARL -->|Evaluate Q-Variance| D{Decision Action}
    end
    
    subgraph Execution Phase
    D -->|Execute Local| E1[Local GPU/CPU]
    D -->|Edge-to-Edge Offload Left| E2[Neighbor Dept A]
    D -->|Edge-to-Edge Offload Right| E3[Neighbor Dept B]
    D -->|Delay| E4[Wait Queue]
    end
    
    style GNN fill:#e8f5e9,stroke:#2e7d32
    style MARL fill:#fff9c4,stroke:#fbc02d
    style SCNN fill:#e8f5e9,stroke:#2e7d32
```

---

## 5. Our 15 Contributions

1. **Fully Decentralized Architecture:** Propose the first fully decentralized peer-to-peer hospital edge orchestration framework without any central scheduler.
2. **Horizontal Edge-to-Edge Collaboration:** Shift the paradigm from vertical (Device-to-Cloud) offloading to lateral Peer-to-Peer department collaboration.
3. **Dynamic Graph Modeling:** Model the hospital infrastructure as a dynamic graph where departments are nodes and communication links are graph edges.
4. **Hybrid Perception Module (SCNN + GNN):** Introduce a dual-perception architecture combining SCNNs for high-dimensional local states and GNNs for neighborhood-aware resource representation.
5. **GraphMARL Framework:** Develop a GraphMARL scheduling framework by integrating Graph Neural Networks with Multi-Agent Reinforcement Learning.
6. **Uncertainty-Aware Decentralized Policy:** Implement an uncertainty estimation mechanism that calculates Q-value variance to prevent high-risk offloading in dynamic, partially observable network conditions.
7. **Human-in-the-Loop (HITL) Override:** Design a closed-loop feedback mechanism allowing clinicians to inject real-time priority overrides that instantly propagate through the graph.
8. **Lightweight Negotiation Protocol:** Design a lightweight neighbor-to-neighbor negotiation protocol (ANNP) using graph embeddings instead of complete resource exchange.
9. **HSI-Integrated Scheduling:** Develop deadline-aware and Health Severity Index (HSI)-based intelligent task scheduling for critical healthcare applications.
10. **Congestion-Aware Offloading:** Introduce adaptive congestion-aware task offloading using graph-based neighborhood intelligence.
11. **Dynamic Failure Recovery:** Develop dynamic failure recovery mechanisms for node failures, stale information, and network partitioning.
12. **Fairness-Aware Workloads:** Incorporate fairness-aware decentralized scheduling to balance workloads among hospital departments.
13. **Low Communication Overhead:** Minimize communication overhead through localized graph aggregation and neighbor-only information exchange.
14. **Real Hardware Validation:** Validate the proposed framework on a real Jetson-based hospital edge testbed in addition to extensive simulations.
15. **Comprehensive Superiority:** Demonstrate improved latency, deadline satisfaction, resource utilization, scalability, robustness, and fault tolerance over existing healthcare edge scheduling approaches.
