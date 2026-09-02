"""Baseline and expert routing policies for GraphMARL training and evaluation."""


def greedy_action(env, node_id):
    """Greedy queue baseline: offload to neighbor with smallest current queue."""
    min_q = len(env.edge_nodes[node_id].task_queue) + len(
        env.edge_nodes[node_id].local_execution_queue
    )
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action


def local_action(env, node_id):
    """Local-only baseline: always execute on the current node."""
    return 0


def expert_action(env, node_id, offload_margin=2):
    """
    Conservative expert: execute locally unless a neighbor queue is
    significantly shorter (avoids costly ping-pong offloading).
    """
    local_q = len(env.edge_nodes[node_id].task_queue) + len(
        env.edge_nodes[node_id].local_execution_queue
    )
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr in env.graph.crashed_nodes:
            continue
        msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
        if msg and msg.queue_length + offload_margin <= local_q:
            return i + 1
    return 0


POLICY_REGISTRY = {
    "greedy": greedy_action,
    "local": local_action,
    "expert": expert_action,
}
