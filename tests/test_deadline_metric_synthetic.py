"""
Synthetic evaluation test to verify deadline-miss metric accounting.

Verifies:
1. An easy task that finishes before deadline -> completed_before_deadline (Success)
2. A task that finishes after deadline -> completed_after_deadline (Deadline Miss)
3. A task that remains in queue after its deadline -> unfinished_past_deadline (Deadline Miss)
4. A task that remains in queue within its deadline -> unfinished_within_deadline (Success / Pending)
5. A task that is dropped due to queue capacity -> dropped_or_cancelled

Checks that:
- Every task is accounted for exactly once
- deadline_misses = completed_after_deadline + unfinished_past_deadline
- deadline_miss_rate = deadline_misses / total_generated_tasks
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.workload_generator import MedicalTask, TaskType, TaskStatus
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator
from src.utils.eval_helpers import collect_episode_metrics


def run_synthetic_test():
    print("=" * 70)
    print("RUNNING SYNTHETIC DEADLINE-MISS METRIC VERIFICATION")
    print("=" * 70)

    # 1. Create a controlled environment
    graph = HospitalGraph(seed=42)
    workload = WorkloadGenerator(seed=42)
    env = HospitalEdgeEnv(
        hospital_graph=graph,
        workload_gen=workload,
        max_steps=10,
        task_arrival_prob=0.0, # Disable background random spawning for full control
        seed=42,
    )
    env.reset(seed=42)

    # Clear initial auto-generated tasks so we have 100% control over the test tasks
    for agent in env.agents:
        env.edge_nodes[agent].task_queue.clear()
        env.edge_nodes[agent].local_execution_queue.clear()
        env.edge_nodes[agent].completed_tasks.clear()
    env.all_tasks.clear()

    # Define 3 specific synthetic tasks as requested:
    # Task 1: Easy task that finishes before deadline
    # 0.5 GFLOPs on 50 GFLOPs node takes 10 ms. Deadline is 100 ms.
    t1 = MedicalTask(
        task_id="t1_easy_success",
        patient_id="P1",
        origin_node="ER",
        task_type=TaskType.EMERGENCY_TRIAGE,
        arrival_time_ms=0.0,
        data_size_mb=0.1,
        required_gflops=0.5,
        deadline_ms=100.0,
        hsi=0.9,
        priority_score=1.0,
    )

    # Task 2: Task that finishes after deadline
    # 4.0 GFLOPs on 50 GFLOPs node takes 80 ms. Deadline is 30 ms.
    t2 = MedicalTask(
        task_id="t2_slow_miss",
        patient_id="P2",
        origin_node="ER",
        task_type=TaskType.REALTIME_ECG_ARRHYTHMIA,
        arrival_time_ms=0.0,
        data_size_mb=1.0,
        required_gflops=4.0,
        deadline_ms=30.0,
        hsi=0.6,
        priority_score=0.8,
    )

    # Task 3: Task that remains in queue after its deadline
    # Left in queue on ICU node (Action QUEUE or never executed).
    # Deadline is 25 ms. Episode will run for 100 ms (2 steps of 50 ms).
    t3 = MedicalTask(
        task_id="t3_queued_past_deadline",
        patient_id="P3",
        origin_node="ICU",
        task_type=TaskType.EMERGENCY_TRIAGE,
        arrival_time_ms=0.0,
        data_size_mb=0.1,
        required_gflops=1.0,
        deadline_ms=25.0,
        hsi=0.85,
        priority_score=0.9,
    )

    # Add all 3 tasks to environment tracking
    env.all_tasks.extend([t1, t2, t3])

    # Put t1 and t2 on ER
    env.edge_nodes["ER"].push_task(t1)
    env.edge_nodes["ER"].push_task(t2)

    # Put t3 on ICU
    env.edge_nodes["ICU"].push_task(t3)

    # Step 1 (0 -> 50 ms):
    # Action 0 (LOCAL) on ER schedules t1 to local_execution_queue.
    # Action 7 (QUEUE) on ICU keeps t3 waiting in task_queue.
    actions = {
        "ER": 0, # LOCAL for t1
        "ICU": env.action_dim - 1, # QUEUE for t3
    }
    env.step(actions)

    # After step 1 (sim_time_ms = 50 ms):
    # t1 (took 10 ms) is COMPLETED at 10 ms! Latency = 10 ms <= 100 ms deadline -> Success
    # ER task_queue now has t2 at front.

    # Step 2 (50 -> 100 ms):
    # Action 0 (LOCAL) on ER schedules t2 to local_execution_queue.
    # Action 7 (QUEUE) on ICU continues keeping t3 waiting in task_queue.
    actions = {
        "ER": 0, # LOCAL for t2
        "ICU": env.action_dim - 1, # QUEUE for t3
    }
    env.step(actions)

    # After step 2 (sim_time_ms = 100 ms):
    # t2 (duration 80 ms, started at 50 ms) is not finished yet or finishes?
    # At 100 ms, t2 has run for 50 ms of its 80 ms.
    
    # Step 3 (100 -> 150 ms):
    # Let t2 complete (needs 30 ms more, so finishes at 130 ms).
    actions = {
        "ER": 0,
        "ICU": env.action_dim - 1,
    }
    env.step(actions)

    # Now at sim_time_ms = 150 ms:
    # t1: completed at 10 ms. Latency = 10 ms <= 100 ms -> completed_before_deadline
    # t2: completed at 130 ms. Latency = 130 ms > 30 ms -> completed_after_deadline
    # t3: never executed. At 150 ms, elapsed = 150 ms > 25 ms -> unfinished_past_deadline

    # Collect metrics using the corrected evaluation
    m = collect_episode_metrics(env)

    print("\nMETRICS BREAKDOWN:")
    print(f"Generated tasks:            {m['total_generated_tasks']}")
    print(f"Completed before deadline:  {m['completed_before_deadline']}")
    print(f"Completed after deadline:   {m['completed_after_deadline']}")
    print(f"Unfinished past deadline:   {m['unfinished_past_deadline']}")
    print(f"Unfinished within deadline: {m['unfinished_within_deadline']}")
    print(f"Dropped/cancelled:          {m['dropped_or_cancelled']}")
    print(f"Total deadline misses:      {m['deadline_misses']}")
    print(f"Deadline-miss rate:         {m['deadline_miss_rate']:.4f} ({m['deadline_miss_rate']*100:.1f}%)")

    # Verification assertions
    assert m["total_generated_tasks"] == 3, f"Expected 3 tasks, got {m['total_generated_tasks']}"
    assert m["completed_before_deadline"] == 1, f"Expected 1 completed before deadline, got {m['completed_before_deadline']}"
    assert m["completed_after_deadline"] == 1, f"Expected 1 completed after deadline, got {m['completed_after_deadline']}"
    assert m["unfinished_past_deadline"] == 1, f"Expected 1 unfinished past deadline, got {m['unfinished_past_deadline']}"
    assert m["dropped_or_cancelled"] == 0, f"Expected 0 dropped, got {m['dropped_or_cancelled']}"
    assert m["deadline_misses"] == 2, f"Expected 2 deadline misses (1 completed after + 1 unfinished past), got {m['deadline_misses']}"
    assert m["deadline_misses"] == m["completed_after_deadline"] + m["unfinished_past_deadline"]
    assert abs(m["deadline_miss_rate"] - (2.0 / 3.0)) < 1e-5

    # Check conservation law: sum of mutually exclusive states equals total generated
    partition_sum = (
        m["completed_before_deadline"]
        + m["completed_after_deadline"]
        + m["unfinished_past_deadline"]
        + m["unfinished_within_deadline"]
        + m["dropped_or_cancelled"]
    )
    assert partition_sum == m["total_generated_tasks"], (
        f"Partition sum {partition_sum} != total generated {m['total_generated_tasks']}"
    )

    print("\nALL ASSERTIONS PASSED! The metric correctly classifies:")
    print("  Case 1 (Easy task finishes before deadline):  SUCCESS (completed_before_deadline)")
    print("  Case 2 (Task finishes after deadline):        DEADLINE MISS (completed_after_deadline)")
    print("  Case 3 (Task remains in queue past deadline): DEADLINE MISS (unfinished_past_deadline)")
    print("  Conservation law: sum(subsets) == total_generated_tasks holds exactly.")
    print("=" * 70)


def run_extended_synthetic_test():
    print("\n" + "=" * 70)
    print("RUNNING EXTENDED 5-CASE SYNTHETIC VERIFICATION")
    print("=" * 70)

    graph = HospitalGraph(seed=42)
    workload = WorkloadGenerator(seed=42)
    env = HospitalEdgeEnv(
        hospital_graph=graph,
        workload_gen=workload,
        max_steps=10,
        task_arrival_prob=0.0,
        seed=42,
    )
    env.reset(seed=42)

    for agent in env.agents:
        env.edge_nodes[agent].task_queue.clear()
        env.edge_nodes[agent].local_execution_queue.clear()
        env.edge_nodes[agent].completed_tasks.clear()
    env.all_tasks.clear()

    # 1. Completed before deadline (arrival=0, finished=10, deadline=100) -> SUCCESS
    t1 = MedicalTask("t1", "P1", "ER", TaskType.EMERGENCY_TRIAGE, 0.0, 0.1, 0.5, 100.0, 0.9, 1.0)
    # 2. Completed after deadline (arrival=0, finished=130, deadline=30) -> MISS
    t2 = MedicalTask("t2", "P2", "ER", TaskType.REALTIME_ECG_ARRHYTHMIA, 0.0, 1.0, 4.0, 30.0, 0.6, 0.8)
    # 3. Queued past deadline (arrival=0, unfinished at 150, deadline=25) -> MISS
    t3 = MedicalTask("t3", "P3", "ICU", TaskType.EMERGENCY_TRIAGE, 0.0, 0.1, 1.0, 25.0, 0.85, 0.9)
    # 4. Queued within deadline (arrival=120, unfinished at 150, deadline=100) -> PENDING (elapsed 30 <= 100)
    t4 = MedicalTask("t4", "P4", "ICU", TaskType.ROUTINE_VITALS_LOGGING, 120.0, 0.1, 0.5, 100.0, 0.2, 0.3)
    # 5. Dropped task (queue full reject: status=GENERATED) -> DROPPED
    t5 = MedicalTask("t5", "P5", "ER", TaskType.ROUTINE_VITALS_LOGGING, 0.0, 0.1, 0.5, 100.0, 0.2, 0.3)
    t5.status = TaskStatus.GENERATED

    env.all_tasks.extend([t1, t2, t3, t4, t5])

    env.edge_nodes["ER"].push_task(t1)
    env.edge_nodes["ER"].push_task(t2)
    env.edge_nodes["ICU"].push_task(t3)
    # Push t4 at step 3 (sim_time_ms = 120)
    
    # Step 1 (0 -> 50 ms)
    env.step({"ER": 0, "ICU": env.action_dim - 1})
    # Step 2 (50 -> 100 ms)
    env.step({"ER": 0, "ICU": env.action_dim - 1})
    
    # Push t4 into ICU queue
    env.edge_nodes["ICU"].push_task(t4)
    # Step 3 (100 -> 150 ms)
    env.step({"ER": 0, "ICU": env.action_dim - 1})

    m = collect_episode_metrics(env)

    print("\nEXTENDED METRICS BREAKDOWN:")
    print(f"Generated tasks:            {m['total_generated_tasks']}")
    print(f"Completed before deadline:  {m['completed_before_deadline']}")
    print(f"Completed after deadline:   {m['completed_after_deadline']}")
    print(f"Unfinished past deadline:   {m['unfinished_past_deadline']}")
    print(f"Unfinished within deadline: {m['unfinished_within_deadline']}")
    print(f"Dropped/cancelled:          {m['dropped_or_cancelled']}")
    print(f"Total deadline misses:      {m['deadline_misses']}")
    print(f"Deadline-miss rate:         {m['deadline_miss_rate']:.4f} ({m['deadline_miss_rate']*100:.1f}%)")

    assert m["total_generated_tasks"] == 5
    assert m["completed_before_deadline"] == 1
    assert m["completed_after_deadline"] == 1
    assert m["unfinished_past_deadline"] == 1
    assert m["unfinished_within_deadline"] == 1
    assert m["dropped_or_cancelled"] == 1
    assert m["deadline_misses"] == 2
    assert abs(m["deadline_miss_rate"] - (2.0 / 5.0)) < 1e-5

    partition_sum = (
        m["completed_before_deadline"]
        + m["completed_after_deadline"]
        + m["unfinished_past_deadline"]
        + m["unfinished_within_deadline"]
        + m["dropped_or_cancelled"]
    )
    assert partition_sum == m["total_generated_tasks"]
    print("\nALL 5 CASES ACCURATELY CLASSIFIED AND VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    run_synthetic_test()
    run_extended_synthetic_test()
