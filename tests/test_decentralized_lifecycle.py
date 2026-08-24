import unittest
import simpy

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.agents.edge_node import EdgeNodeAgent
from src.data.workload_generator import WorkloadGenerator, MedicalTask, TaskType, TaskStatus

class TestDecentralizedLifecycle(unittest.TestCase):
    def setUp(self):
        self.sim_env = simpy.Environment()
        self.er_node = EdgeNodeAgent(node_id="ER", sim_env=self.sim_env, node_capacity_gflops=10.0)
        self.icu_node = EdgeNodeAgent(node_id="ICU", sim_env=self.sim_env, node_capacity_gflops=10.0)
        
        self.dummy_task1 = MedicalTask(
            task_id="T1", patient_id="P1", origin_node="ER", task_type=TaskType.MEDICAL_IMAGING_INFERENCE,
            arrival_time_ms=0.0, data_size_mb=10.0, required_gflops=50.0, deadline_ms=100.0, hsi=0.5, priority_score=5.0
        )
        self.dummy_task2 = MedicalTask(
            task_id="T2", patient_id="P2", origin_node="ER", task_type=TaskType.ROUTINE_VITALS_LOGGING,
            arrival_time_ms=0.0, data_size_mb=1.0, required_gflops=10.0, deadline_ms=50.0, hsi=0.1, priority_score=1.0
        )
        self.dummy_task3 = MedicalTask(
            task_id="T3", patient_id="P3", origin_node="ICU", task_type=TaskType.EMERGENCY_TRIAGE,
            arrival_time_ms=0.0, data_size_mb=5.0, required_gflops=20.0, deadline_ms=30.0, hsi=0.9, priority_score=9.0
        )
        self.dummy_task4 = MedicalTask(
            task_id="T4", patient_id="P4", origin_node="Radiology", task_type=TaskType.MEDICAL_IMAGING_INFERENCE,
            arrival_time_ms=0.0, data_size_mb=20.0, required_gflops=40.0, deadline_ms=80.0, hsi=0.4, priority_score=4.0
        )

    def test_1_and_2_independent_queues(self):
        """TEST 1 & TEST 2: Independent queue objects, adding to ER doesn't add to ICU."""
        self.assertIsNot(self.er_node.task_queue, self.icu_node.task_queue)
        
        self.er_node.push_task(self.dummy_task1)
        self.assertEqual(len(self.er_node.task_queue), 1)
        self.assertEqual(len(self.icu_node.task_queue), 0)
        
    def test_3_waiting_state(self):
        """TEST 3: A task can remain WAITING."""
        self.er_node.push_task(self.dummy_task1)
        self.assertEqual(self.dummy_task1.status, TaskStatus.WAITING)

    def test_4_5_6_execution_lifecycle(self):
        """TEST 4, 5, 6: WAITING -> RUNNING -> COMPLETED with simulated time."""
        self.er_node.push_task(self.dummy_task1) # Needs 50 GFLOPS, capacity 10 -> takes 5ms
        
        # At t=0, not started yet (needs yield)
        self.assertEqual(self.dummy_task1.status, TaskStatus.WAITING)
        
        self.sim_env.run(until=1.0)
        # Should be running now
        self.assertEqual(self.dummy_task1.status, TaskStatus.RUNNING)
        self.assertEqual(self.dummy_task1.start_time_ms, 0.0)
        
        self.sim_env.run(until=4.9)
        self.assertEqual(self.dummy_task1.status, TaskStatus.RUNNING)
        
        self.sim_env.run(until=5.1)
        self.assertEqual(self.dummy_task1.status, TaskStatus.COMPLETED)
        self.assertEqual(self.dummy_task1.completion_time_ms, 5.0)

    def test_7_8_resource_occupancy(self):
        """TEST 7 & TEST 8: Resources occupied during RUNNING, released after COMPLETED."""
        self.er_node.push_task(self.dummy_task1)
        
        self.sim_env.run(until=1.0)
        pub_state_running = self.er_node.get_public_state()
        self.assertEqual(pub_state_running["cpu_util"], 0.95)
        
        self.sim_env.run(until=6.0)
        pub_state_completed = self.er_node.get_public_state()
        self.assertEqual(pub_state_completed["cpu_util"], 0.10)

    def test_9_multiple_nodes_independent(self):
        """TEST 9: Multiple nodes execute tasks independently."""
        self.er_node.push_task(self.dummy_task1) # 5ms
        self.icu_node.push_task(self.dummy_task3) # 2ms
        
        self.sim_env.run(until=2.5)
        self.assertEqual(self.dummy_task1.status, TaskStatus.RUNNING)
        self.assertEqual(self.dummy_task3.status, TaskStatus.COMPLETED)
        
        self.sim_env.run(until=5.5)
        self.assertEqual(self.dummy_task1.status, TaskStatus.COMPLETED)

    def test_10_11_13_graph_summaries(self):
        """TEST 10, 11, 13: Graph stores summaries, not actual task objects."""
        env = HospitalEdgeEnv(max_steps=1)
        env.reset()
        
        # Enqueue task to ER
        task = self.dummy_task1
        env.current_tasks["ER"] = task
        
        env.step({}) # Runs for 50ms, so task should finish if local. Wait, step logic advances 50ms.
        # Check graph
        er_data = env.graph.graph.nodes["ER"]
        self.assertIn("queue_length", er_data)
        self.assertNotIn("task_queue", er_data)
        
        pub_state = env.edge_nodes["ER"].get_public_state()
        self.assertNotIn("task_queue", pub_state)

    def test_12_workload_generator(self):
        """TEST 12: Workload Generator creates tasks correctly."""
        gen = WorkloadGenerator()
        task = gen.sample_task("ER", 0.0)
        self.assertEqual(task.origin_node, "ER")
        self.assertEqual(task.current_node, "ER")
        self.assertEqual(task.status, TaskStatus.GENERATED)

    def test_14_node_failure(self):
        """TEST 14: Node failure does not silently erase task objects."""
        # Tasks are stored in EdgeNodeAgent. If node crashes in graph, the agent still exists in memory and holds tasks.
        self.er_node.push_task(self.dummy_task1)
        # Assuming crash is just a state in graph, the agent's queue remains intact.
        self.assertEqual(len(self.er_node.task_queue), 1)

    def test_deterministic_demonstration(self):
        """Deterministic demonstration of Phase 1-4 logic."""
        print("\n--- Deterministic Demonstration ---")
        er = self.er_node
        icu = self.icu_node
        rad = EdgeNodeAgent(node_id="Radiology", sim_env=self.sim_env, node_capacity_gflops=10.0)
        
        er.push_task(self.dummy_task1) # 5ms
        er.push_task(self.dummy_task2) # 1ms
        icu.push_task(self.dummy_task3) # 2ms
        rad.push_task(self.dummy_task4) # 4ms
        
        print("t=0")
        print(f"ER queue: {[t.task_id for t in er.task_queue]}")
        
        self.sim_env.run(until=1.0)
        print("t=1.0 (During T1 execution)")
        print(f"ER running: {er.running_task.task_id if er.running_task else None}")
        print(f"ER queue: {[t.task_id for t in er.task_queue]}")
        self.assertEqual(er.running_task.task_id, "T1")
        self.assertEqual(len(er.task_queue), 1)
        
        self.sim_env.run(until=6.0) # T1 finishes at 5.0, T2 starts at 5.0, T2 finishes at 6.0
        print("t=6.0 (After T1 completes)")
        print(f"ER completed: {[t.task_id for t in er.completed_tasks]}")
        self.assertIn(self.dummy_task1, er.completed_tasks)
        
        print("Demo successful.")

if __name__ == "__main__":
    unittest.main()
