import simpy
import random
import networkx as nx
import uuid

class Task:
    """Represents a medical AI inference task (e.g., ECG, X-Ray)"""
    def __init__(self, env, task_id, source_node, size_mb, compute_cycles, deadline_ms, hsi_score):
        self.env = env
        self.task_id = task_id
        self.source_node = source_node
        self.size_mb = size_mb
        self.compute_cycles = compute_cycles
        self.deadline_ms = deadline_ms
        self.hsi_score = hsi_score
        self.arrival_time = env.now
        
    def __repr__(self):
        return f"Task({self.task_id[:4]}, HSI={self.hsi_score:.2f}, Dead={self.deadline_ms}ms)"

class EdgeNode:
    """Represents a Hospital Department (e.g., ICU, ER) with compute capacity"""
    def __init__(self, env, name, cpu_capacity):
        self.env = env
        self.name = name
        self.cpu_capacity = cpu_capacity
        self.processor = simpy.Resource(env, capacity=cpu_capacity)
        self.task_queue = []
        
    def process_task(self, task):
        """Simulates processing a task on this node"""
        print(f"[{self.env.now:.2f}] {self.name}: Received {task}")
        self.task_queue.append(task)
        
        with self.processor.request() as req:
            yield req
            # Time to compute is roughly cycles / capacity
            process_time = task.compute_cycles / self.cpu_capacity
            yield self.env.timeout(process_time)
            
            self.task_queue.remove(task)
            finish_time = self.env.now
            latency = finish_time - task.arrival_time
            missed = latency > task.deadline_ms
            
            status = "MISSED" if missed else "MET"
            print(f"[{self.env.now:.2f}] {self.name}: Finished {task} | Latency: {latency:.2f}ms [{status}]")

def task_generator(env, node, arrival_rate):
    """Generates random medical tasks for a specific node"""
    while True:
        # Exponential inter-arrival time
        yield env.timeout(random.expovariate(arrival_rate))
        
        task = Task(
            env=env,
            task_id=str(uuid.uuid4()),
            source_node=node.name,
            size_mb=random.uniform(5, 50),
            compute_cycles=random.uniform(100, 1000),
            deadline_ms=random.uniform(50, 300),
            hsi_score=random.uniform(0.1, 1.0)
        )
        # Start processing locally (no offloading yet - this is the baseline!)
        env.process(node.process_task(task))

def build_hospital_graph():
    """Builds the topological graph of the hospital departments"""
    G = nx.Graph()
    
    # Add nodes (Departments)
    departments = ['ICU', 'ER', 'Radiology', 'Ward_A', 'Ward_B', 'Pharmacy', 'Lab']
    for dept in departments:
        G.add_node(dept)
        
    # Add edges (Network links)
    G.add_edges_from([
        ('ER', 'ICU'),
        ('ER', 'Ward_A'),
        ('ICU', 'Radiology'),
        ('ICU', 'Ward_A'),
        ('Ward_A', 'Ward_B'),
        ('Ward_B', 'Pharmacy'),
        ('Ward_A', 'Lab')
    ])
    
    return G

def run_simulation():
    print("Initializing GraphMARL Hospital Simulator...")
    env = simpy.Environment()
    
    # 1. Build Network Topology
    hospital_graph = build_hospital_graph()
    
    # 2. Initialize Edge Nodes
    nodes = {}
    nodes['ER'] = EdgeNode(env, 'ER', cpu_capacity=50)        # ER is fast
    nodes['ICU'] = EdgeNode(env, 'ICU', cpu_capacity=40)
    nodes['Radiology'] = EdgeNode(env, 'Radiology', cpu_capacity=100) # Big GPU
    nodes['Ward_A'] = EdgeNode(env, 'Ward_A', cpu_capacity=10) # Low power
    
    # 3. Start Task Generators for each department
    # ER gets tasks very quickly (arrival_rate = 1.5 tasks/ms)
    env.process(task_generator(env, nodes['ER'], arrival_rate=1.5))
    
    # Ward_A gets tasks slowly (arrival_rate = 0.2 tasks/ms)
    env.process(task_generator(env, nodes['Ward_A'], arrival_rate=0.2))
    
    # 4. Run the simulation for 100 milliseconds
    print("Starting Simulation...")
    env.run(until=100)
    print("Simulation Complete.")

if __name__ == "__main__":
    run_simulation()
