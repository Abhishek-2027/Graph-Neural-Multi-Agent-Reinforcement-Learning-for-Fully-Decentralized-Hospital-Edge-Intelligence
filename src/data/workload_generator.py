"""
Hybrid Task Characterization Framework (HTCF) and Medical Workload Generator for GraphMARL.

Processes real healthcare and IoT sensor datasets to generate realistic, prioritized
medical task streams for decentralized hospital edge simulation.
"""

from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from enum import Enum
import os
import pandas as pd
import numpy as np

from src.utils.hsi_calculator import HSICalculator


class TaskType(Enum):
    EMERGENCY_TRIAGE = "Emergency_Triage"
    REALTIME_ECG_ARRHYTHMIA = "Realtime_ECG_Arrhythmia"
    MEDICAL_IMAGING_INFERENCE = "Medical_Imaging_Inference"
    ROUTINE_VITALS_LOGGING = "Routine_Vitals_Logging"


class TaskStatus(Enum):
    GENERATED = "Generated"
    ASSIGNED = "Assigned"
    WAITING = "Waiting"
    RUNNING = "Running"
    COMPLETED = "Completed"
    ORPHANED = "Orphaned"
    RECOVERY_PENDING = "Recovery_Pending"


@dataclass
class MedicalTask:
    task_id: str
    patient_id: str
    origin_node: str
    task_type: TaskType
    arrival_time_ms: float
    data_size_mb: float
    required_gflops: float
    deadline_ms: float
    hsi: float
    priority_score: float
    hitl_override: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Persistent lifecycle state (Phases 1-4)
    current_node: str = ""
    status: TaskStatus = TaskStatus.GENERATED
    start_time_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None
    remaining_compute_gflops: float = 0.0
    total_compute_gflops: float = 0.0

    def __post_init__(self):
        if not self.current_node:
            self.current_node = self.origin_node
        if self.total_compute_gflops == 0.0:
            self.total_compute_gflops = self.required_gflops
        if self.remaining_compute_gflops == 0.0:
            self.remaining_compute_gflops = self.required_gflops

    @property
    def is_emergency(self) -> bool:
        return self.hsi >= 0.80 or self.hitl_override


class WorkloadGenerator:
    """
    Loads real patient and IoT datasets, extracts clinical features,
    and produces streaming task workloads across hospital edge departments.
    """

    def __init__(
        self,
        healthcare_csv_path: Optional[str] = None,
        iot_csv_path: Optional[str] = None,
        hsi_calc: Optional[HSICalculator] = None,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.hsi_calc = hsi_calc or HSICalculator()
        self.tasks_generated = 0

        # Resolve dataset paths relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.healthcare_csv = healthcare_csv_path or os.path.join(
            base_dir, "data", "healthcare_dataset", "healthcare_dataset.csv"
        )
        self.iot_csv = iot_csv_path or os.path.join(
            base_dir, "data", "healthcare_iot_dataset", "healthcare_iot_target_dataset_5000.csv"
        )

        self.patient_records = self._load_patient_records()
        self.iot_records = self._load_iot_records()

    def _load_patient_records(self) -> List[Dict[str, Any]]:
        records = []
        if os.path.exists(self.healthcare_csv):
            try:
                df = pd.read_csv(self.healthcare_csv)
                # Sample up to 10,000 for fast in-memory access
                df_sample = df.sample(min(len(df), 10000), random_state=42)
                for _, row in df_sample.iterrows():
                    records.append(
                        {
                            "name": str(row.get("Name", "Unknown")),
                            "age": float(row.get("Age", 50)),
                            "gender": str(row.get("Gender", "Male")),
                            "medical_condition": str(row.get("Medical Condition", "Routine")),
                            "admission_type": str(row.get("Admission Type", "Elective")),
                            "test_results": str(row.get("Test Results", "Normal")),
                        }
                    )
            except Exception as e:
                print(f"[WorkloadGenerator] Warning loading healthcare CSV: {e}")

        if not records:
            # Fallback synthetic patient pool
            records = [
                {"age": 65, "gender": "Male", "medical_condition": "Cardiac Arrest", "admission_type": "Emergency"},
                {"age": 45, "gender": "Female", "medical_condition": "Cancer", "admission_type": "Urgent"},
                {"age": 30, "gender": "Male", "medical_condition": "Asthma", "admission_type": "Emergency"},
                {"age": 72, "gender": "Female", "medical_condition": "Hypertension", "admission_type": "Elective"},
            ]
        return records

    def _load_iot_records(self) -> List[Dict[str, Any]]:
        records = []
        if os.path.exists(self.iot_csv):
            try:
                df = pd.read_csv(self.iot_csv)
                for _, row in df.iterrows():
                    records.append(
                        {
                            "patient_id": str(row.get("Patient_ID", self.rng.randint(1000, 9999))),
                            "sensor_type": str(row.get("Sensor_Type", "MultiVital")),
                            "temperature": float(row.get("Temperature (°C)", 37.0)),
                            "systolic_bp": float(row.get("Systolic_BP (mmHg)", 120.0)),
                            "diastolic_bp": float(row.get("Diastolic_BP (mmHg)", 80.0)),
                            "heart_rate": float(row.get("Heart_Rate (bpm)", 75.0)),
                            "battery_level": float(row.get("Battery_Level (%)", 90.0)),
                            "health_status": str(row.get("Target_Health_Status", "Healthy")),
                        }
                    )
            except Exception as e:
                print(f"[WorkloadGenerator] Warning loading IoT CSV: {e}")

        if not records:
            # Fallback synthetic IoT readings
            records = [
                {"heart_rate": 135.0, "systolic_bp": 175.0, "diastolic_bp": 110.0, "temperature": 39.2, "spo2": 89.0},
                {"heart_rate": 78.0, "systolic_bp": 122.0, "diastolic_bp": 82.0, "temperature": 37.1, "spo2": 98.0},
                {"heart_rate": 52.0, "systolic_bp": 92.0, "diastolic_bp": 58.0, "temperature": 36.2, "spo2": 94.0},
            ]
        return records

    def sample_task(
        self,
        node_id: str,
        arrival_time_ms: float,
        task_type_override: Optional[TaskType] = None,
        force_emergency: bool = False,
    ) -> MedicalTask:
        """
        Creates a single realistic medical task synthesizing patient condition and IoT vitals.
        """
        p_rec = self.patient_records[self.rng.randint(0, len(self.patient_records))]
        iot_rec = self.iot_records[self.rng.randint(0, len(self.iot_records))]

        combined = {**p_rec, **iot_rec}
        if force_emergency:
            combined["hitl_override"] = True
            combined["admission_type"] = "Emergency"
            combined["heart_rate"] = 145.0
            combined["spo2"] = 86.0

        hsi = self.hsi_calc.compute_hsi(combined, hitl_override=force_emergency)

        if task_type_override is not None:
            task_type = task_type_override
        else:
            # Determine task type based on HSI and node context
            if hsi >= 0.80 or force_emergency:
                task_type = TaskType.EMERGENCY_TRIAGE
            elif hsi >= 0.50:
                task_type = self.rng.choice(
                    [TaskType.REALTIME_ECG_ARRHYTHMIA, TaskType.EMERGENCY_TRIAGE],
                    p=[0.7, 0.3],
                )
            elif hsi >= 0.30:
                task_type = self.rng.choice(
                    [TaskType.MEDICAL_IMAGING_INFERENCE, TaskType.REALTIME_ECG_ARRHYTHMIA],
                    p=[0.6, 0.4],
                )
            else:
                task_type = self.rng.choice(
                    [TaskType.ROUTINE_VITALS_LOGGING, TaskType.MEDICAL_IMAGING_INFERENCE],
                    p=[0.7, 0.3],
                )

        # Assign compute and bandwidth parameters based on task category
        if task_type == TaskType.EMERGENCY_TRIAGE:
            data_size = float(self.rng.uniform(0.1, 0.5))  # MB
            gflops = float(self.rng.uniform(0.5, 2.0))     # GFLOPs
            deadline = float(self.rng.uniform(25.0, 50.0)) # ms
        elif task_type == TaskType.REALTIME_ECG_ARRHYTHMIA:
            data_size = float(self.rng.uniform(1.0, 3.5))
            gflops = float(self.rng.uniform(2.5, 6.0))
            deadline = float(self.rng.uniform(60.0, 120.0))
        elif task_type == TaskType.MEDICAL_IMAGING_INFERENCE:
            data_size = float(self.rng.uniform(10.0, 35.0))
            gflops = float(self.rng.uniform(8.0, 22.0))
            deadline = float(self.rng.uniform(200.0, 400.0))
        else:  # ROUTINE_VITALS_LOGGING
            data_size = float(self.rng.uniform(0.05, 0.2))
            gflops = float(self.rng.uniform(0.2, 1.0))
            deadline = float(self.rng.uniform(400.0, 800.0))

        priority = self.hsi_calc.compute_priority_score(
            hsi=hsi,
            deadline_ms=deadline,
            time_elapsed_ms=0.0,
            data_size_mb=data_size,
            hitl_override=force_emergency,
        )

        task_id = f"task_{node_id}_{int(arrival_time_ms)}_{self.rng.randint(1000, 9999)}"
        patient_id = str(iot_rec.get("patient_id", f"P_{self.rng.randint(100, 999)}"))

        task = MedicalTask(
            task_id=task_id,
            patient_id=patient_id,
            origin_node=node_id,
            task_type=task_type,
            arrival_time_ms=arrival_time_ms,
            data_size_mb=data_size,
            required_gflops=gflops,
            deadline_ms=deadline,
            hsi=hsi,
            priority_score=priority,
            hitl_override=force_emergency,
            metadata=combined,
        )
        self.tasks_generated += 1
        return task

    def generate_poisson_stream(
        self,
        node_ids: List[str],
        duration_ms: float = 10000.0,
        base_rate_per_sec: float = 20.0,
        burst_prob: float = 0.05,
        burst_multiplier: float = 4.0,
    ) -> List[MedicalTask]:
        """
        Generates a chronological list of tasks following a Poisson arrival process
        with optional burstiness to simulate mass-casualty emergency spikes.
        """
        all_tasks = []
        for node in node_ids:
            curr_t = 0.0
            while curr_t < duration_ms:
                # Check for burst surge
                is_burst = self.rng.rand() < burst_prob
                rate = base_rate_per_sec * (burst_multiplier if is_burst else 1.0)
                # Inter-arrival time from exponential distribution (converted to ms)
                inter_arrival = self.rng.exponential(1000.0 / rate)
                curr_t += inter_arrival
                if curr_t >= duration_ms:
                    break

                force_emergency = is_burst and (self.rng.rand() < 0.4)
                task = self.sample_task(
                    node_id=node,
                    arrival_time_ms=curr_t,
                    force_emergency=force_emergency,
                )
                all_tasks.append(task)

        # Sort chronologically
        all_tasks.sort(key=lambda t: t.arrival_time_ms)
        return all_tasks
