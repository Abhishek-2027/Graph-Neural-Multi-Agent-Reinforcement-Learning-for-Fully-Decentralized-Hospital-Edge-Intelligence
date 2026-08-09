"""
Unit tests for Health Severity Index (HSI), Dynamic Priority, and AMRF / FATS Rewards.
"""

import unittest
import numpy as np

from src.utils.hsi_calculator import HSICalculator
from src.utils.amrf_reward import AMRFReward


class TestHSIAndAMRF(unittest.TestCase):

    def setUp(self):
        self.hsi_calc = HSICalculator()
        self.amrf = AMRFReward()

    def test_hsi_normal_patient(self):
        patient = {
            "heart_rate": 75.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "temperature": 37.0,
            "spo2": 99.0,
            "medical_condition": "Routine",
            "admission_type": "Elective",
        }
        hsi = self.hsi_calc.compute_hsi(patient)
        self.assertGreaterEqual(hsi, 0.0)
        self.assertLess(hsi, 0.30)

    def test_hsi_critical_emergency(self):
        patient = {
            "heart_rate": 150.0,  # Severe tachycardia
            "systolic_bp": 190.0, # Hypertensive crisis
            "diastolic_bp": 125.0,
            "temperature": 39.8,  # Severe fever
            "spo2": 84.0,         # Severe hypoxia
            "medical_condition": "Cardiac Arrest",
            "admission_type": "Emergency",
        }
        hsi = self.hsi_calc.compute_hsi(patient)
        self.assertGreaterEqual(hsi, 0.80)

    def test_hitl_override(self):
        patient = {"heart_rate": 75.0, "medical_condition": "Routine"}
        hsi = self.hsi_calc.compute_hsi(patient, hitl_override=True)
        self.assertEqual(hsi, 1.0)

        priority = self.hsi_calc.compute_priority_score(hsi=0.2, deadline_ms=500.0, hitl_override=True)
        self.assertEqual(priority, 100.0)

    def test_fats_gini_fairness(self):
        # Equal queues -> Gini = 0.0 -> FATS = 0.0
        equal_queues = [5.0, 5.0, 5.0, 5.0]
        fats_equal = self.amrf.calculate_fats(equal_queues)
        self.assertAlmostEqual(fats_equal, 0.0, places=3)

        # Extreme inequality -> Gini > 0.5 -> FATS is strongly negative
        unequal_queues = [50.0, 0.0, 0.0, 0.0]
        fats_unequal = self.amrf.calculate_fats(unequal_queues)
        self.assertLess(fats_unequal, fats_equal)

    def test_amrf_emergency_vs_normal_mode(self):
        # Emergency task reward heavily penalizes latency
        em_reward = self.amrf.compute_reward(
            hsi=0.95,
            latency_ms=100.0,
            deadline_ms=50.0,  # Missed
            energy_joules=1.0,
            comm_cost_kb=10.0,
            cpu_util=0.5,
            q_variance=0.8,
        )
        self.assertTrue(em_reward["is_emergency"])
        self.assertLess(em_reward["total_reward"], -10.0)

        # Normal task reward balances utilization and energy
        norm_reward = self.amrf.compute_reward(
            hsi=0.20,
            latency_ms=30.0,
            deadline_ms=200.0,  # Satisfied
            energy_joules=0.2,
            comm_cost_kb=5.0,
            cpu_util=0.70,
            q_variance=0.05,
        )
        self.assertFalse(norm_reward["is_emergency"])
        self.assertGreater(norm_reward["total_reward"], -5.0)


if __name__ == "__main__":
    unittest.main()
