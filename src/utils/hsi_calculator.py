"""
Health Severity Index (HSI) and Dynamic Task Prioritization Calculator for GraphMARL.

Computes clinical anomaly risk scores across multi-modal vital signs (ECG/QTc,
Heart Rate, Blood Pressure, Temperature, SpO2) and combines them with task
deadline sensitivity and Human-in-the-Loop (HITL) emergency overrides.
"""

from typing import Dict, Any, Optional
import math
import numpy as np


class HSICalculator:
    """
    Evaluates clinical severity risk (HSI in [0.0, 1.0]) and dynamic task priority.
    """

    def __init__(
        self,
        weight_hr: float = 0.20,
        weight_bp: float = 0.20,
        weight_temp: float = 0.15,
        weight_spo2: float = 0.25,
        weight_qtc: float = 0.10,
        weight_condition: float = 0.10,
    ):
        total_w = weight_hr + weight_bp + weight_temp + weight_spo2 + weight_qtc + weight_condition
        self.w_hr = weight_hr / total_w
        self.w_bp = weight_bp / total_w
        self.w_temp = weight_temp / total_w
        self.w_spo2 = weight_spo2 / total_w
        self.w_qtc = weight_qtc / total_w
        self.w_cond = weight_condition / total_w

    def score_heart_rate(self, hr: float) -> float:
        """
        Scores heart rate risk (bpm).
        Normal: 60 - 100 bpm.
        Severe: < 45 or > 135 bpm.
        """
        if hr is None or math.isnan(hr):
            return 0.2  # Neutral baseline risk for missing
        if 60.0 <= hr <= 100.0:
            return 0.0
        elif hr < 60.0:
            # Bradycardia
            return float(np.clip((60.0 - hr) / 25.0, 0.0, 1.0))
        else:
            # Tachycardia
            return float(np.clip((hr - 100.0) / 40.0, 0.0, 1.0))

    def score_blood_pressure(self, sbp: float, dbp: float) -> float:
        """
        Scores blood pressure risk (mmHg).
        Normal: SBP 90-120, DBP 60-80.
        Hypertensive Crisis: SBP > 180 or DBP > 120.
        Hypotensive Crisis: SBP < 90 or DBP < 60.
        """
        if sbp is None or math.isnan(sbp):
            sbp = 120.0
        if dbp is None or math.isnan(dbp):
            dbp = 80.0

        sbp_risk = 0.0
        if sbp < 90.0:
            sbp_risk = (90.0 - sbp) / 30.0
        elif sbp > 120.0:
            sbp_risk = (sbp - 120.0) / 60.0

        dbp_risk = 0.0
        if dbp < 60.0:
            dbp_risk = (60.0 - dbp) / 20.0
        elif dbp > 80.0:
            dbp_risk = (dbp - 80.0) / 40.0

        return float(np.clip(max(sbp_risk, dbp_risk), 0.0, 1.0))

    def score_temperature(self, temp_c: float) -> float:
        """
        Scores core body temperature risk (°C).
        Normal: 36.5 - 37.5°C.
        Hypothermia: < 35.0°C.
        Hyperpyrexia / Severe Fever: > 39.5°C.
        """
        if temp_c is None or math.isnan(temp_c):
            return 0.1
        if 36.5 <= temp_c <= 37.5:
            return 0.0
        elif temp_c < 36.5:
            return float(np.clip((36.5 - temp_c) / 2.5, 0.0, 1.0))
        else:
            return float(np.clip((temp_c - 37.5) / 2.5, 0.0, 1.0))

    def score_spo2(self, spo2: float) -> float:
        """
        Scores oxygen saturation risk (%).
        Normal: 95 - 100%.
        Mild Hypoxia: 90 - 94%.
        Severe Hypoxia: < 88%.
        """
        if spo2 is None or math.isnan(spo2):
            return 0.2
        if spo2 >= 95.0:
            return 0.0
        elif spo2 >= 90.0:
            return float((95.0 - spo2) / 10.0)
        else:
            return float(np.clip(0.5 + (90.0 - spo2) / 15.0, 0.0, 1.0))

    def score_qtc(self, qt_ms: Optional[float], rr_s: Optional[float], gender: str = "Male") -> float:
        """
        Calculates Bazett-corrected QTc interval risk: QTc = QT / sqrt(RR).
        Prolongation: > 450ms (Male), > 470ms (Female).
        Critical: > 500ms (high arrhythmia / TdP risk).
        """
        if qt_ms is None or rr_s is None or rr_s <= 0 or math.isnan(qt_ms) or math.isnan(rr_s):
            return 0.0
        qtc = qt_ms / math.sqrt(rr_s)
        threshold = 450.0 if gender.lower() == "male" else 470.0
        if qtc <= threshold:
            return 0.0
        return float(np.clip((qtc - threshold) / (520.0 - threshold), 0.0, 1.0))

    def score_clinical_condition(self, condition: str, admission_type: str) -> float:
        """
        Scores diagnostic condition and admission urgency.
        """
        cond_map = {
            "cardiac arrest": 1.0,
            "sepsis": 0.9,
            "trauma": 0.85,
            "stroke": 0.85,
            "asthma": 0.5,
            "cancer": 0.4,
            "diabetes": 0.3,
            "hypertension": 0.25,
            "obesity": 0.15,
            "routine": 0.05,
        }
        adm_map = {
            "emergency": 0.9,
            "urgent": 0.6,
            "elective": 0.1,
        }
        c_score = cond_map.get(str(condition).strip().lower(), 0.2)
        a_score = adm_map.get(str(admission_type).strip().lower(), 0.2)
        return float(np.clip(0.6 * c_score + 0.4 * a_score, 0.0, 1.0))

    def compute_hsi(self, patient_data: Dict[str, Any], hitl_override: bool = False) -> float:
        """
        Computes the unified Health Severity Index (HSI in [0.0, 1.0]).
        If HITL override is triggered by clinician, returns 1.0 (maximum emergency priority).
        """
        if hitl_override or patient_data.get("hitl_override", False):
            return 1.0

        f_hr = self.score_heart_rate(patient_data.get("heart_rate", patient_data.get("Heart_Rate (bpm)", 75.0)))
        f_bp = self.score_blood_pressure(
            patient_data.get("systolic_bp", patient_data.get("Systolic_BP (mmHg)", 120.0)),
            patient_data.get("diastolic_bp", patient_data.get("Diastolic_BP (mmHg)", 80.0)),
        )
        f_temp = self.score_temperature(patient_data.get("temperature", patient_data.get("Temperature (°C)", 37.0)))
        f_spo2 = self.score_spo2(patient_data.get("spo2", 98.0))
        f_qtc = self.score_qtc(
            patient_data.get("qt_ms", None),
            patient_data.get("rr_s", None),
            patient_data.get("gender", patient_data.get("Gender", "Male")),
        )
        f_cond = self.score_clinical_condition(
            patient_data.get("medical_condition", patient_data.get("Medical Condition", "routine")),
            patient_data.get("admission_type", patient_data.get("Admission Type", "elective")),
        )

        hsi = (
            self.w_hr * f_hr
            + self.w_bp * f_bp
            + self.w_temp * f_temp
            + self.w_spo2 * f_spo2
            + self.w_qtc * f_qtc
            + self.w_cond * f_cond
        )
        return float(np.clip(hsi, 0.0, 1.0))

    def compute_priority_score(
        self,
        hsi: float,
        deadline_ms: float,
        time_elapsed_ms: float = 0.0,
        data_size_mb: float = 1.0,
        hitl_override: bool = False,
    ) -> float:
        """
        Computes dynamic composite priority for queue scheduling and offloading.
        Priority = 0.5 * HSI + 0.3 * (1 / remaining_time) + 0.1 * data_urgency + 0.1 * hitl
        """
        if hitl_override:
            return 100.0  # Top of all queues

        rem_time = max(deadline_ms - time_elapsed_ms, 1.0)
        time_urgency = min(1000.0 / rem_time, 1.0)  # normalized urgency
        size_factor = min(data_size_mb / 50.0, 1.0)

        priority = 0.55 * hsi + 0.35 * time_urgency + 0.10 * size_factor
        return float(priority)

    def calculate_measurement_entropy(self, vital_variances: list) -> float:
        """
        Computes measurement uncertainty entropy u_i = -sum(p_k * log2(p_k)).
        Higher entropy denotes noisy/unreliable IoT sensor readings.
        """
        vars_arr = np.array(vital_variances, dtype=np.float32)
        if len(vars_arr) == 0 or np.sum(vars_arr) <= 1e-8:
            return 0.0
        probs = vars_arr / (np.sum(vars_arr) + 1e-8)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)
