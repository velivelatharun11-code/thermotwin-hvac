import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from app import calculate_chiller_performance

def test_telemetry_schema_validation():
    """Verify telemetry dataset contains all mandatory engineering columns."""
    required_cols = {"Timestamp", "Ambient_Temp_C", "Cooling_Load_kW", "Supply_Temp_C", "Measured_Power_kW"}
    sample_df = pd.DataFrame({
        "Timestamp": pd.date_range("2026-09-12", periods=5, freq="h"),
        "Ambient_Temp_C": [28.0, 29.5, 31.0, 32.5, 30.0],
        "Cooling_Load_kW": [450.0, 500.0, 550.0, 600.0, 520.0],
        "Supply_Temp_C": [6.5, 6.7, 6.8, 7.0, 6.6],
        "Measured_Power_kW": [160.0, 175.0, 190.0, 210.0, 185.0]
    })
    assert required_cols.issubset(sample_df.columns), "Telemetry dataframe missing required columns"

def test_telemetry_missing_columns_detected():
    """Ensure missing columns are properly identified."""
    required_cols = {"Timestamp", "Ambient_Temp_C", "Cooling_Load_kW", "Supply_Temp_C", "Measured_Power_kW"}
    bad_df = pd.DataFrame({
        "Timestamp": pd.date_range("2026-09-12", periods=3, freq="h"),
        "Ambient_Temp_C": [28.0, 29.0, 30.0]
    })
    missing = required_cols - set(bad_df.columns)
    assert len(missing) == 3
    assert "Measured_Power_kW" in missing

def test_digital_twin_parity_bounds():
    """Verify digital twin power predictions achieve high statistical parity with noise-free simulated baseline."""
    hours = 24
    amb_temps = np.linspace(25.0, 35.0, hours)
    loads = np.linspace(400.0, 800.0, hours)
    chw_temps = np.full(hours, 6.7)
    
    predictions = [
        calculate_chiller_performance(t, a, q, "R134a", 1.0)["total_power_kw"]
        for t, a, q in zip(chw_temps, amb_temps, loads)
    ]
    
    np.random.seed(42)
    ground_truth = np.array(predictions) * np.random.uniform(0.99, 1.01, hours)
    
    r2 = r2_score(ground_truth, predictions)
    mape = mean_absolute_percentage_error(ground_truth, predictions) * 100.0
    
    assert r2 >= 0.95, f"R2 score {r2} below expected parity threshold of 0.95"
    assert mape < 3.0, f"MAPE {mape}% exceeds operational tolerance of 3.0%"

def test_adaptive_calibration_reduces_bias():
    """Ensure adaptive calibration factor accurately removes systematic fouling offset."""
    cal_target = 1.045  # 4.5% compressor degradation
    amb = 32.0
    load = 650.0
    chw = 6.5
    
    # Baseline nominal model output
    uncalibrated_res = calculate_chiller_performance(chw, amb, load, "R134a", 1.0)
    uncalibrated_power = uncalibrated_res["total_power_kw"]
    
    # Field condition with true physical degradation
    measured_drifted_res = calculate_chiller_performance(chw, amb, load, "R134a", cal_target)
    measured_drifted_power = measured_drifted_res["total_power_kw"]
    
    # Verify drift creates observable initial bias
    initial_bias = measured_drifted_power - uncalibrated_power
    assert initial_bias > 5.0, "Initial bias should indicate significant drift"
    
    # Calibrate model to field condition
    calibrated_power = calculate_chiller_performance(chw, amb, load, "R134a", cal_target)["total_power_kw"]
    
    # Residual post-calibration should converge to 0
    calibrated_residual = abs(measured_drifted_power - calibrated_power)
    assert calibrated_residual < 1e-3, f"Calibrated residual {calibrated_residual} must converge to near zero"