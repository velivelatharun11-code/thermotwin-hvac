import numpy as np
from scipy.optimize import minimize_scalar
import CoolProp.CoolProp as CP
from sklearn.ensemble import RandomForestRegressor

# 1. Physics Engine
def calculate_chiller_power(chw_supply_c, ambient_c, cooling_load_kw, ref="R134a"):
    t_evap_k = (chw_supply_c - 2.5) + 273.15
    t_cond_k = (ambient_c + 8.0) + 273.15
    
    p_evap = CP.PropsSI('P', 'T', t_evap_k, 'Q', 1, ref) / 1e5
    p_cond = CP.PropsSI('P', 'T', t_cond_k, 'Q', 0, ref) / 1e5
    pressure_ratio = p_cond / p_evap
    
    carnot_cop = t_evap_k / (t_cond_k - t_evap_k)
    isentropic_eff = 0.68 - (0.015 * pressure_ratio)
    system_cop = max(carnot_cop * isentropic_eff * 0.75, 1.2)
    
    compressor_kw = cooling_load_kw / system_cop
    auxiliary_kw = (cooling_load_kw * 0.08) + (0.5 * (chw_supply_c - 5.0))
    return compressor_kw + auxiliary_kw

# 2. Train Surrogate Model
print("Training surrogate model...")
np.random.seed(42)
n_samples = 1000
t_chw = np.random.uniform(5.0, 12.0, n_samples)
t_amb = np.random.uniform(20.0, 45.0, n_samples)
q_load = np.random.uniform(200.0, 1500.0, n_samples)
y_power = np.array([calculate_chiller_power(tc, ta, q) for tc, ta, q in zip(t_chw, t_amb, q_load)])

surrogate = RandomForestRegressor(n_estimators=40, random_state=42)
surrogate.fit(np.column_stack([t_chw, t_amb, q_load]), y_power)

# 3. Define the Operational Scenario
test_ambient = 32.0       # 32°C outside
test_cooling_load = 650.0 # 650 kW building load
baseline_setpoint = 6.0   # Standard industry fixed setpoint (6.0°C)

# 4. Optimization Routine
# We search for the optimal Chilled Water Setpoint (t_chw) that minimizes power draw
def objective_function(t_chw_candidate):
    prediction = surrogate.predict([[t_chw_candidate, test_ambient, test_cooling_load]])[0]
    return prediction

# Bounds: 5.0°C (freeze risk margin) to 11.5°C (dehumidification upper limit)
opt_result = minimize_scalar(objective_function, bounds=(5.0, 11.5), method='bounded')

optimal_setpoint = opt_result.x
optimal_power = opt_result.fun
baseline_power = surrogate.predict([[baseline_setpoint, test_ambient, test_cooling_load]])[0]

power_saved = baseline_power - optimal_power
savings_pct = (power_saved / baseline_power) * 100

# Estimated 24-hour financial impact assuming $0.12 per kWh
daily_kwh_saved = power_saved * 24
daily_cost_saved = daily_kwh_saved * 0.12

print("\n--- Optimization Results ---")
print(f"Operational Load        : {test_cooling_load} kW at {test_ambient}°C Ambient")
print(f"Standard Baseline Temp  : {baseline_setpoint:.1f} °C -> Power: {baseline_power:.2f} kW")
print(f"Optimal Reset Setpoint  : {optimal_setpoint:.2f} °C -> Power: {optimal_power:.2f} kW")
print(f"Continuous Power Saved  : {power_saved:.2f} kW (-{savings_pct:.2f}%)")
print(f"Estimated Cost Savings  : ${daily_cost_saved:.2f} / day")