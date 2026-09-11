import pytest
import numpy as np
from scipy.optimize import minimize_scalar
import CoolProp.CoolProp as CP
from sklearn.ensemble import RandomForestRegressor


def calculate_chiller_power(chw_supply_c, ambient_c, cooling_load_kw, ref="R134a"):
    """Physics model calculating total chiller power."""
    t_evap_k = (chw_supply_c - 2.5) + 273.15
    t_cond_k = (ambient_c + 8.0) + 273.15

    p_evap = CP.PropsSI("P", "T", t_evap_k, "Q", 1, ref) / 1e5
    p_cond = CP.PropsSI("P", "T", t_cond_k, "Q", 0, ref) / 1e5
    pressure_ratio = p_cond / p_evap

    carnot_cop = t_evap_k / (t_cond_k - t_evap_k)
    isentropic_eff = 0.68 - (0.015 * pressure_ratio)
    system_cop = max(carnot_cop * isentropic_eff * 0.75, 1.2)

    compressor_kw = cooling_load_kw / system_cop
    auxiliary_kw = (cooling_load_kw * 0.08) + (0.5 * (chw_supply_c - 5.0))
    return compressor_kw + auxiliary_kw


@pytest.fixture(scope="module")
def optimization_pipeline():
    """Trains the surrogate model and executes the bounded setpoint optimization."""
    np.random.seed(42)
    n_samples = 1000
    t_chw = np.random.uniform(5.0, 12.0, n_samples)
    t_amb = np.random.uniform(20.0, 45.0, n_samples)
    q_load = np.random.uniform(200.0, 1500.0, n_samples)
    y_power = np.array([
        calculate_chiller_power(tc, ta, q) for tc, ta, q in zip(t_chw, t_amb, q_load)
    ])

    surrogate = RandomForestRegressor(n_estimators=40, random_state=42)
    surrogate.fit(np.column_stack([t_chw, t_amb, q_load]), y_power)

    test_ambient = 32.0
    test_cooling_load = 650.0
    baseline_setpoint = 6.0

    def objective_function(t_chw_candidate):
        return surrogate.predict([[t_chw_candidate, test_ambient, test_cooling_load]])[0]

    bounds = (5.0, 11.5)
    opt_result = minimize_scalar(objective_function, bounds=bounds, method="bounded")

    baseline_power = surrogate.predict([[baseline_setpoint, test_ambient, test_cooling_load]])[0]
    power_saved = baseline_power - opt_result.fun

    return {
        "opt_result": opt_result,
        "bounds": bounds,
        "baseline_power": baseline_power,
        "optimal_power": opt_result.fun,
        "optimal_setpoint": opt_result.x,
        "power_saved": power_saved,
    }


def test_optimization_convergence(optimization_pipeline):
    """Verify the bounded scalar optimization converged successfully."""
    opt_result = optimization_pipeline["opt_result"]
    assert opt_result.success is True


def test_optimal_setpoint_within_operational_bounds(optimization_pipeline):
    """Ensure the chosen setpoint respects freeze safety (5.0°C) and dehumidification limit (11.5°C)."""
    setpoint = optimization_pipeline["optimal_setpoint"]
    low, high = optimization_pipeline["bounds"]
    assert low <= setpoint <= high


def test_optimal_setpoint_reduces_power_vs_baseline(optimization_pipeline):
    """Verify that elevated setpoint optimization achieves strictly lower power than 6.0°C baseline."""
    power_saved = optimization_pipeline["power_saved"]
    optimal_setpoint = optimization_pipeline["optimal_setpoint"]

    assert power_saved > 0.0
    assert optimal_setpoint > 6.0