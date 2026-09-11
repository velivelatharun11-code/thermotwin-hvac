import pytest
import numpy as np
import CoolProp.CoolProp as CP
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def calculate_chiller_power(chw_supply_c, ambient_c, cooling_load_kw, ref="R134a"):
    """Thermodynamic power calculation based on Carnot COP and isentropic efficiency."""
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
def trained_surrogate_model():
    """Trains a Random Forest surrogate regressor on 1,000 synthetic thermodynamic points."""
    np.random.seed(42)
    n_samples = 1000

    t_chw = np.random.uniform(5.0, 12.0, n_samples)
    t_amb = np.random.uniform(20.0, 45.0, n_samples)
    q_load = np.random.uniform(200.0, 1500.0, n_samples)

    y_power = np.array([
        calculate_chiller_power(tc, ta, q)
        for tc, ta, q in zip(t_chw, t_amb, q_load)
    ])

    X = np.column_stack([t_chw, t_amb, q_load])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_power, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_surrogate_model_r2_score(trained_surrogate_model):
    """Ensure surrogate model fits with high fidelity (R² >= 0.98)."""
    model = trained_surrogate_model["model"]
    X_test = trained_surrogate_model["X_test"]
    y_test = trained_surrogate_model["y_test"]

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 >= 0.98


def test_surrogate_model_mae_tolerance(trained_surrogate_model):
    """Ensure mean absolute error stays within acceptable bounds for HVAC power (< 10 kW)."""
    model = trained_surrogate_model["model"]
    X_test = trained_surrogate_model["X_test"]
    y_test = trained_surrogate_model["y_test"]

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    assert mae < 10.0


def test_single_point_physics_parity(trained_surrogate_model):
    """Check prediction error at design rating point (7°C supply, 35°C ambient, 800 kW)."""
    model = trained_surrogate_model["model"]
    sample_input = np.array([[7.0, 35.0, 800.0]])

    predicted_kw = model.predict(sample_input)[0]
    actual_kw = calculate_chiller_power(7.0, 35.0, 800.0)

    # Allow up to 5% relative tolerance between surrogate and numerical physics model
    assert predicted_kw == pytest.approx(actual_kw, rel=0.05)