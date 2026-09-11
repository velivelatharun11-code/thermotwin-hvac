import CoolProp.CoolProp as CP


def calculate_chiller_performance(
    chw_supply_c, ambient_c, cooling_load_kw, refrigerant="R134a"
):
    # 1. Heat exchanger approach temperatures
    t_evap_c = chw_supply_c - 2.5
    t_cond_c = ambient_c + 8.0

    t_evap_k = t_evap_c + 273.15
    t_cond_k = t_cond_c + 273.15

    # 2. Saturation pressures (both converted to bar)
    p_evap = CP.PropsSI("P", "T", t_evap_k, "Q", 1, refrigerant) / 1e5
    p_cond = CP.PropsSI("P", "T", t_cond_k, "Q", 0, refrigerant) / 1e5
    pressure_ratio = p_cond / p_evap

    # 3. Thermodynamic COP calculation
    carnot_cop = t_evap_k / (t_cond_k - t_evap_k)
    isentropic_eff = 0.68 - (0.015 * pressure_ratio)
    system_cop = max(carnot_cop * isentropic_eff * 0.75, 1.2)

    # 4. Electrical power consumption
    compressor_kw = cooling_load_kw / system_cop
    auxiliary_kw = (cooling_load_kw * 0.08) + (0.5 * (chw_supply_c - 5.0))
    total_power_kw = compressor_kw + auxiliary_kw

    return {
        "p_evap_bar": p_evap,
        "p_cond_bar": p_cond,
        "cop": system_cop,
        "total_power_kw": total_power_kw,
    }


# ==========================================
# PYTEST TEST SUITES
# ==========================================


def test_chiller_baseline_and_elevated_savings():
    """Verify that elevating chilled water setpoint increases COP and reduces total power."""
    ambient_temp = 35.0
    cooling_demand = 800.0

    baseline = calculate_chiller_performance(6.0, ambient_temp, cooling_demand)
    optimized = calculate_chiller_performance(9.0, ambient_temp, cooling_demand)

    saved_kw = baseline["total_power_kw"] - optimized["total_power_kw"]
    saved_pct = (saved_kw / baseline["total_power_kw"]) * 100

    # Physical sanity checks
    assert baseline["cop"] > 1.5, "Baseline COP should exceed minimum bound"
    assert optimized["cop"] > baseline["cop"], "Elevating CHW setpoint must increase COP"
    assert (
        optimized["p_evap_bar"] > baseline["p_evap_bar"]
    ), "Higher CHW temp should raise evaporator pressure"

    # Energy reduction checks
    assert saved_kw > 0, "Elevated setpoint must yield power savings"
    assert (
        5.0 < saved_pct < 10.0
    ), f"Expected savings between 5-10%, got {saved_pct:.2f}%"


def test_chiller_physical_bounds():
    """Ensure returned pressures and power values fall within realistic operating bounds."""
    res = calculate_chiller_performance(
        chw_supply_c=7.0, ambient_c=32.0, cooling_load_kw=500.0
    )

    assert 2.5 <= res["p_evap_bar"] <= 5.0, "Evaporator pressure outside typical R134a range"
    assert 6.0 <= res["p_cond_bar"] <= 16.0, "Condenser pressure outside typical R134a range"
    assert res["total_power_kw"] > 0, "Power consumption must be positive"


# ==========================================
# STANDALONE EXECUTION
# ==========================================
def test_operational_comparison_baseline_vs_elevated():
    """Verify elevated setpoint (9.0°C) reduces power vs baseline (6.0°C) under rated conditions."""
    ambient_temp = 35.0
    cooling_demand = 800.0

    baseline = calculate_chiller_performance(6.0, ambient_temp, cooling_demand)
    optimized = calculate_chiller_performance(9.0, ambient_temp, cooling_demand)

    saved_kw = baseline["total_power_kw"] - optimized["total_power_kw"]
    saved_pct = (saved_kw / baseline["total_power_kw"]) * 100

    assert saved_kw > 0.0
    assert saved_pct > 0.0
    assert optimized["cop"] > baseline["cop"]