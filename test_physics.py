import pytest
import CoolProp.CoolProp as CP

REFRIGERANT = "R134a"

def test_evaporator_saturation_pressure():
    """Verify evaporator pressure at 4.0 deg C matches standard R134a saturation tables (~3.38 bar)."""
    t_evap_k = 4.0 + 273.15
    p_evap_bar = CP.PropsSI("P", "T", t_evap_k, "Q", 1, REFRIGERANT) / 1e5
    assert p_evap_bar == pytest.approx(3.376, rel=1e-2)

def test_condenser_saturation_pressure():
    """Verify condenser pressure at 40.0 deg C matches standard R134a saturation tables (~10.17 bar)."""
    t_cond_k = 40.0 + 273.15
    p_cond_bar = CP.PropsSI("P", "T", t_cond_k, "Q", 0, REFRIGERANT) / 1e5
    assert p_cond_bar == pytest.approx(10.166, rel=1e-2)

def test_compression_ratio_bounds():
    """Verify compression ratio for 4 deg C evap and 40 deg C cond falls within typical chiller design bounds (2.5 - 3.5)."""
    t_evap_k = 4.0 + 273.15
    t_cond_k = 40.0 + 273.15

    p_evap_bar = CP.PropsSI("P", "T", t_evap_k, "Q", 1, REFRIGERANT) / 1e5
    p_cond_bar = CP.PropsSI("P", "T", t_cond_k, "Q", 0, REFRIGERANT) / 1e5
    compression_ratio = p_cond_bar / p_evap_bar

    assert p_cond_bar > p_evap_bar
    assert 2.5 <= compression_ratio <= 3.5
    assert compression_ratio == pytest.approx(3.01, abs=0.05)