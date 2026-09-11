import CoolProp.CoolProp as CP

def calculate_chiller_performance(chw_supply_c, ambient_c, cooling_load_kw, refrigerant="R134a"):
    # 1. Heat exchanger approach temperatures
    t_evap_c = chw_supply_c - 2.5
    t_cond_c = ambient_c + 8.0
    
    t_evap_k = t_evap_c + 273.15
    t_cond_k = t_cond_c + 273.15
    
    # 2. Saturation pressures (both converted to bar)
    p_evap = CP.PropsSI('P', 'T', t_evap_k, 'Q', 1, refrigerant) / 1e5
    p_cond = CP.PropsSI('P', 'T', t_cond_k, 'Q', 0, refrigerant) / 1e5
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
        "total_power_kw": total_power_kw
    }

# --- COMPARISON TEST ---
ambient_temp = 35.0
cooling_demand = 800.0

baseline = calculate_chiller_performance(6.0, ambient_temp, cooling_demand)
optimized = calculate_chiller_performance(9.0, ambient_temp, cooling_demand)

saved_kw = baseline["total_power_kw"] - optimized["total_power_kw"]
saved_pct = (saved_kw / baseline["total_power_kw"]) * 100

print(f"--- Operational Comparison (Load: {cooling_demand} kW at {ambient_temp}°C Ambient) ---")
print(f"Baseline Setpoint (6.0°C)  : {baseline['total_power_kw']:.2f} kW | COP: {baseline['cop']:.2f}")
print(f"Elevated Setpoint (9.0°C)  : {optimized['total_power_kw']:.2f} kW | COP: {optimized['cop']:.2f}")
print(f"Power Reduction            : {saved_kw:.2f} kW (-{saved_pct:.1f}%)")