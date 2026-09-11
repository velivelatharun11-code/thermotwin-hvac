import CoolProp.CoolProp as CP

refrigerant = "R134a"

t_evap_c = 4.0
t_cond_c = 40.0

t_evap_k = t_evap_c + 273.15
t_cond_k = t_cond_c + 273.15

p_evap_bar = CP.PropsSI('P', 'T', t_evap_k, 'Q', 1, refrigerant) / 1e5
p_cond_bar = CP.PropsSI('P', 'T', t_cond_k, 'Q', 0, refrigerant) / 1e5

compression_ratio = p_cond_bar / p_evap_bar

print(f"--- Thermodynamic Verification ({refrigerant}) ---")
print(f"Evaporator Pressure ({t_evap_c}°C) : {p_evap_bar:.2f} bar")
print(f"Condenser Pressure ({t_cond_c}°C)  : {p_cond_bar:.2f} bar")
print(f"Compression Ratio                  : {compression_ratio:.2f}")