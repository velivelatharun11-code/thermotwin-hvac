import numpy as np
import CoolProp.CoolProp as CP
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Physics Engine function from Phase 2
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

# 2. Generate Synthetic Training Points
print("Generating 1,000 thermodynamic operational points...")
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

# 3. Split into Train & Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_power, test_size=0.2, random_state=42)

# 4. Train the Random Forest Regressor
print("Fitting Random Forest Surrogate Regressor...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- Surrogate Model Performance ---")
print(f"R² Score              : {r2:.4f}")
print(f"Mean Absolute Error   : {mae:.2f} kW")

# Single Test Point Comparison
sample_input = np.array([[7.0, 35.0, 800.0]])
predicted_kw = model.predict(sample_input)[0]
actual_kw = calculate_chiller_power(7.0, 35.0, 800.0)

print(f"\nSingle Point Test (7°C supply, 35°C ambient, 800 kW load):")
print(f"Surrogate Prediction  : {predicted_kw:.2f} kW")
print(f"True Physics Solution : {actual_kw:.2f} kW")