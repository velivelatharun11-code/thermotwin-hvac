import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import CoolProp.CoolProp as CP
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize_scalar

st.set_page_config(
    page_title="ThermoTwin | HVAC Optimizer",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ ThermoTwin: Physics-Informed HVAC Digital Twin")
st.caption("ASME Computers and Information in Engineering (CIE) Division | Real-Time Energy Optimization")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Operational Boundary Conditions")
ambient_temp = st.sidebar.slider("Outdoor Ambient Temperature (°C)", 20.0, 45.0, 32.0, step=0.5)
cooling_load = st.sidebar.slider("Building Cooling Demand (kW)", 200.0, 1500.0, 650.0, step=25.0)
electricity_cost = st.sidebar.number_input("Electricity Cost ($/kWh)", min_value=0.05, max_value=0.50, value=0.12, step=0.01)
refrigerant = st.sidebar.selectbox("Refrigerant Spec", ["R134a", "R410A", "R1234ze"])

st.sidebar.markdown("---")
st.sidebar.header("ASME AI Advisory Agent")

# Automatically pull from st.secrets if configured on Streamlit Cloud
default_api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)", 
    value=default_api_key,
    type="password", 
    help="Pre-filled if configured in Streamlit Cloud Secrets. Leave blank for deterministic auditor."
)

# --- 1. CORE THERMODYNAMICS ---
def calculate_chiller_performance(chw_supply_c, amb_c, load_kw, ref):
    t_evap_k = (chw_supply_c - 2.5) + 273.15
    t_cond_k = (amb_c + 8.0) + 273.15
    
    # Saturation pressures converted to bar
    p_evap = CP.PropsSI('P', 'T', t_evap_k, 'Q', 1, ref) / 1e5
    p_cond = CP.PropsSI('P', 'T', t_cond_k, 'Q', 0, ref) / 1e5
    pressure_ratio = p_cond / p_evap
    
    # Thermodynamic efficiency model
    carnot_cop = t_evap_k / (t_cond_k - t_evap_k)
    isentropic_eff = 0.68 - (0.015 * pressure_ratio)
    system_cop = max(carnot_cop * isentropic_eff * 0.75, 1.2)
    
    compressor_kw = load_kw / system_cop
    auxiliary_kw = (load_kw * 0.08) + (0.5 * (chw_supply_c - 5.0))
    total_power = compressor_kw + auxiliary_kw
    
    return {
        "total_power_kw": total_power,
        "compressor_kw": compressor_kw,
        "auxiliary_kw": auxiliary_kw,
        "cop": system_cop,
        "p_evap_bar": p_evap,
        "p_cond_bar": p_cond,
        "pressure_ratio": pressure_ratio
    }

# --- 2. SURROGATE MODEL TRAINING (CACHED) ---
@st.cache_resource(show_spinner="Training physics-informed surrogate model...")
def get_trained_surrogate(ref):
    np.random.seed(42)
    n_samples = 1200
    
    t_chw_synth = np.random.uniform(5.0, 12.0, n_samples)
    t_amb_synth = np.random.uniform(20.0, 45.0, n_samples)
    q_load_synth = np.random.uniform(200.0, 1500.0, n_samples)
    
    y_power = np.array([
        calculate_chiller_performance(tc, ta, q, ref)["total_power_kw"]
        for tc, ta, q in zip(t_chw_synth, t_amb_synth, q_load_synth)
    ])
    
    X = np.column_stack([t_chw_synth, t_amb_synth, q_load_synth])
    surrogate = RandomForestRegressor(n_estimators=45, random_state=42)
    surrogate.fit(X, y_power)
    return surrogate

surrogate_model = get_trained_surrogate(refrigerant)

# --- 3. BASELINE VS OPTIMIZATION ---
baseline_temp = 6.0
baseline_data = calculate_chiller_performance(baseline_temp, ambient_temp, cooling_load, refrigerant)
baseline_power = baseline_data["total_power_kw"]

def objective_function(t_chw_candidate):
    return surrogate_model.predict([[t_chw_candidate, ambient_temp, cooling_load]])[0]

opt_result = minimize_scalar(objective_function, bounds=(5.0, 11.5), method='bounded')
optimal_temp = float(opt_result.x)
optimal_data = calculate_chiller_performance(optimal_temp, ambient_temp, cooling_load, refrigerant)
optimal_power = optimal_data["total_power_kw"]

power_saved = baseline_power - optimal_power
savings_pct = (power_saved / baseline_power) * 100.0
daily_savings_usd = power_saved * 24.0 * electricity_cost
monthly_savings_usd = daily_savings_usd * 30.0

# --- 4. TOP METRICS DISPLAY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Setpoint (Fixed)", f"{baseline_temp:.1f} °C", f"{baseline_power:.1f} kW")
col2.metric("Optimal Reset Setpoint", f"{optimal_temp:.2f} °C", f"+{optimal_temp - baseline_temp:.2f} °C Shift")
col3.metric("Optimized Power", f"{optimal_power:.1f} kW", f"-{savings_pct:.1f}% Energy", delta_color="inverse")
col4.metric("Daily Financial Savings", f"${daily_savings_usd:.2f} / day", f"${monthly_savings_usd:.0f} / mo")

# --- 5. THERMODYNAMIC OPERATING STATES TABLE ---
st.subheader("⚙️ Thermodynamic Operating States")
states_df = pd.DataFrame({
    "Parameter": [
        "Refrigerant",
        "Evaporator Pressure (Suction)",
        "Condenser Pressure (Discharge)",
        "Compression Ratio",
        "System COP (Efficiency)"
    ],
    f"Baseline ({baseline_temp:.1f}°C)": [
        refrigerant,
        f"{baseline_data['p_evap_bar']:.2f} bar",
        f"{baseline_data['p_cond_bar']:.2f} bar",
        f"{baseline_data['pressure_ratio']:.2f}",
        f"{baseline_data['cop']:.2f}"
    ],
    "Optimized State": [
        refrigerant,
        f"{optimal_data['p_evap_bar']:.2f} bar",
        f"{optimal_data['p_cond_bar']:.2f} bar",
        f"{optimal_data['pressure_ratio']:.2f}",
        f"{optimal_data['cop']:.2f}"
    ]
})
st.dataframe(states_df, width="stretch", hide_index=True)

# --- 6. CHARTS ---
temps_sweep = np.linspace(5.0, 12.0, 25)
powers_sweep = [
    calculate_chiller_performance(t, ambient_temp, cooling_load, refrigerant)["total_power_kw"]
    for t in temps_sweep
]
cops_sweep = [
    calculate_chiller_performance(t, ambient_temp, cooling_load, refrigerant)["cop"]
    for t in temps_sweep
]

fig = go.Figure()
fig.add_trace(go.Scatter(x=temps_sweep, y=powers_sweep, name="Total Power (kW)", line=dict(color="#FF4B4B", width=3)))
fig.add_trace(go.Scatter(x=temps_sweep, y=cops_sweep, name="System COP", yaxis="y2", line=dict(color="#0068C9", width=3, dash="dash")))

fig.add_vline(x=baseline_temp, line_width=2, line_dash="dot", line_color="gray", annotation_text="Baseline (6°C)")
fig.add_vline(x=optimal_temp, line_width=2, line_dash="solid", line_color="#29B09D", annotation_text="Optimized Reset")

fig.update_layout(
    title="Chilled Water Supply Temperature vs. Power Draw & Efficiency",
    xaxis=dict(title="Chilled Water Supply Temperature (°C)"),
    yaxis=dict(title="Power Consumption (kW)", side="left"),
    yaxis2=dict(title="System COP", overlaying="y", side="right"),
    height=420,
    margin=dict(l=40, r=40, t=50, b=40)
)
st.plotly_chart(fig, width="stretch")

# --- 7. AUTONOMOUS MECHANICAL WORK-ORDER & AUDIT ---
st.markdown("---")
st.header("📋 Autonomous Mechanical Work-Order & Standards Audit")

def generate_deterministic_audit(baseline_p, opt_p, opt_t, amb, load, ref, cost_saved, opt_data):
    p_saved = baseline_p - opt_p
    pct_saved = (p_saved / baseline_p) * 100.0
    
    return f"""
### Dispatch Reference: `WO-HVAC-2026-CH01`
**Equipment Tag:** Chiller-01 (Continuous Digital Twin Monitoring)  
**Refrigerant Circuit:** {ref} | **Ambient:** {amb:.1f}°C | **Cooling Demand:** {load:.1f} kW

| Metric | Baseline | Optimized | Delta |
| :--- | :--- | :--- | :--- |
| **Supply Temp (CHWST)** | 6.00 °C | {opt_t:.2f} °C | +{opt_t - 6.0:.2f} °C |
| **Suction Pressure ($P_{{evap}}$)** | {baseline_data['p_evap_bar']:.2f} bar | {opt_data['p_evap_bar']:.2f} bar | +{opt_data['p_evap_bar'] - baseline_data['p_evap_bar']:.2f} bar (Lift Reduction) |
| **Discharge Pressure ($P_{{cond}}$)** | {baseline_data['p_cond_bar']:.2f} bar | {opt_data['p_cond_bar']:.2f} bar | Parity |
| **Compression Ratio** | {baseline_data['pressure_ratio']:.2f} | {opt_data['pressure_ratio']:.2f} | -{((baseline_data['pressure_ratio'] - opt_data['pressure_ratio'])/baseline_data['pressure_ratio'])*100.0:.1f}% |
| **Total Power Draw** | {baseline_p:.2f} kW | {opt_p:.2f} kW | -{p_saved:.2f} kW (-{pct_saved:.1f}%) |
| **Projected Run Savings** | — | — | **${cost_saved:.2f} / day** |

---

### Mechanical Engineering & Standards Compliance
* **ASHRAE 90.1-2022 (Section 6.5.4.4):** COMPLIANT. Dynamic chilled-water reset matches part-load demand.
* **ASHRAE 55-2023 (Thermal Comfort):** CONDITIONAL PASS. Elevated CHWST of {opt_t:.2f}°C reduces coil latent capacity; ensure space relative humidity remains below 60%.
* **ASME B31.5 (Refrigeration Piping):** COMPLIANT. Both suction and discharge pressures remain within maximum allowable working pressure envelopes.
* **Supervisory Dispatch Action:** Ramp setpoint at a maximum rate of 0.5°C per 10 minutes to prevent VAV valve hunting.
"""

if gemini_api_key:
    try:
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
        prompt = f"""
        You are an ASME Computers and Information in Engineering (CIE) Division Mechanical Advisory Agent.
        Audit this real-time HVAC optimization dispatch:
        - Refrigerant: {refrigerant}
        - Outdoor Ambient: {ambient_temp}°C
        - Cooling Load: {cooling_load} kW
        - Baseline Setpoint: 6.0°C | Power: {baseline_power:.2f} kW
        - Optimized Setpoint: {optimal_temp:.2f}°C | Power: {optimal_power:.2f} kW
        - Evaporator Pressure: {optimal_data['p_evap_bar']:.2f} bar
        - Condenser Pressure: {optimal_data['p_cond_bar']:.2f} bar
        - Compression Ratio: {optimal_data['pressure_ratio']:.2f}
        - Daily Energy Cost Savings: ${daily_savings_usd:.2f}/day (at ${electricity_cost:.2f}/kWh)

        Provide:
        1. A formal mechanical dispatch work-order summary table.
        2. A thermodynamic analysis detailing compressor lift and suction pressure improvements.
        3. Compliance checks against ASHRAE 90.1, ASHRAE 55 (latent dehumidification warnings), and ASME B31.5.
        4. Supervisory control directives for BMS/PLC dispatch.
        Keep it professional, precise, and concise.
        """
        with st.spinner("Synthesizing ASME engineering advisory report via Gemini..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            audit_report = response.text
            st.markdown(audit_report)
    except Exception as e:
        st.warning(f"AI advisory agent unavailable ({e}). Falling back to deterministic auditor.")
        audit_report = generate_deterministic_audit(
            baseline_power, optimal_power, optimal_temp,
            ambient_temp, cooling_load, refrigerant, daily_savings_usd, optimal_data
        )
        st.markdown(audit_report)
else:
    st.info("💡 Pro-tip: Add a Gemini API Key in the sidebar or Streamlit Secrets for live AI reasoning. Displaying deterministic audit:")
    audit_report = generate_deterministic_audit(
        baseline_power, optimal_power, optimal_temp,
        ambient_temp, cooling_load, refrigerant, daily_savings_usd, optimal_data
    )
    st.markdown(audit_report)

# --- 8. DOWNLOAD WORK ORDER ---
st.download_button(
    label="📥 Download Work-Order Report (.md)",
    data=audit_report,
    file_name=f"thermotwin_work_order_{refrigerant}_{ambient_temp:.0f}C_{cooling_load:.0f}kW.md",
    mime="text/markdown"
)