import streamlit as st
from engine import run_simulation

st.set_page_config(page_title="Berlin Pulse Depot Charging Optimizer", layout="wide")

st.title("Berlin Pulse Depot Charging Optimizer")
st.caption(
    "Compare naive vs carbon-optimal overnight charging strategies for Berlin's "
    "electric bus fleet using real SMARD grid data (2025)."
)

WINDOW_PRESETS = {
    "22:00 – 05:00": (22, 5),
    "23:00 – 05:00": (23, 5),
    "22:00 – 06:00": (22, 6),
    "19:00 – 06:00": (19, 6),
}

with st.sidebar:
    st.header("Simulation inputs")
    n_buses = st.number_input("Fleet size (buses)", min_value=1, value=277, step=1)
    kwh_per_bus = st.number_input("kWh per bus per night", min_value=1.0, value=240.0, step=10.0)
    charger_kw = st.number_input("Charger power (kW)", min_value=1.0, value=50.0, step=5.0)
    window_label = st.selectbox("Charging window", list(WINDOW_PRESETS.keys()))
    run = st.button("Run", type="primary", use_container_width=True)

if run:
    with st.spinner("Running simulation…"):
        res = run_simulation(
            n_buses=int(n_buses),
            kwh_per_bus=float(kwh_per_bus),
            charger_kw=float(charger_kw),
            window_hours=WINDOW_PRESETS[window_label],
        )

    st.subheader("Headline results")

    col1, col2 = st.columns(2)
    col1.metric("CO₂ saved", f"{res['carbon_saving_pct']:.2f} %")
    col2.metric("Cost saved", f"{res['cost_saving_pct']:.2f} %")

    col3, col4 = st.columns(2)
    col3.metric("Fleet CO₂ saved / year", f"{res['fleet_co2_saved_tonnes']:.1f} tonnes")
    col4.metric("Fleet cost saved / year", f"€ {res['fleet_cost_saved_eur']:,.0f}")

    col5, col6 = st.columns(2)
    col5.metric("CO₂ saved per bus / year", f"{res['per_bus']['co2_saved_kg']:.1f} kg")
    col6.metric("Cost saved per bus / year", f"€ {res['per_bus']['cost_saved_eur']:.2f}")

    st.caption(f"Based on {res['n_nights']} complete nights, window {window_label}.")
