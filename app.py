import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import run_simulation, run_simulation_deployable

st.set_page_config(page_title="Berlin Pulse Depot Charging Optimizer", layout="wide")

st.title("Berlin Pulse Depot Charging Optimizer")
st.caption(
    "Compare naive vs carbon-optimal overnight charging strategies for Berlin's "
    "electric bus fleet using real SMARD grid data (2025)."
)

# ---- Mode toggle at the top --------------------------------------------------
mode = st.radio(
    "Scheduling mode",
    options=["Oracle (theoretical upper bound)", "Deployable (day-ahead information only)"],
    horizontal=True,
    help=(
        "Oracle uses per-night perfect foresight of the grid mix — a theoretical "
        "upper bound that cannot be achieved in practice. Deployable uses a fixed "
        "hour-ranking learned from historical data, representing what is achievable "
        "with only day-ahead information."
    ),
)
is_oracle = mode.startswith("Oracle")

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
        sim_kwargs = dict(
            n_buses=int(n_buses),
            kwh_per_bus=float(kwh_per_bus),
            charger_kw=float(charger_kw),
            window_hours=WINDOW_PRESETS[window_label],
        )
        if is_oracle:
            res = run_simulation(**sim_kwargs)
        else:
            res = run_simulation_deployable(**sim_kwargs)

    st.subheader("Headline results")

    if is_oracle:
        st.caption(
            "⚠️ **Upper bound only.** Theoretical maximum assuming perfect foresight; "
            "the deployable carbon saving is much smaller."
        )

    col1, col2 = st.columns(2)
    carbon_label = "CO₂ saved (upper bound)" if is_oracle else "CO₂ saved (deployable)"
    col1.metric(carbon_label, f"{res['carbon_saving_pct']:.2f} %")
    col2.metric("Cost saved", f"{res['cost_saving_pct']:.2f} %")

    col3, col4 = st.columns(2)
    col3.metric("Fleet CO₂ saved / year", f"{res['fleet_co2_saved_tonnes']:.1f} tonnes")
    col4.metric("Fleet cost saved / year", f"€ {res['fleet_cost_saved_eur']:,.0f}")

    col5, col6 = st.columns(2)
    col5.metric("CO₂ saved per bus / year", f"{res['per_bus']['co2_saved_kg']:.1f} kg")
    col6.metric("Cost saved per bus / year", f"€ {res['per_bus']['cost_saved_eur']:.2f}")

    mode_note = "oracle / perfect foresight" if is_oracle else "deployable / frozen day-ahead ranking"
    st.caption(f"Based on {res['n_nights']} complete nights, window {window_label}, mode: {mode_note}.")

    # ---- Chart 1: Carbon-intensity profile across the charging window ----
    st.subheader("Grid Carbon-Intensity Profile")
    profile = res["intensity_profile"]
    hours = list(profile.keys())
    hour_labels = [f"{h:02d}:00" for h in hours]
    means = [profile[h]["mean"] for h in hours]
    p10s = [profile[h]["p10"] for h in hours]
    p90s = [profile[h]["p90"] for h in hours]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hour_labels, y=p90s, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig1.add_trace(go.Scatter(
        x=hour_labels, y=p10s, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
        name="10th–90th percentile",
    ))
    fig1.add_trace(go.Scatter(
        x=hour_labels, y=means, mode="lines+markers",
        line=dict(color="rgb(99,110,250)", width=2.5),
        name="Nightly mean",
    ))
    fig1.update_layout(
        title="Mean Grid Carbon Intensity Across Charging Window",
        xaxis_title="Hour (Berlin time)",
        yaxis_title="Production intensity (g CO₂ / kWh)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: Schedule comparison for one example night ----
    strategy_label = "Carbon-optimal oracle" if is_oracle else "Deployable (frozen ranking)"
    st.subheader(f"Example Night: Naive vs {strategy_label} Schedule")
    ex = res["example_night"]
    ex_hours = ex["hours"]
    ex_labels = [f"{h:02d}:00" for h in ex_hours]
    n_hours = len(ex_hours)

    naive_charging = [1 if i in ex["a_slots"] else 0 for i in range(n_hours)]
    optimal_charging = [1 if i in ex["b_slots"] else 0 for i in range(n_hours)]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=ex_labels, y=[v * 0.9 for v in naive_charging],
        name="Naive (Strategy A)", marker_color="rgba(239,85,59,0.7)",
        width=0.35, offset=-0.2,
    ))
    fig2.add_trace(go.Bar(
        x=ex_labels, y=[v * 0.9 for v in optimal_charging],
        name=f"{strategy_label} (Strategy B)", marker_color="rgba(0,204,150,0.7)",
        width=0.35, offset=0.15,
    ))
    fig2.add_trace(go.Scatter(
        x=ex_labels, y=ex["intensity"], mode="lines+markers",
        name="Intensity (g CO₂/kWh)", yaxis="y2",
        line=dict(color="rgb(99,110,250)", width=2, dash="dot"),
        marker=dict(size=6),
    ))
    fig2.update_layout(
        title=f"Charging Schedule — Night of {ex['night_id']} (median saving)",
        xaxis_title="Hour (Berlin time)",
        yaxis=dict(title="Charging (on / off)", range=[0, 1.1],
                   tickvals=[0, 1], ticktext=["Off", "On"]),
        yaxis2=dict(title="Intensity (g CO₂ / kWh)", overlaying="y",
                    side="right"),
        barmode="group",
        legend=dict(yanchor="top", y=1.12, xanchor="left", x=0.0,
                    orientation="h"),
        margin=dict(t=60, b=40),
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Chart 3: Cumulative CO₂ saved across the year ----
    st.subheader("Cumulative CO₂ Saved Across the Year")
    if is_oracle:
        st.caption(
            "Theoretical maximum assuming perfect foresight; "
            "the deployable carbon saving is much smaller."
        )
    pn = pd.DataFrame(res["per_night"])
    pn["date"] = pd.to_datetime(pn["night_id"])
    pn = pn.sort_values("date")
    fleet_n = res["inputs"]["n_buses"]
    pn["cum_co2_saved_t"] = (pn["co2_saved_kg"] * fleet_n).cumsum() / 1000.0

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=pn["date"], y=pn["cum_co2_saved_t"],
        mode="lines", fill="tozeroy",
        line=dict(color="rgb(0,204,150)", width=2),
        fillcolor="rgba(0,204,150,0.15)",
        name="Cumulative CO₂ saved",
    ))
    fig3.update_layout(
        title=f"Cumulative Fleet CO₂ Saved — {'Oracle (upper bound)' if is_oracle else 'Deployable'}",
        xaxis_title="Date",
        yaxis_title="Cumulative CO₂ saved (tonnes)",
        margin=dict(t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)
