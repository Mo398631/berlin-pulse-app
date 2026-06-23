import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import run_simulation, run_simulation_deployable, run_sensitivity
from scenario_engine import (
    compute_scenario, display_corridor_relief_pct,
    LOW as SCENARIO_LOW, MEDIUM as SCENARIO_MEDIUM, HIGH as SCENARIO_HIGH,
    CORRIDOR_DISPLAY_BAND as SCENARIO_BAND,
    INSINC_PEAK_SHIFT_PCT as INSINC_ANCHOR,
    JR_EAST_PEAK_SHIFT_PCT as JR_EAST_ANCHOR,
)

st.set_page_config(page_title="Berlin Pulse Depot Charging Optimizer", layout="wide")

st.warning(
    "This is a transparent calculator of the equations in the Berlin Pulse\n"
    "   research paper. No operational system exists; all figures are illustrative\n"
    "   or computed from public SMARD data, as described in the paper."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Depot Optimizer", "Scenario Explorer", "Deployability Gap",
    "Robustness (Monte Carlo)", "Unified Model",
])

with tab1:
    st.title("Berlin Pulse Depot Charging Optimizer")
    st.caption(
        "Compare naive vs carbon-optimal overnight charging strategies for Berlin's "
        "electric bus fleet using real SMARD grid data (2025)."
    )

    with st.expander("About this app"):
        st.markdown(
            """
This interactive tool accompanies the research paper:

> **Berlin Pulse: Carbon-Optimal Depot Charging for Electric Bus Fleets**
> *Available on SSRN:* [link forthcoming]
<!-- TODO: replace with actual SSRN URL once published -->

**What it does.** Berlin's electric bus fleet charges overnight at the depot.
The grid's carbon intensity varies hour by hour depending on the generation mix
(wind, solar, coal, gas). This simulator compares two strategies:

- **Strategy A (naive):** start charging as soon as the window opens and fill
  slots chronologically.
- **Strategy B (optimised):** rank the available hours by carbon intensity and
  fill the cleanest ones first.

You can choose between an *oracle* mode (theoretical upper bound using perfect
foresight of each night's grid mix) and a *deployable* mode (a fixed
hour-ranking learned from historical averages — what a real operator could
implement today with only day-ahead information).

Adjust the fleet size, battery capacity, charger power, and charging window in
the sidebar, then press **Run** to see the headline savings, intensity profiles,
an example night's schedule, cumulative CO2 saved over the year, and a
sensitivity tornado chart.

**Data source.** Hourly generation and day-ahead price data from
[SMARD.de](https://www.smard.de) for the calendar year 2025 (364 complete
nights after excluding data-edge dates).
            """
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
        "22:00 – 05:00 (7 h)": (22, 5),
        "23:00 – 05:00 (6 h)": (23, 5),
        "22:00 – 06:00 (8 h)": (22, 6),
        "19:00 – 06:00 (11 h)": (19, 6),
    }

    with st.sidebar:
        st.header("Simulation inputs")
        n_buses = st.number_input("Fleet size (buses)", min_value=1, value=277, step=1)
        kwh_per_bus = st.number_input(
            "Energy per bus per night (kWh)", min_value=1.0, value=240.0, step=10.0,
            help="Total energy each bus needs to charge overnight.",
        )
        charger_kw = st.number_input(
            "Charger power (kW)", min_value=1.0, value=50.0, step=5.0,
            help="Rated power of each charger. Determines how many hours are needed per bus.",
        )
        window_label = st.selectbox("Charging window", list(WINDOW_PRESETS.keys()))
        run_btn = st.button("Run", type="primary", use_container_width=True)

    # ---- Input validation --------------------------------------------------------

    window_tuple = WINDOW_PRESETS[window_label]
    start_h, end_h = window_tuple
    if start_h <= end_h:
        window_length_h = end_h - start_h
    else:
        window_length_h = (24 - start_h) + end_h

    hours_needed = kwh_per_bus / charger_kw
    validation_errors = []

    if charger_kw <= 0:
        validation_errors.append("Charger power must be positive.")
    if kwh_per_bus <= 0:
        validation_errors.append("Energy per bus must be positive.")
    if hours_needed > window_length_h:
        validation_errors.append(
            f"The bus needs **{hours_needed:.1f} hours** to charge "
            f"({kwh_per_bus:.0f} kWh at {charger_kw:.0f} kW), but the selected "
            f"window is only **{window_length_h} hours** long. "
            f"Increase charger power, reduce energy per bus, or widen the window."
        )

    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    if run_btn:
        with st.spinner("Running simulation..."):
            sim_kwargs = dict(
                n_buses=int(n_buses),
                kwh_per_bus=float(kwh_per_bus),
                charger_kw=float(charger_kw),
                window_hours=window_tuple,
            )
            if is_oracle:
                res = run_simulation(**sim_kwargs)
            else:
                res = run_simulation_deployable(**sim_kwargs)

        # ---- Headline results ----
        st.subheader("Headline results")

        if is_oracle:
            st.caption(
                "**Upper bound only.** Theoretical maximum assuming perfect foresight; "
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

        st.divider()

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
        st.subheader(f"Example Night: Naive vs {strategy_label}")
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

        # ---- Chart 3: Cumulative CO2 saved across the year ----
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

        st.divider()

        # ---- Section 4: Sensitivity tornado chart ----
        st.subheader("Sensitivity Analysis")
        st.caption(
            "Each bar shows the range of % saving when one parameter is varied "
            "across a plausible range while all others are held at their default. "
            "The diamond marks the default value."
        )

        with st.spinner("Running sensitivity sweep..."):
            sweeps = run_sensitivity(
                is_oracle=is_oracle,
                base_kwh=float(kwh_per_bus),
                base_kw=float(charger_kw),
                base_window=window_tuple,
                base_n_buses=int(n_buses),
            )

        for metric_key, metric_title in [("carbon_pcts", "CO₂ Saving (%)"),
                                          ("cost_pcts", "Cost Saving (%)")]:
            bars = []
            for s in sweeps:
                vals = s[metric_key]
                lo, hi = min(vals), max(vals)
                default_idx = None
                for i, v in enumerate(s["values"]):
                    if v == s["default"] or str(v) == str(s["default"]):
                        default_idx = i
                        break
                default_val = vals[default_idx] if default_idx is not None else vals[len(vals) // 2]
                range_str = f"{s['values'][0]}–{s['values'][-1]}"
                bars.append(dict(label=s["label"], lo=lo, hi=hi,
                                 default_val=default_val, range_str=range_str,
                                 spread=hi - lo))

            bars.sort(key=lambda b: b["spread"], reverse=True)

            fig_t = go.Figure()
            labels = [b["label"] for b in bars]

            fig_t.add_trace(go.Bar(
                y=labels, x=[b["hi"] - b["lo"] for b in bars],
                base=[b["lo"] for b in bars],
                orientation="h",
                marker_color="rgba(99,110,250,0.6)",
                hovertext=[
                    f"{b['label']}: {b['lo']:.3f}% – {b['hi']:.3f}% "
                    f"(default {b['default_val']:.3f}%, range {b['range_str']})"
                    for b in bars
                ],
                hoverinfo="text",
                showlegend=False,
            ))

            fig_t.add_trace(go.Scatter(
                y=labels, x=[b["default_val"] for b in bars],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="rgb(239,85,59)",
                            line=dict(width=1, color="white")),
                name="Default",
                hovertext=[f"Default: {b['default_val']:.3f}%" for b in bars],
                hoverinfo="text",
            ))

            fig_t.update_layout(
                title=f"Tornado — {metric_title}",
                xaxis_title=metric_title,
                yaxis=dict(autorange="reversed"),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                margin=dict(t=40, b=40, l=160),
                height=300,
            )
            st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    st.title("Scenario Explorer")
    st.caption(
        "Interactive version of the paper's Section 6 congestion scenarios. "
        "Move the adoption sliders (or pick a preset) to see how a modest "
        "behavioural peak-shift cascades into network-wide and corridor-level "
        "peak-trip relief."
    )

    with st.expander("About these scenarios"):
        st.markdown(
            """
These are **illustrative what-if figures**, computed directly from the paper's
Section 6 equations — they are *not* forecasts and rely on no operational data.

The model is pure arithmetic, a chained product of adoption shares:

```
network_peak_reduction = registered_share x active_share x peak_shift_share
corridor_relief        = network_peak_reduction / corridor_share
```

`corridor_share` defaults to **0.25**: the targeted corridors carry roughly a
quarter of peak trips, so the same network-wide shift concentrates about
four-fold on those corridors.

The **peak-shift** slider is anchored to two real-world field programmes:
the **INSINC** trial (≈ **7.49 %** of peak trips shifted) and **JR East**'s
off-peak incentive scheme (≈ **8.5 %**).
            """
        )

    # ---- Preset handling via session_state --------------------------------------
    # Sliders are keyed; preset buttons write their values into session_state
    # *before* the widgets are rendered (on_click runs first), so the sliders
    # pick up the preset on the same rerun.
    SC_DEFAULTS = {
        "sc_registered": 20.0,   # %  (Medium preset as a sensible starting point)
        "sc_active": 45.0,       # %
        "sc_peak_shift": 7.5,    # %
        "sc_corridor": 25.0,     # %
        "sc_energy": 20.0,       # %
    }
    for _k, _v in SC_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    def _apply_scenario_preset(preset):
        st.session_state["sc_registered"] = preset["registered_share"] * 100.0
        st.session_state["sc_active"] = preset["active_share"] * 100.0
        st.session_state["sc_peak_shift"] = preset["peak_shift_share"] * 100.0
        st.session_state["sc_energy"] = preset["energy_shift_share"] * 100.0
        # corridor_share stays at the user's current setting (0.25 default)

    st.markdown("**Presets**")
    pc1, pc2, pc3 = st.columns(3)
    pc1.button("Low", use_container_width=True,
               on_click=_apply_scenario_preset, args=(SCENARIO_LOW,))
    pc2.button("Medium", use_container_width=True,
               on_click=_apply_scenario_preset, args=(SCENARIO_MEDIUM,))
    pc3.button("High", use_container_width=True,
               on_click=_apply_scenario_preset, args=(SCENARIO_HIGH,))

    st.markdown("**Adoption settings**")
    sld1, sld2 = st.columns(2)
    with sld1:
        registered_pct = st.slider(
            "Registered travellers (%)", 0.0, 100.0, step=1.0, key="sc_registered",
            help="Share of travellers enrolled in the incentive scheme.",
        )
        active_pct = st.slider(
            "Active responders among registered (%)", 0.0, 100.0, step=1.0,
            key="sc_active",
            help="Share of enrolled travellers who actually change behaviour.",
        )
        peak_shift_pct = st.slider(
            "Peak trips shifted by an active user (%)", 0.0, 30.0, step=0.1,
            key="sc_peak_shift",
            help="Empirical anchors: INSINC 7.49%, JR East 8.5%.",
        )
    with sld2:
        corridor_pct = st.slider(
            "Corridor concentration (%)", 1.0, 100.0, step=1.0, key="sc_corridor",
            help="Share of peak trips carried by the targeted corridors "
                 "(paper default 25%).",
        )
        energy_pct = st.slider(
            "Energy shifted (%)", 0.0, 100.0, step=1.0, key="sc_energy",
            help="Share of energy demand shifted off-peak (reported alongside "
                 "the trip figures).",
        )

    # ---- Compute (pure engine) ---------------------------------------------------
    sc = compute_scenario(
        registered_share=registered_pct / 100.0,
        active_share=active_pct / 100.0,
        peak_shift_share=peak_shift_pct / 100.0,
        corridor_share=corridor_pct / 100.0,
        energy_shift_share=energy_pct / 100.0,
    )
    corridor_display = display_corridor_relief_pct(sc["corridor_relief_pct"])
    corridor_capped = corridor_display < sc["corridor_relief_pct"]

    st.subheader("Scenario outcome")
    m1, m2, m3 = st.columns(3)
    m1.metric("Network-wide peak reduction", f"{sc['network_peak_reduction_pct']:.2f} %")
    m2.metric(
        "Corridor relief",
        f"{corridor_display:.2f} %",
        help=(f"Raw arithmetic value is {sc['corridor_relief_pct']:.2f} %; "
              f"capped to the paper's {SCENARIO_BAND[0]:.0f}–{SCENARIO_BAND[1]:.0f}% "
              "reported band for display." if corridor_capped else None),
    )
    m3.metric("Energy shifted off-peak", f"{sc['energy_shift_pct']:.0f} %")
    if corridor_capped:
        st.caption(
            f"↑ Corridor relief is shown capped to the paper's reported "
            f"{SCENARIO_BAND[0]:.0f}–{SCENARIO_BAND[1]:.0f}% band "
            f"(raw arithmetic: {sc['corridor_relief_pct']:.1f}%)."
        )

    # ---- Comparison bar chart: current vs the three presets ----------------------
    st.subheader("Your settings vs. the paper's presets")

    scenarios = [("Your settings", dict(
        registered_share=registered_pct / 100.0,
        active_share=active_pct / 100.0,
        peak_shift_share=peak_shift_pct / 100.0,
        corridor_share=corridor_pct / 100.0,
        energy_shift_share=energy_pct / 100.0,
    ))]
    for _name, _preset in [("Low", SCENARIO_LOW), ("Medium", SCENARIO_MEDIUM),
                           ("High", SCENARIO_HIGH)]:
        scenarios.append((_name, dict(_preset)))  # presets use the 0.25 default

    labels = [name for name, _ in scenarios]
    network_vals, corridor_vals = [], []
    for _name, kwargs in scenarios:
        r = compute_scenario(**kwargs)
        network_vals.append(r["network_peak_reduction_pct"])
        corridor_vals.append(display_corridor_relief_pct(r["corridor_relief_pct"]))

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=labels, y=network_vals, name="Network-wide peak reduction (%)",
        marker_color="rgba(99,110,250,0.75)",
    ))
    fig_sc.add_trace(go.Bar(
        x=labels, y=corridor_vals, name="Corridor relief (%, capped)",
        marker_color="rgba(0,204,150,0.75)",
    ))
    # Empirical peak-shift adoption anchors (input-side reference, shown for scale).
    # Labels are pushed to the right side and staggered vertically (INSINC below
    # its line, JR East above its line) so they clear the y-axis and each other.
    fig_sc.add_hline(
        y=INSINC_ANCHOR, line_dash="dash", line_color="rgba(239,85,59,0.9)",
        annotation_text=f"INSINC {INSINC_ANCHOR:.2f}% peak-shift",
        annotation_position="bottom right",
    )
    fig_sc.add_hline(
        y=JR_EAST_ANCHOR, line_dash="dot", line_color="rgba(150,100,30,0.9)",
        annotation_text=f"JR East {JR_EAST_ANCHOR:.1f}%",
        annotation_position="top right",
    )
    fig_sc.update_layout(
        title=dict(text="Illustrative Section 6 Scenario Outcomes",
                   y=0.95, yanchor="top"),
        xaxis_title="Scenario",
        yaxis_title="Reduction / relief (%)",
        barmode="group",
        # Legend moved below the plot so it never touches the title.
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5),
        margin=dict(t=70, b=90, r=40),
        height=440,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.caption(
        "**Illustrative what-if figures from the paper's Section 6 equations — "
        "not forecasts.** Bars are computed from the adoption sliders; the dashed "
        "lines mark empirical *peak-shift* adoption levels from the INSINC "
        "(7.49%) and JR East (8.5%) field programmes, shown as a real-world "
        "reference for the peak-shift input."
    )

with tab3:
    st.info("Coming soon.")

with tab4:
    st.info("Coming soon.")

with tab5:
    st.info("Coming soon.")
