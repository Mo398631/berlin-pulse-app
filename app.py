import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import run_simulation, run_simulation_deployable, run_sensitivity, load_dataset
from crossgrid_engine import compare_grids
from scenario_engine import (
    compute_scenario, display_corridor_relief_pct,
    LOW as SCENARIO_LOW, MEDIUM as SCENARIO_MEDIUM, HIGH as SCENARIO_HIGH,
    CORRIDOR_DISPLAY_BAND as SCENARIO_BAND,
    INSINC_PEAK_SHIFT_PCT as INSINC_ANCHOR,
    JR_EAST_PEAK_SHIFT_PCT as JR_EAST_ANCHOR,
)
import forecast_engine

st.set_page_config(page_title="Berlin Pulse Depot Charging Optimizer", layout="wide")

st.warning(
    "This is a transparent calculator of the equations in the Berlin Pulse\n"
    "   research paper. No operational system exists; all figures are illustrative\n"
    "   or computed from public SMARD data, as described in the paper."
)

# ---- Global sidebar identity (shown on every tab) ----------------------------
with st.sidebar:
    st.subheader("Berlin Pulse")
    st.caption(
        "A transparent calculator for carbon-aware depot charging and "
        "demand-response models of Berlin transit."
    )
    st.markdown("[Read the paper (SSRN)](https://ssrn.com/abstract=6974299)")
    st.caption("Data: SMARD; ENTSO-E.")

overview_tab, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Depot Optimizer", "Scenario Explorer", "Deployability Gap",
    "Robustness (Monte Carlo)", "Unified Model", "Network Prototype",
    "Berlin Pulse Rider", "Cross-Grid Comparison", "Forecast Recovery",
])

with overview_tab:
    st.title("Overview")
    st.write(
        'Berlin Pulse is the interactive companion to the research paper "Berlin '
        'Pulse: A Case Study of AI, Urban Mobility, and Energy-Aware Transport '
        'Policy." Every result here can be explored and checked. Real computed '
        "results and illustrative simulations are labelled throughout."
    )

    # One card per content tab: name (bold) + one-line description.
    overview_cards = [
        ("Depot Optimizer",
         "The core result: naive vs carbon-optimal overnight charging for "
         "Berlin's 277-bus fleet, on real SMARD 2025 grid data."),
        ("Scenario Explorer",
         "Interactive congestion-incentive scenarios from Section 6 "
         "(illustrative what-if figures)."),
        ("Deployability Gap",
         "The honest gap between perfect-foresight and a deployable rule, for "
         "carbon and cost (Appendix B)."),
        ("Robustness (Monte Carlo)",
         "A 10,000-draw stress test of the carbon result under emission-factor "
         "uncertainty (Appendix D)."),
        ("Unified Model",
         "The welfare model: participation vs welfare, and channel "
         "separability (Appendix A)."),
        ("Network Prototype",
         "The redirection mechanism on the real Berlin street and bus network, "
         "with illustrative demand (simulation)."),
        ("Berlin Pulse Rider",
         "A playable demo: be a rider, get rerouted, earn illustrative rewards "
         "(simulation, synthetic data)."),
        ("Cross-Grid Comparison",
         "Does it generalize? The same optimizer on five real European grids "
         "(ENTSO-E data), with Germany as the validated anchor."),
    ]

    for row_start in range(0, len(overview_cards), 2):
        cols = st.columns(2)
        for col, (name, description) in zip(
            cols, overview_cards[row_start:row_start + 2]
        ):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.write(description)

    st.subheader("What is real vs illustrative")
    st.write(
        "Real, computed results: Depot Optimizer, Deployability Gap, "
        "Robustness, Cross-Grid Comparison. Illustrative scenarios and "
        "clearly-labelled simulations: Scenario Explorer, Unified Model, "
        "Network Prototype, Berlin Pulse Rider."
    )

    st.subheader("Start here")
    st.write(
        "Short on time? Open the Depot Optimizer for the core result. Want the "
        "big picture? Open Cross-Grid Comparison. Want to play? Open Berlin "
        "Pulse Rider."
    )

    st.caption(
        "Paper: SSRN 6974299. Grid data: SMARD (Germany 2025) and ENTSO-E "
        "Transparency Platform. Street data: OpenStreetMap (ODbL). Transit "
        "data: VBB (CC BY)."
    )

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

> **Berlin Pulse: A Case Study of AI, Urban Mobility, and Energy-Aware Transport Policy**
> *Available on SSRN:* [ssrn.com/abstract=6974299](https://ssrn.com/abstract=6974299)

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
the **Simulation inputs** panel above, then press **Run** to see the headline
savings, intensity profiles,
an example night's schedule, cumulative CO2 saved over the year, and a
sensitivity tornado chart.

**Data source.** Hourly generation and day-ahead price data from
[SMARD.de](https://www.smard.de) for the calendar year 2025 (364 complete
nights after excluding data-edge dates).

**Data attribution.** Street data © OpenStreetMap contributors (ODbL);
transit data © VBB Verkehrsverbund Berlin-Brandenburg GmbH (CC BY).
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

    with st.container(border=True):
        st.subheader("Simulation inputs")
        in_col1, in_col2, in_col3, in_col4 = st.columns(4)
        with in_col1:
            n_buses = st.number_input("Fleet size (buses)", min_value=1, value=277, step=1)
        with in_col2:
            kwh_per_bus = st.number_input(
                "Energy per bus per night (kWh)", min_value=1.0, value=240.0, step=10.0,
                help="Total energy each bus needs to charge overnight.",
            )
        with in_col3:
            charger_kw = st.number_input(
                "Charger power (kW)", min_value=1.0, value=50.0, step=5.0,
                help="Rated power of each charger. Determines how many hours are needed per bus.",
            )
        with in_col4:
            window_label = st.selectbox("Charging window", list(WINDOW_PRESETS.keys()))
        run_btn = st.button("Run", type="primary")

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
    st.title("Deployability Gap")
    st.caption(
        "Visualises the paper's Appendix B (Table B.2): how much of each "
        "perfect-foresight saving survives once the operator is restricted to a "
        "fixed, day-ahead-only ranking learned from history — split by channel "
        "(carbon vs cost) and by window (train / test / full year)."
    )

    with st.expander("About this gap"):
        st.markdown(
            """
The **oracle** rule reorders each night's charging hours using *that night's*
realised grid mix or price — a perfect-foresight upper bound no operator can
achieve. The **deployable** rule instead freezes a single hour-ranking learned
on the training window (Jan–Sep) and applies it blind to every night, which is
exactly what a real depot could run today with only day-ahead information.

The gap between the two bars is the **foresight premium**: the part of the
theoretical saving that depends on knowing the future and therefore does *not*
transfer to deployment.

- **Train** = Q1–Q3 (273 nights, the window the frozen ranking is learned on).
- **Test** = Q4 (91 nights, genuinely out of sample).
- **Full** = the full year (364 nights).

The numbers are computed by `gate.deployability_results`, reusing the same
simulation primitives as the gate — no physics is re-implemented here.
            """
        )

    from gate import deployability_results

    dep = deployability_results(load_dataset())

    WINDOW_ORDER = [("train", "Train (Q1–Q3)"), ("test", "Test (Q4)"),
                    ("full", "Full year")]
    win_keys = [k for k, _ in WINDOW_ORDER]
    win_labels = [lbl for _, lbl in WINDOW_ORDER]

    COL_DEPLOY = "rgba(0,204,150,0.85)"   # deployable — what survives
    COL_ORACLE = "rgba(99,110,250,0.85)"  # oracle — upper bound

    def _channel_panel(channel, title, yaxis_title):
        deploy_vals = [dep[k][channel]["deployable"] for k in win_keys]
        oracle_vals = [dep[k][channel]["oracle"] for k in win_keys]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=win_labels, y=deploy_vals, name="Deployable (frozen, day-ahead)",
            marker_color=COL_DEPLOY,
            text=[f"{v:.2f}%" for v in deploy_vals], textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=win_labels, y=oracle_vals, name="Oracle (perfect foresight)",
            marker_color=COL_ORACLE,
            text=[f"{v:.2f}%" for v in oracle_vals], textposition="outside",
        ))
        ymax = max(oracle_vals + deploy_vals + [0.1]) * 1.25
        fig.update_layout(
            title=title,
            xaxis_title="Window",
            yaxis=dict(title=yaxis_title, range=[0, ymax]),
            barmode="group",
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="center", x=0.5),
            margin=dict(t=50, b=70),
            height=440,
        )
        return fig

    panel_carbon, panel_cost = st.columns(2)
    with panel_carbon:
        st.plotly_chart(
            _channel_panel("carbon", "Carbon channel", "CO₂ saving vs naive (%)"),
            use_container_width=True,
        )
    with panel_cost:
        st.plotly_chart(
            _channel_panel("cost", "Cost channel", "Cost saving vs naive (%)"),
            use_container_width=True,
        )

    # ---- Headline gap figures, in plain numbers ---------------------------------
    c_dep_full = dep["full"]["carbon"]["deployable"]
    c_orc_full = dep["full"]["carbon"]["oracle"]
    k_dep_full = dep["full"]["cost"]["deployable"]
    k_orc_full = dep["full"]["cost"]["oracle"]
    k_dep_test = dep["test"]["cost"]["deployable"]
    k_orc_test = dep["test"]["cost"]["oracle"]
    cost_transfer = k_dep_test / k_orc_test * 100.0 if k_orc_test else 0.0

    g1, g2, g3 = st.columns(3)
    g1.metric("Deployable carbon (full yr)", f"{c_dep_full:.2f} %",
              help=f"Oracle upper bound is {c_orc_full:.2f}%.")
    g2.metric("Deployable cost (Q4, out of sample)", f"{k_dep_test:.2f} %",
              help=f"Oracle upper bound is {k_orc_test:.2f}%.")
    g3.metric("Cost saving retained out of sample", f"{cost_transfer:.0f} %",
              help="Deployable Q4 cost saving as a share of the Q4 oracle.")

    # ---- Plain-English explanation ----------------------------------------------
    st.subheader("What the gap means")
    st.markdown(
        f"""
The two channels behave very differently:

- **Carbon: the saving lives entirely in the foresight gap.** The deployable
  carbon rule collapses to **≈ 0%** on train, test, *and* the full year (green
  bars flat to the floor). The frozen "cleanest-hours-first" ranking reduces to
  ordinary naive charging because the **overnight carbon profile is nearly
  flat** — every hour from 22:00 to 05:00 has almost the same grid intensity, so
  a fixed ranking can't beat simply charging when the window opens. The
  oracle's **{c_orc_full:.2f}%** full-year carbon saving exists *only* because it
  exploits each night's tiny, unforecastable wiggles — it does not survive
  deployment.

- **Cost: the saving transfers out of sample almost intact.** The deployable
  price rule captures **{k_dep_full:.2f}%** over the full year against an oracle
  of **{k_orc_full:.2f}%**, and out of sample (Q4) it holds **{k_dep_test:.2f}%**
  against a **{k_orc_test:.2f}%** oracle — roughly **{cost_transfer:.0f}% of the
  oracle retained**. Day-ahead prices have a stable, learnable overnight shape
  (consistently cheaper in the small hours), so a frozen ranking captures most
  of the benefit.

**The honest result:** the deployable **carbon co-benefit is near zero**, while
the deployable **cost saving is robust** and survives out of sample. A real
operator should expect meaningful cost savings from day-ahead-aware charging,
but should *not* claim the headline oracle carbon figure as an achievable
co-benefit.
        """
    )

    st.caption(
        "Computed from `gate.deployability_results` (Appendix B, Table B.2). "
        "Bars are savings vs naive Strategy A; the frozen ranking is learned on "
        "the training window and applied blind to every window."
    )

with tab4:
    st.title("Robustness (Monte Carlo)")
    st.caption(
        "Appendix D, Pillar One: an emission-factor Monte Carlo on the carbon "
        "saving. The four conventional direct-combustion factors are replaced by "
        "distributions centred on the paper's values, drawn 10,000 times under a "
        "fixed seed; each draw rebuilds the intensity series and re-runs Strategy "
        "A vs B over the 364 complete nights."
    )

    with st.expander("About this test"):
        st.markdown(
            """
The grid carbon-intensity series is built from per-technology generation using
direct-combustion factors (lignite **1054**, hard coal **884**, fossil gas
**401**, other conventional **700** g CO₂/kWh). How sensitive is the headline
carbon saving to those exact numbers?

This test treats them as distributions whose **mode/centre is the paper value**,
so it measures the *spread around* the existing result rather than relocating it
(Appendix D, Table D.1):

| Source | Paper value | Distribution | Range (g/kWh) |
|---|---|---|---|
| Lignite | 1,054 | Triangular | 980 – 1,054 – 1,140 |
| Hard coal | 884 | Triangular | 800 – 884 – 965 |
| Fossil gas | 401 | Triangular | 350 – 401 – 500 |
| Other conventional | 700 | Uniform | 500 – 900 |
| Renewables, nuclear, biomass, pumped storage | 0 | Fixed | 0 |

For each of **10,000** draws (seed `20250619`) the hourly intensity is rebuilt
via `baseline_simulation.add_intensity`, Strategy A vs B are re-run over the 364
complete nights using the reused scheduling primitives, and the carbon saving
and the Appendix B flatness metric (the spread between the highest and lowest
per-clock-hour mean intensity over the training window) are recorded.
            """
        )

    @st.cache_data(show_spinner="Running the 10,000-draw emission-factor Monte Carlo (~45 s, computed once)...")
    def _load_montecarlo():
        from montecarlo_engine import run_emission_factor_mc
        return run_emission_factor_mc(n_draws=10000, seed=20250619)

    mc = _load_montecarlo()
    savings = mc["carbon_savings"]
    p5, p95 = mc["carbon_p5"], mc["carbon_p95"]
    baseline_repro = mc["deterministic_baseline"]      # 2.388 reproduced anchor
    paper_det = mc["paper_deterministic"]              # 2.41 published headline

    # ---- Headline metrics --------------------------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean carbon saving", f"{mc['carbon_saving_mean']:.3f} %",
              help="Paper Appendix D.2.2: 2.393%.")
    m2.metric("Median carbon saving", f"{mc['carbon_saving_median']:.3f} %",
              help="Paper Appendix D.2.2: 2.392%.")
    m3.metric("90% interval", f"{p5:.3f} – {p95:.3f} %",
              help="5th–95th percentile across 10,000 draws. Paper: 2.329–2.461%.")

    # ---- Histogram of per-draw carbon savings -----------------------------------
    fig_mc = go.Figure()

    # Shade the 90% interval first so the bars and lines sit on top of it.
    fig_mc.add_vrect(
        x0=p5, x1=p95, fillcolor="rgba(0,204,150,0.12)", line_width=0,
        annotation_text="90% interval", annotation_position="top left",
    )
    fig_mc.add_trace(go.Histogram(
        x=savings, nbinsx=60, marker_color="rgba(99,110,250,0.65)",
        name="Per-draw carbon saving",
    ))
    # Reference lines: reproduced baseline (2.388) and the paper's deterministic 2.41.
    fig_mc.add_vline(
        x=baseline_repro, line_dash="dash", line_color="rgb(0,150,110)",
        annotation_text=f"Reproduced baseline {baseline_repro:.3f}%",
        annotation_position="top right",
    )
    fig_mc.add_vline(
        x=paper_det, line_dash="dot", line_color="rgb(239,85,59)",
        annotation_text=f"Paper deterministic {paper_det:.2f}%",
        annotation_position="bottom right",
    )
    fig_mc.update_layout(
        title="Distribution of the Carbon Saving Across 10,000 Emission-Factor Draws",
        xaxis_title="Carbon saving, Strategy B vs A (%)",
        yaxis_title="Number of draws",
        bargap=0.02,
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
        margin=dict(t=50, b=60),
        height=440,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ---- Secondary metrics ------------------------------------------------------
    s1, s2, s3 = st.columns(3)
    s1.metric("Flatness (mean)", f"{mc['flatness_mean']:.2f} g/kWh",
              help="Spread between the highest and lowest per-clock-hour mean "
                   "intensity over the training window. Paper: 12.39 g/kWh. The "
                   "overnight profile stays flat under every plausible factor "
                   "combination.")
    s2.metric("Max carbon saving (any draw)", f"{float(savings.max()):.3f} %",
              help="Single-digit on every one of the 10,000 draws.")
    s3.metric("Incidental cost saving (control)", f"{mc['cost_saving_mean']:.3f} %",
              help="The cost of the carbon schedule, recorded as a control; it "
                   "moves only trivially across the Monte Carlo.")

    # ---- Caption / interpretation ------------------------------------------------
    st.caption(
        f"**The in-sample carbon ceiling is robust.** It stays single-digit "
        f"(max {float(savings.max()):.2f}%) and near 2.4% across 10,000 plausible "
        f"emission-factor combinations — the paper's deterministic {paper_det:.2f}% "
        f"and the reproduced {baseline_repro:.3f}% baseline both fall inside the "
        f"90% interval [{p5:.2f}, {p95:.2f}]. The flat-overnight-profile finding "
        f"also survives: the flatness metric averages {mc['flatness_mean']:.2f} g/kWh, "
        f"well within the band that makes a forecast-free carbon rule ineffective."
    )

    st.info(
        "**Scope.** This is Appendix D **Pillar One** (the emission-factor "
        "Monte Carlo). The consumption-based **Pillar Two** needs ENTSO-E "
        "neighbour-zone import/export data that is not bundled with this app, and "
        "is left as future work."
    )

with tab5:
    from unified_model import (
        welfare, marginal_welfare, bus_block_optimum, passenger_welfare,
        welfare_aggregates, optimum_rho, scenario_rhos, SCENARIO_LABELS,
        DEFAULT_PARAMS,
    )

    st.title("Unified Model")
    st.caption(
        "A conceptual illustration of Appendix A - the social-planner "
        "optimization from which the paper's four claims are derived. Welfare "
        "**W** splits into two independent channels: a passenger (congestion) "
        "block driven by the participation rate **ρ**, and a bus (energy) block "
        "solved by merit-order water-filling. The two are *separable*: the "
        "energy-side optimum does not depend on ρ."
    )

    with st.expander("About this model"):
        st.markdown(
            """
Appendix A poses a single social planner's problem and proves that its welfare
function decomposes additively into two **non-interacting** blocks (Theorem A.22):
an **energy channel** (the bus-side optimum) and a **congestion channel** (the
passenger block).
            """
        )
        st.latex(r"W(\rho) = W_{\mathrm{bus}}^{*} + W_{\mathrm{pax}}(\rho)")
        st.markdown(
            """
**Channel separability (Eq. A.18, Corollary A.23).** The cross-partial Hessian
of the two blocks vanishes, the constraint set factorizes, and so the constrained
maximum decomposes. The practical payoff: the **bus-side optimum is identical for
every ρ** — the energy result of Section 4.7 / 6.5 stands on its own regardless
of passenger uptake.

**Passenger block (Section A.4).** The welfare gradient is *linear* in ρ
(Eq. A.24f):
            """
        )
        st.latex(r"\frac{dW_{\mathrm{pax}}}{d\rho} = A - B\,\rho")
        st.markdown(
            "and welfare is the concave quadratic that integrates it (Eq. A.24j):"
        )
        st.latex(r"W_{\mathrm{pax}}(\rho) = A\,\rho - \frac{1}{2}\,B\,\rho^{2}")
        st.markdown("with the aggregates built from the economic primitives:")
        st.latex(
            r"A = \frac{2\alpha}{K_{\mathrm{peak}}} + \chi + \sigma\,\Delta\varepsilon"
            r" - \beta\,\Delta t"
        )
        st.latex(
            r"B = 2\alpha\left(\frac{1}{K_{\mathrm{peak}}}"
            r" + \frac{1}{K_{\mathrm{off}}}\right) + \chi"
        )
        st.markdown(
            """
Because **B is positive** whenever **α is positive**, the passenger welfare is
**strictly concave**: welfare rises with participation but with **diminishing
marginal returns**. The factor of two on α is the Wardrop-vs-system-optimum gap
— the planner internalizes the crowding externality the individual passenger
ignores (Section A.2.5).

**Energy block (Eq. A.9d / A.25).** The planner charges the nightly energy into
the lowest-weight hours, ranked by the joint cost-and-carbon merit order (the
hourly electricity price plus a carbon weight on each hour's emissions). The
value shown is the fractional saving of that water-filling schedule over naive
earliest-hours charging.

> **This is a conceptual illustration, not a forecast.** The coefficients are
> illustrative (in α-units) and are not Berlin elasticities. The defaults place
> the unconstrained optimum ρ\\* = A / B **beyond** the displayed [0, 1]
> interval, so the curve is shown on its rising, concave arm. Pulling the
> benefit sliders down (or raising β·Δt) moves ρ\\* into view and reproduces the
> interior turning point of the paper's Fig. A.1.
            """
        )

    # ---- Sliders for the main coefficients --------------------------------------
    st.subheader("Passenger-block coefficients")
    c1, c2, c3 = st.columns(3)
    with c1:
        u_alpha = st.slider(
            "α — crowding aversion", 0.01, 1.00, float(DEFAULT_PARAMS["alpha"]),
            0.01, help="Value passengers place on crowding relief. Raises both A "
                       "and B; the factor-of-two internalization makes it the "
                       "main driver of the welfare gain.",
            key="um_alpha")
        u_chi = st.slider(
            "χ — operator crowding cost", 0.0, 1.0, float(DEFAULT_PARAMS["chi"]),
            0.01, help="Convex operator cost of peak crowding (Section A.4.2). "
                       "Adds equally to A and B.",
            key="um_chi")
    with c2:
        u_beta = st.slider(
            "β — schedule-delay coefficient", 0.0, 2.0, float(DEFAULT_PARAMS["beta"]),
            0.01, help="Disutility of arriving Δt late. Enters A as −β·Δt, so it "
                       "lowers the welfare value of shifting.",
            key="um_beta")
        u_dt = st.slider(
            "Δt — shift magnitude (hours)", 0.0, 1.0, float(DEFAULT_PARAMS["dt"]),
            0.05, help="Proposed time-shift (paper: 15–30 min ⇒ ~0.25–0.5 h).",
            key="um_dt")
    with c3:
        u_sdE = st.slider(
            "σ·Δε — carbon damage differential", 0.0, 2.0,
            float(DEFAULT_PARAMS["sigma_dE"]), 0.01,
            help="Social cost of carbon × peak-vs-off-peak per-trip emission gap. "
                 "A pure benefit term: raises A only.",
            key="um_sdE")
        u_koff = st.slider(
            "K_off — off-peak capacity", 0.25, 2.0, float(DEFAULT_PARAMS["K_off"]),
            0.05, help="Off-peak service capacity (K_peak normalised to 1). Lower "
                       "K_off ⇒ off-peak fills up faster ⇒ stronger curvature B.",
            key="um_koff")

    st.subheader("Energy-block (bus) coefficient")
    e1, e2 = st.columns(2)
    with e1:
        u_sigma = st.slider(
            "σ — carbon weight in merit order", 0.0, 5.0,
            float(DEFAULT_PARAMS["sigma"]), 0.1,
            help="Weight on grid carbon intensity vs price in the water-filling "
                 "rank π_t + σ·e_t (Eq. A.9d). Affects only the energy channel.",
            key="um_sigma")
    with e2:
        u_coupling = st.slider(
            "coupling — passenger→bus (0 = separable)", 0.0, 5.0,
            float(DEFAULT_PARAMS["coupling"]), 0.1,
            help="The decoupling assumptions (DA1)–(DA4) set this to zero. At 0 "
                 "the bus optimum is ρ-invariant (separability). Any positive "
                 "value is the specification Section A.3.6 shows would break it.",
            key="um_coupling")

    params = {
        "alpha": u_alpha, "beta": u_beta, "dt": u_dt, "chi": u_chi,
        "sigma_dE": u_sdE, "K_peak": 1.0, "K_off": u_koff,
        "sigma": u_sigma, "coupling": u_coupling,
    }

    A, B = welfare_aggregates(params)
    rho_star = optimum_rho(params)
    w_bus = bus_block_optimum(0.0, params)

    # ---- Headline metrics -------------------------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("A — gradient intercept", f"{A:.3f}",
              help="dW/dρ at ρ = 0 (Eq. A.24f).")
    m2.metric("B — gradient slope", f"{B:.3f}",
              help="Curvature; B > 0 ⇒ strictly concave, diminishing returns.")
    m3.metric("ρ* = A/B — unconstrained optimum", f"{rho_star:.3f}",
              help="Welfare-maximising participation (Eq. A.24i). Beyond 1 with "
                   "the defaults, so [0,1] is the rising arm.")

    # ---- Two-panel figure: W(rho) and dW/drho -----------------------------------
    from plotly.subplots import make_subplots

    rhos = [i / 200.0 for i in range(201)]          # 0.00 ... 1.00
    w_curve = [welfare(r, params) for r in rhos]
    g_curve = [marginal_welfare(r, params) for r in rhos]

    scen_rhos = scenario_rhos()
    scen_w = [welfare(r, params) for r in scen_rhos]
    scen_g = [marginal_welfare(r, params) for r in scen_rhos]

    fig_um = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Welfare  W*(ρ) = W*_bus + W_pax(ρ)",
                        "Marginal welfare  dW*/dρ = A − B·ρ"),
    )

    # Top panel: total welfare.
    fig_um.add_trace(go.Scatter(
        x=rhos, y=w_curve, mode="lines", name="W*(ρ)",
        line=dict(color="rgb(99,110,250)", width=3)), row=1, col=1)
    # The bus-block constant floor (energy channel, ρ-invariant). The line stays
    # at y = w_bus; the label is parked in the empty bottom-right of the panel
    # (above the line, with a vertical gap) so it clears the green line, the blue
    # curve, and the Low/Medium/High markers.
    fig_um.add_hline(
        y=w_bus, line_dash="dot", line_color="rgb(0,150,110)",
        annotation_text=f"W*_bus = {w_bus:.4f} (energy channel, ρ-invariant)",
        annotation_position="top right", annotation_yshift=14,
        annotation_font_color="rgb(0,150,110)", row=1, col=1)
    # Scenario points.
    fig_um.add_trace(go.Scatter(
        x=list(scen_rhos), y=scen_w, mode="markers+text",
        text=SCENARIO_LABELS, textposition="top center",
        marker=dict(color="rgb(239,85,59)", size=11, symbol="circle"),
        name="Scenarios (ρ = 0.05, 0.20, 0.40)"), row=1, col=1)

    # Bottom panel: marginal welfare.
    fig_um.add_trace(go.Scatter(
        x=rhos, y=g_curve, mode="lines", name="dW*/dρ",
        line=dict(color="rgb(171,99,250)", width=3),
        showlegend=True), row=2, col=1)
    fig_um.add_hline(y=0.0, line_dash="dash", line_color="grey", row=2, col=1)
    fig_um.add_trace(go.Scatter(
        x=list(scen_rhos), y=scen_g, mode="markers+text",
        text=SCENARIO_LABELS, textposition="top center",
        marker=dict(color="rgb(239,85,59)", size=11, symbol="circle"),
        showlegend=False), row=2, col=1)
    # If the optimum falls inside the frame, mark it.
    if 0.0 <= rho_star <= 1.0:
        fig_um.add_vline(
            x=rho_star, line_dash="dot", line_color="rgb(150,150,150)",
            annotation_text=f"ρ* = {rho_star:.3f}",
            annotation_position="top", row=1, col=1)
        fig_um.add_vline(x=rho_star, line_dash="dot",
                         line_color="rgb(150,150,150)", row=2, col=1)

    fig_um.update_xaxes(title_text="Participation / registration rate  ρ",
                        row=2, col=1)
    fig_um.update_yaxes(title_text="Welfare (norm.)", row=1, col=1)
    fig_um.update_yaxes(title_text="dW*/dρ", row=2, col=1)
    fig_um.update_layout(
        height=620, margin=dict(t=50, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_um, use_container_width=True)

    # ---- Plain-English explanation ----------------------------------------------
    sep_note = (
        "**zero (separable)**: the energy-side optimum is identical at every ρ"
        if u_coupling == 0.0 else
        f"**{u_coupling:.1f} (broken)**: a positive coupling makes the bus "
        "optimum drift with ρ — the very specification Section A.3.6 flags"
    )
    st.markdown(
        f"""
**What the two panels say.**

- **The channels are mathematically independent (separable).** The green dotted
  floor in the top panel is the energy-channel optimum $W^{{*}}_{{\\text{{bus}}}}$.
  It is a flat line because it **does not move with ρ** — the merit-order
  water-filling result of the bus side stands on its own no matter how many
  passengers participate. The passenger→bus coupling slider is currently
  {sep_note}.

- **Welfare rises with participation, with diminishing returns.** $W^{{*}}(ρ)$
  climbs across the interval but **flattens** (top panel), because its slope
  $dW^{{*}}/dρ = A − B·ρ$ falls **linearly** (bottom panel). Each extra
  registrant adds less welfare than the last: the textbook concavity of
  Eq. A.24j. The three scenario points (Low/Medium/High = ρ 0.05/0.20/0.40)
  sit on the rising arm, so Medium beats Low and High beats Medium — the
  paper's welfare ordering as a corollary of strict concavity, not three
  separate stories.
        """
    )

    st.caption(
        "Conceptual illustration of Appendix A (Eq. A.17, A.18, A.24f, A.24j). "
        "Coefficients are illustrative, in α-units, and are not Berlin "
        "elasticities. The bus-block value is a fractional merit-order saving on "
        "the illustrative overnight profile of Appendix B; no operational system "
        "exists."
    )

with tab6:
    import json as _json
    from pathlib import Path as _Path

    import pydeck as pdk

    from prototype_engine import simulate_redirection

    st.title("Network Prototype")
    st.caption(
        "Map view of the Section 6 redirection prototype: it spreads the same "
        "network-wide peak-shift across Berlin's named arterial corridors so you "
        "can see *where* the relief lands."
    )

    # ---- STEP 2: mandatory honesty banner ---------------------------------------
    st.warning(
        "Illustrative simulation. Demand is synthetic and the redirection "
        "mechanism reproduces the paper's Section 6 scenario figures on the real "
        "Berlin street and bus network. This is not a deployed system and uses no "
        "live or personal data."
    )

    # ---- STEP 3: controls (sliders + Low/Medium/High presets) -------------------
    # Sliders are keyed; preset buttons write their values into session_state
    # before the widgets render (on_click runs first), so the same rerun picks
    # them up -- the pattern already used by the Scenario Explorer tab.
    NP_DEFAULTS = {
        "np_registered": 20.0,   # %  (Medium preset, a sensible starting point)
        "np_active": 45.0,       # %
        "np_peak_shift": 7.5,    # %
        "np_corridor": 25.0,     # %  (corridor concentration, paper default)
    }
    for _k, _v in NP_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    def _apply_network_preset(preset):
        st.session_state["np_registered"] = preset["registered_share"] * 100.0
        st.session_state["np_active"] = preset["active_share"] * 100.0
        st.session_state["np_peak_shift"] = preset["peak_shift_share"] * 100.0
        # corridor concentration stays at the user's current setting (0.25 default)

    st.markdown("**Presets** (Section 6 scenarios)")
    pc1, pc2, pc3 = st.columns(3)
    pc1.button("Low", use_container_width=True, key="np_preset_low",
               on_click=_apply_network_preset, args=(SCENARIO_LOW,))
    pc2.button("Medium", use_container_width=True, key="np_preset_med",
               on_click=_apply_network_preset, args=(SCENARIO_MEDIUM,))
    pc3.button("High", use_container_width=True, key="np_preset_high",
               on_click=_apply_network_preset, args=(SCENARIO_HIGH,))

    st.markdown("**Adoption settings**")
    sc1, sc2 = st.columns(2)
    with sc1:
        np_registered_pct = st.slider(
            "Registered travellers (%)", 0.0, 100.0, step=1.0,
            key="np_registered",
            help="Share of travellers enrolled in the incentive scheme.")
        np_active_pct = st.slider(
            "Active responders among registered (%)", 0.0, 100.0, step=1.0,
            key="np_active",
            help="Share of enrolled travellers who actually change behaviour.")
    with sc2:
        np_peak_shift_pct = st.slider(
            "Peak trips shifted by an active user (%)", 0.0, 30.0, step=0.1,
            key="np_peak_shift",
            help="Empirical anchors: INSINC 7.49%, JR East 8.5%.")
        np_corridor_pct = st.slider(
            "Corridor concentration (%)", 1.0, 100.0, step=1.0,
            key="np_corridor",
            help="Share of peak trips carried by the targeted corridors "
                 "(paper default 25%). Lower = the network shift concentrates "
                 "more strongly on the targeted arterials.")

    # ---- Compute the spatial redirection (pure engine, reconciled aggregates) ---
    sim = simulate_redirection(
        registered_share=np_registered_pct / 100.0,
        active_share=np_active_pct / 100.0,
        peak_shift_share=np_peak_shift_pct / 100.0,
        corridor_share=np_corridor_pct / 100.0,
    )

    # ---- STEP 4: pydeck map -----------------------------------------------------
    view = st.radio(
        "Map view", ["Before", "After"], index=1, horizontal=True,
        help="Colour each corridor by its peak load: 'Before' uses the baseline "
             "peak, 'After' uses the post-redirection peak. Targeted corridors "
             "shift visibly toward green; non-targeted ones barely change.")

    def _as_lines(coords):
        """Flatten GeoJSON LineString / MultiLineString coords to a list of lines.

        Each returned line is a list of [lon, lat] pairs, ready for a pydeck
        PathLayer `path`.
        """
        if not coords:
            return []
        first = coords[0]
        if (isinstance(first, (list, tuple)) and len(first) == 2
                and all(isinstance(v, (int, float)) for v in first)):
            return [coords]                      # already a single line
        lines = []
        for sub in coords:
            lines.extend(_as_lines(sub))
        return lines

    # Colour scale: normalise each corridor's load against the global BEFORE peak
    # so the scale is identical in both views. High load -> red, low -> green;
    # because the After view uses the (smaller) after_peak on the same scale,
    # relieved corridors slide down toward green.
    max_before = max((c["before_peak"] for c in sim["per_corridor"]), default=1.0) or 1.0

    def _load_color(load):
        t = min(max(load / max_before, 0.0), 1.0)
        r = int(220 * t + 30)
        g = int(200 * (1.0 - t) + 30)
        return [r, g, 45]

    corridor_rows = []
    for c in sim["per_corridor"]:
        load = c["before_peak"] if view == "Before" else c["after_peak"]
        color = _load_color(load)
        for line in _as_lines(c["coords"]):
            corridor_rows.append({
                "name": c["name"],
                "path": [[float(x), float(y)] for x, y in line],
                "color": color,
                "before_peak": int(round(c["before_peak"])),
                "after_peak": int(round(c["after_peak"])),
                "relief_pct": round(c["relief_pct"], 1),
                "targeted": "targeted" if c["targeted"] else "non-targeted",
            })

    layers = []

    # Faint secondary layer: the real BVG bus lines, if the file is present.
    _bus_path = _Path(__file__).resolve().parent / "prototype_data" / "bus_lines.geojson"
    if _bus_path.exists():
        try:
            with open(_bus_path, "r", encoding="utf-8") as _fh:
                _bus_fc = _json.load(_fh)
            bus_rows = []
            for _feat in _bus_fc.get("features", []):
                _geom = (_feat.get("geometry") or {}).get("coordinates")
                for _line in _as_lines(_geom):
                    bus_rows.append({"path": [[float(x), float(y)] for x, y in _line]})
            if bus_rows:
                layers.append(pdk.Layer(
                    "PathLayer", data=bus_rows, get_path="path",
                    get_color=[150, 150, 160], get_width=12,
                    width_min_pixels=1, width_max_pixels=3,
                    opacity=0.25, pickable=False,
                ))
        except (OSError, ValueError, KeyError):
            pass   # the bus layer is optional decoration; never block the map

    # Primary layer: the corridors, coloured by load.
    layers.append(pdk.Layer(
        "PathLayer", data=corridor_rows, get_path="path",
        get_color="color", get_width=30,
        width_min_pixels=3, width_max_pixels=9,
        pickable=True, auto_highlight=True,
    ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=52.52, longitude=13.405, zoom=11, pitch=0,
        ),
        map_provider="carto",
        map_style="road",
        tooltip={
            "html": "<b>{name}</b> ({targeted})<br/>"
                    "Before peak: {before_peak}<br/>"
                    "After peak: {after_peak}<br/>"
                    "Relief: {relief_pct}%",
            "style": {"backgroundColor": "rgba(20,20,30,0.9)", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        "Red = high peak load, green = relieved. Bus lines (BVG) are the faint "
        "grey background layer. Switch between **Before** and **After** to watch "
        "the targeted corridors slide toward green while the rest hold steady. "
        "Geometry is the real Berlin street/bus network; the peak loads are "
        "synthetic, seeded demand — not measured traffic."
    )

    # ---- STEP 5: metrics, corridor table, reconciliation caption ----------------
    st.subheader("Aggregate outcome")
    np_corridor_display = display_corridor_relief_pct(sim["corridor_relief_pct"])
    np_capped = np_corridor_display < sim["corridor_relief_pct"]
    nm1, nm2 = st.columns(2)
    nm1.metric("Network-wide peak reduction",
               f"{sim['network_peak_reduction_pct']:.2f} %")
    nm2.metric(
        "Corridor relief",
        f"{np_corridor_display:.2f} %",
        help=(f"Raw arithmetic value is {sim['corridor_relief_pct']:.2f} %; "
              f"capped to the paper's {SCENARIO_BAND[0]:.0f}-{SCENARIO_BAND[1]:.0f}% "
              "reported band for display, exactly as the Scenario Explorer tab "
              "does." if np_capped else None))

    np_df = pd.DataFrame([{
        "Corridor": c["name"],
        "Targeted": "Yes" if c["targeted"] else "No",
        "Before peak": int(round(c["before_peak"])),
        "After peak": int(round(c["after_peak"])),
        "Relief %": round(c["relief_pct"], 2),
    } for c in sim["per_corridor"]])
    st.dataframe(np_df, use_container_width=True, hide_index=True)

    st.caption(
        "These aggregate figures match the Scenario Explorer tab because they use "
        "the same validated Section 6 model; the map only distributes the effect "
        "spatially over illustrative demand."
    )

# --- Tab 7 phone component: one self-contained HTML/CSS/JS document. All state
#     and tap-to-tap navigation live in the browser; Python only injects the
#     precomputed rider_engine outcomes at "__RIDER_DATA__". No f-string here so
#     the CSS/JS braces stay literal. ---------------------------------------------
_RIDER_COMPONENT_TEMPLATE = r"""
<!doctype html>
<meta charset="utf-8">
<style>
  :root{
    --accent:#1d9e75; --accent-soft:rgba(29,158,117,.16);
    --bg:#0e1116; --panel:#161b22; --panel2:#1c232c; --line:#2a323d;
    --text:#e6edf3; --muted:#8b949e; --warn:#e3a008; --red:#e5534b;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:transparent;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  #stage{display:flex;justify-content:center;padding:14px 8px 22px;}
  #phone{
    position:relative;width:360px;height:740px;background:var(--bg);
    border-radius:38px;border:1px solid var(--line);
    box-shadow:0 18px 50px rgba(0,0,0,.55),0 0 0 8px #05070a;
    color:var(--text);overflow:hidden;display:flex;flex-direction:column;
  }
  #notch{position:absolute;top:0;left:50%;transform:translateX(-50%);
    width:140px;height:24px;background:#05070a;border-radius:0 0 16px 16px;z-index:5;}
  #statusbar{display:flex;justify-content:space-between;align-items:center;
    padding:34px 18px 8px;font-size:12.5px;color:var(--muted);}
  #clock{font-weight:600;color:var(--text);letter-spacing:.3px;}
  #wallet{display:flex;align-items:center;gap:5px;background:var(--accent-soft);
    color:var(--accent);padding:4px 9px;border-radius:999px;font-weight:600;
    border:1px solid rgba(29,158,117,.35);}
  #wallet .hex{font-size:12px;}
  #screens{position:relative;flex:1;overflow:hidden;}
  .screen{position:absolute;inset:0;padding:6px 20px 12px;overflow-y:auto;
    opacity:0;transform:translateX(14px);pointer-events:none;
    transition:opacity .32s ease,transform .32s ease;}
  .screen.active{opacity:1;transform:none;pointer-events:auto;}
  .screen::-webkit-scrollbar{width:0;}
  #footer{padding:8px 16px 12px;font-size:10px;line-height:1.35;color:var(--muted);
    text-align:center;border-top:1px solid var(--line);background:rgba(0,0,0,.18);}
  h2{font-size:21px;margin:6px 0 4px;letter-spacing:.2px;}
  h3{font-size:15px;margin:14px 0 8px;color:var(--text);}
  p.tag{color:var(--muted);font-size:13.5px;line-height:1.45;margin:2px 0 14px;}
  .btn{display:block;width:100%;border:0;cursor:pointer;border-radius:14px;
    padding:14px;font-size:15px;font-weight:650;color:#04130d;
    background:var(--accent);transition:transform .08s ease,filter .15s ease;}
  .btn:hover{filter:brightness(1.07);} .btn:active{transform:scale(.985);}
  .btn.ghost{background:var(--panel2);color:var(--text);border:1px solid var(--line);}
  label.fld{display:block;font-size:12px;color:var(--muted);margin:12px 0 5px;
    text-transform:uppercase;letter-spacing:.6px;}
  select{width:100%;padding:13px 12px;border-radius:12px;background:var(--panel);
    color:var(--text);border:1px solid var(--line);font-size:14.5px;appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),
      linear-gradient(135deg,var(--muted) 50%,transparent 50%);
    background-position:calc(100% - 18px) 19px,calc(100% - 13px) 19px;
    background-size:5px 5px,5px 5px;background-repeat:no-repeat;}
  .chips{display:flex;gap:8px;}
  .chip{flex:1;text-align:center;padding:11px 6px;border-radius:12px;cursor:pointer;
    background:var(--panel);border:1px solid var(--line);font-size:13px;color:var(--text);
    transition:.15s;}
  .chip small{display:block;color:var(--muted);font-size:10.5px;margin-top:2px;}
  .chip.sel{background:var(--accent-soft);border-color:var(--accent);color:var(--accent);}
  .chip.sel small{color:var(--accent);}
  .brand{display:flex;flex-direction:column;align-items:center;text-align:center;
    margin-top:54px;}
  .mark{width:96px;height:96px;border-radius:50%;display:grid;place-items:center;
    background:radial-gradient(circle at 50% 40%,rgba(29,158,117,.32),transparent 70%);
    margin-bottom:14px;}
  .mark .core{width:30px;height:30px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 0 8px rgba(29,158,117,.25),0 0 0 18px rgba(29,158,117,.12);
    animation:beat 1.8s ease-in-out infinite;}
  @keyframes beat{0%,100%{transform:scale(.92);}50%{transform:scale(1.12);}}
  .disc{margin-top:18px;font-size:11px;color:var(--muted);line-height:1.4;
    background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:10px 12px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:14px;margin:10px 0;}
  .banner{border-radius:14px;padding:13px 14px;margin:8px 0 4px;font-size:13.5px;
    line-height:1.45;}
  .banner.red{background:rgba(229,83,75,.12);border:1px solid rgba(229,83,75,.5);color:#ffb4ae;}
  .banner.ok{background:var(--accent-soft);border:1px solid rgba(29,158,117,.5);color:#7ee0bd;}
  .route{font-size:13px;color:var(--muted);margin:2px 0 0;}
  .route b{color:var(--text);}
  .opt{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:13px;margin:10px 0;cursor:pointer;transition:.15s;}
  .opt:hover{border-color:var(--accent);background:var(--panel2);}
  .opt:active{transform:scale(.99);}
  .opt .top{display:flex;justify-content:space-between;align-items:center;gap:8px;}
  .opt .ty{font-weight:700;font-size:14px;}
  .opt .pts{color:var(--accent);font-weight:700;font-size:14px;white-space:nowrap;}
  .opt .desc{font-size:12.5px;color:var(--muted);margin:6px 0 8px;line-height:1.4;}
  .tagrow{display:flex;flex-wrap:wrap;gap:6px;}
  .pill{font-size:11px;padding:3px 8px;border-radius:999px;background:var(--panel2);
    border:1px solid var(--line);color:var(--text);}
  .pill.green{color:var(--accent);border-color:rgba(29,158,117,.4);background:var(--accent-soft);}
  .loadwrap{display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:100%;text-align:center;}
  .pulse{width:70px;height:70px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 0 0 rgba(29,158,117,.55);animation:ring 1.4s ease-out infinite;}
  @keyframes ring{0%{transform:scale(.7);box-shadow:0 0 0 0 rgba(29,158,117,.55);}
    70%{transform:scale(1);box-shadow:0 0 0 30px rgba(29,158,117,0);}
    100%{transform:scale(.7);box-shadow:0 0 0 0 rgba(29,158,117,0);}}
  .status{margin-top:26px;color:var(--muted);font-size:13.5px;min-height:18px;}
  svg.map{width:100%;height:200px;background:var(--panel);border:1px solid var(--line);
    border-radius:14px;display:block;}
  .big{font-size:26px;font-weight:750;text-align:center;margin:18px 0 2px;}
  .sub{text-align:center;color:var(--muted);font-size:13px;margin-bottom:6px;}
  .reward-line{display:flex;justify-content:space-between;padding:8px 0;
    border-bottom:1px solid var(--line);font-size:13.5px;}
  .reward-line:last-child{border-bottom:0;}
  .reward-line b{color:var(--accent);}
  .partner{font-size:11.5px;color:var(--muted);margin-top:10px;line-height:1.4;}

  /* ---- loading: "reading the network" with rotating tips / facts / jokes ---- */
  .load-title{margin-top:24px;font-size:16px;font-weight:650;color:var(--text);
    letter-spacing:.2px;}
  .tipcard{margin-top:22px;width:100%;max-width:300px;background:var(--panel);
    border:1px solid var(--line);border-radius:14px;padding:16px;min-height:116px;
    display:flex;flex-direction:column;gap:10px;align-items:flex-start;
    transition:opacity .32s ease,transform .32s ease;}
  .tipcard.fade{opacity:0;transform:translateY(7px);}
  .tiplabel{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    color:var(--accent);background:var(--accent-soft);
    border:1px solid rgba(29,158,117,.35);border-radius:999px;padding:3px 10px;}
  .tiplabel.joke{color:var(--warn);background:rgba(227,160,8,.12);
    border-color:rgba(227,160,8,.4);}
  .tiplabel.fact{color:#79c0ff;background:rgba(56,139,253,.12);
    border-color:rgba(56,139,253,.4);}
  .tiptext{font-size:13.5px;line-height:1.5;color:var(--text);text-align:left;}
  .loaddots{display:flex;gap:6px;margin-top:20px;}
  .loaddots i{width:6px;height:6px;border-radius:50%;background:var(--muted);
    opacity:.4;animation:ld 1.2s ease-in-out infinite;}
  .loaddots i:nth-child(2){animation-delay:.18s;}
  .loaddots i:nth-child(3){animation-delay:.36s;}
  @keyframes ld{0%,100%{opacity:.35;transform:translateY(0);}
    50%{opacity:1;transform:translateY(-3px);}}

  /* ---- confetti burst (pure CSS/JS, no libraries) ---- */
  #confetti{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:30;}
  .confetti-pc{position:absolute;top:-16px;border-radius:2px;will-change:transform,opacity;
    animation:confFall linear forwards;}
  @keyframes confFall{0%{transform:translateY(0) rotateZ(0);opacity:1;}
    100%{transform:translateY(780px) rotateZ(560deg);opacity:.9;}}

  /* ---- arrival celebration extras ---- */
  .walletrow{text-align:center;color:var(--muted);font-size:13px;margin:0 0 10px;}
  .walletrow b{color:var(--accent);}
  .ticket{position:relative;display:flex;align-items:center;gap:12px;margin:12px 0 2px;
    padding:14px 16px;border-radius:14px;color:#04130d;overflow:hidden;
    background:linear-gradient(110deg,#27d99a,#1d9e75);
    box-shadow:0 8px 22px rgba(29,158,117,.35);}
  .ticket::after{content:"";position:absolute;top:0;left:-60%;width:55%;height:100%;
    background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent);
    animation:shine 2.2s ease-in-out infinite;}
  @keyframes shine{0%{left:-60%;}55%,100%{left:135%;}}
  .ticket .tk-ico{font-size:24px;animation:wiggle 1.8s ease-in-out infinite;}
  @keyframes wiggle{0%,100%{transform:rotate(-8deg);}50%{transform:rotate(8deg);}}
  .ticket .tk-main{font-weight:750;font-size:14px;}
  .ticket .tk-sub{font-size:11px;opacity:.85;}
  .co2line{text-align:center;font-size:12.5px;color:var(--accent);margin:12px 0 6px;
    line-height:1.5;}
  .co2line b{color:var(--text);} .co2line small{color:var(--muted);}

  /* ---- bottom tab bar ---- */
  #tabbar{display:flex;border-top:1px solid var(--line);background:rgba(0,0,0,.25);}
  .tabbtn{flex:1;border:0;background:transparent;cursor:pointer;color:var(--muted);
    padding:9px 0 8px;font-size:10.5px;font-weight:600;display:flex;
    flex-direction:column;align-items:center;gap:3px;transition:.15s;}
  .tabbtn:hover{color:var(--text);} .tabbtn.sel{color:var(--accent);}
  .tabbtn .ti{font-size:18px;line-height:1;}

  /* ---- rewards screen ---- */
  .wallet-hero{margin:8px 0 4px;padding:18px;border-radius:16px;text-align:center;
    background:radial-gradient(circle at 50% 0%,var(--accent-soft),var(--panel) 72%);
    border:1px solid rgba(29,158,117,.35);}
  .wallet-hero-pts{font-size:34px;font-weight:800;color:var(--accent);line-height:1;}
  .wallet-hero-lbl{font-size:12px;color:var(--muted);margin-top:4px;
    text-transform:uppercase;letter-spacing:1px;}
  .rw-stats{display:flex;gap:10px;margin:10px 0;}
  .rw-stat{flex:1;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:11px;text-align:center;}
  .rw-stat b{display:block;font-size:20px;color:var(--text);}
  .rw-stat small{color:var(--muted);font-size:11px;}
  .illus{font-size:9.5px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
    color:var(--warn);background:rgba(227,160,8,.12);border:1px solid rgba(227,160,8,.4);
    border-radius:999px;padding:2px 7px;margin-left:6px;vertical-align:middle;}
  .challenge{display:flex;flex-direction:column;gap:9px;}
  .challenge .ch-top{display:flex;justify-content:space-between;font-size:13.5px;
    color:var(--text);}
  .challenge .ch-top b{color:var(--accent);}
  .ch-bar{height:8px;border-radius:999px;background:var(--panel2);overflow:hidden;}
  .ch-fill{height:100%;border-radius:999px;background:var(--accent);transition:width .4s ease;}
  .badges{display:flex;gap:9px;}
  .badge{flex:1;text-align:center;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:12px 6px;opacity:.45;filter:grayscale(1);transition:.25s;}
  .badge.earned{opacity:1;filter:none;border-color:rgba(29,158,117,.45);
    background:var(--accent-soft);}
  .badge .bi{font-size:24px;line-height:1;}
  .badge .bl{font-size:10.5px;margin-top:6px;color:var(--text);line-height:1.25;}
  .perk{display:flex;justify-content:space-between;align-items:center;gap:10px;
    background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:11px 13px;margin:8px 0;}
  .perk .pk-name{font-size:13px;color:var(--text);}
  .perk .pk-cost{font-size:11.5px;color:var(--muted);margin-top:2px;}
  .perk .pk-state{font-size:11px;font-weight:650;white-space:nowrap;padding:5px 10px;
    border-radius:999px;}
  .perk .pk-state.ok{color:var(--accent);background:var(--accent-soft);
    border:1px solid rgba(29,158,117,.4);}
  .perk .pk-state.locked{color:var(--muted);background:var(--panel2);
    border:1px solid var(--line);}
</style>

<div id="stage">
  <div id="phone">
    <div id="notch"></div>
    <div id="statusbar">
      <span id="clock">08:15</span>
      <span id="wallet"><span class="hex">&#x2B22;</span> <b id="walletPts">0</b> Pulse Points</span>
    </div>

    <div id="screens">
      <!-- a) WELCOME -->
      <section class="screen active" id="s-welcome">
        <div class="brand">
          <div class="mark"><div class="core"></div></div>
          <h2>Berlin Pulse Rider</h2>
          <p class="tag">Shift your trip, ease the peak,<br>earn Pulse Points.</p>
          <button class="btn" onclick="go('plan')">Start</button>
          <div class="disc">Demo &mdash; simulated Berlin, synthetic conditions,
            illustrative rewards.</div>
        </div>
      </section>

      <!-- b) PLAN -->
      <section class="screen" id="s-plan">
        <h2>Plan your trip</h2>
        <p class="tag">Pick where you're going and when you'd leave.</p>
        <label class="fld">From</label>
        <select id="fromSel" onchange="onFromChange()"></select>
        <label class="fld">To</label>
        <select id="toSel"></select>
        <label class="fld">When are you leaving?</label>
        <div class="chips" id="chipRow"></div>
        <div style="height:18px"></div>
        <button class="btn" onclick="findRoute()">Find my route</button>
      </section>

      <!-- c) LOADING: "reading the network" + rotating tips / facts / jokes -->
      <section class="screen" id="s-loading">
        <div class="loadwrap">
          <div class="pulse"></div>
          <div class="load-title">Reading the network&hellip;</div>
          <div class="tipcard" id="tipCard">
            <span class="tiplabel" id="tipLabel">TIP</span>
            <div class="tiptext" id="tipText"></div>
          </div>
          <div class="loaddots"><i></i><i></i><i></i></div>
        </div>
      </section>

      <!-- d) VERDICT -->
      <section class="screen" id="s-verdict">
        <h2 id="verdictTitle">Your route</h2>
        <div id="verdictBody"></div>
        <button class="btn ghost" style="margin-top:10px" onclick="go('plan')">Change trip</button>
      </section>

      <!-- e) CHOOSE / animate -->
      <section class="screen" id="s-choose">
        <h2 id="chooseTitle">On your way</h2>
        <p class="tag" id="chooseSub">Following your chosen route&hellip;</p>
        <svg class="map" id="map" viewBox="0 0 300 200" preserveAspectRatio="xMidYMid meet"></svg>
      </section>

      <!-- f) ARRIVAL: confetti + celebratory reward card -->
      <section class="screen" id="s-arrival">
        <h2 style="text-align:center">You've arrived! &#x1F389;</h2>
        <div class="big" id="arrivePts">+0</div>
        <div class="sub">Pulse Points earned</div>
        <div class="walletrow">Wallet total: <b id="arriveWallet">0</b> Pulse Points</div>
        <div class="card" id="arriveCard"></div>
        <div class="ticket" id="arriveTicket" style="display:none"></div>
        <div class="co2line" id="arriveCo2"></div>
        <button class="btn" onclick="go('plan')">Plan another trip</button>
        <button class="btn ghost" style="margin-top:8px" onclick="openRewards()">See my rewards</button>
      </section>

      <!-- g) REWARDS: wallet, perks, lottery, badges, weekly challenge -->
      <section class="screen" id="s-rewards">
        <h2>Rewards</h2>
        <p class="tag">Everything you've earned this session.</p>
        <div class="wallet-hero">
          <div class="wallet-hero-pts" id="rwWallet">0</div>
          <div class="wallet-hero-lbl">Pulse Points</div>
        </div>
        <div class="rw-stats">
          <div class="rw-stat"><b id="rwEntries">0</b><small>Lottery entries</small></div>
          <div class="rw-stat"><b id="rwTrips">0</b><small>Trips completed</small></div>
        </div>

        <h3>Weekly challenge</h3>
        <div class="card challenge">
          <div class="ch-top"><span>Take 3 off-peak trips
            <span id="rwChallengeDone" style="display:none">&#x2705;</span></span>
            <b id="rwChallengeTxt">0/3</b></div>
          <div class="ch-bar"><div class="ch-fill" id="rwChallengeFill" style="width:0%"></div></div>
        </div>

        <h3>Badges</h3>
        <div class="badges" id="rwBadges"></div>

        <h3>Partner perks <span class="illus">illustrative</span></h3>
        <div id="rwPerks"></div>

        <h3>Prize draw</h3>
        <div class="card" id="rwLottery"></div>

        <div class="disc">All perks, badges and prize draws here are fictional and
          illustrative &mdash; session-only, with no real brands, money or prizes.</div>
      </section>
    </div>

    <div id="tabbar">
      <button class="tabbtn sel" id="tabTrip" onclick="go('plan')">
        <span class="ti">&#x1F9ED;</span>Trip</button>
      <button class="tabbtn" id="tabRewards" onclick="openRewards()">
        <span class="ti">&#x1F381;</span>Rewards</button>
    </div>

    <div id="confetti"></div>

    <div id="footer">Demo &mdash; simulated Berlin, synthetic conditions, illustrative
      rewards. No accounts, no network, no live data.</div>
  </div>
</div>

<script>
const DATA = __RIDER_DATA__;
const LM = DATA.landmarks;
const CHIPS = DATA.chips;
const ACCENT = DATA.accent;

const state = {
  from: LM[0].name,
  to: LM[3] ? LM[3].name : LM[1].name,
  chip: "now",
  wallet: 0,
  entry: null,
  pending: 0,
  pendingReward: null,
  pendingMeta: null,
  // session-only reward state (resets on reload, exactly right for a demo)
  entries: 0,            // lottery / prize-draw entries
  trips: 0,              // completed trips
  offpeak: 0,            // off-peak trips, for the weekly challenge
  badges: {first:false, green:false, peak:false},
};

function $(id){return document.getElementById(id);}

// ---------- screen transitions + tab-bar sync ----------
function setTab(name){
  const onRewards = (name === "rewards");
  $("tabRewards").classList.toggle("sel", onRewards);
  $("tabTrip").classList.toggle("sel", !onRewards);
}
function go(name){
  document.querySelectorAll(".screen").forEach(s=>s.classList.remove("active"));
  const el = $("s-"+name);
  el.classList.add("active");
  el.scrollTop = 0;
  setTab(name);
}

// ---------- status-bar clock ----------
function chipByID(id){return CHIPS.find(c=>c.id===id) || CHIPS[0];}
function refreshClock(){ $("clock").textContent = chipByID(state.chip).hhmm; }

// ---------- plan screen wiring ----------
function fillSelect(sel, selected){
  sel.innerHTML = "";
  LM.forEach(lm=>{
    const o = document.createElement("option");
    o.value = lm.name; o.textContent = lm.name;
    if(lm.name===selected) o.selected = true;
    sel.appendChild(o);
  });
}
function onFromChange(){
  state.from = $("fromSel").value;
  // never let From == To
  if(state.to === state.from){
    const alt = LM.find(l=>l.name!==state.from);
    state.to = alt.name;
    fillSelect($("toSel"), state.to);
  }
}
function buildChips(){
  const row = $("chipRow"); row.innerHTML = "";
  CHIPS.forEach(c=>{
    const d = document.createElement("div");
    d.className = "chip" + (c.id===state.chip ? " sel":"");
    d.innerHTML = c.label + "<small>"+c.hhmm+"</small>";
    d.onclick = ()=>{ state.chip = c.id;
      row.querySelectorAll(".chip").forEach(x=>x.classList.remove("sel"));
      d.classList.add("sel"); refreshClock(); };
    row.appendChild(d);
  });
}

// ---------- loading: "reading the network" with rotating tips/facts/jokes ----
const LOAD_MS = 3300;          // long enough to read a couple of cards
let loadCards = [];
function buildLoadCards(){
  const out = [];
  (DATA.tips  || []).forEach(t => out.push({k:"TIP",            cls:"",     t}));
  (DATA.facts || []).forEach(t => out.push({k:"DID YOU KNOW?",  cls:"fact", t}));
  (DATA.jokes || []).forEach(t => out.push({k:"JOKE",           cls:"joke", t}));
  // Fisher-Yates shuffle so the mix feels fresh each load.
  for(let i=out.length-1;i>0;i--){
    const j = Math.floor(Math.random()*(i+1));
    const tmp = out[i]; out[i] = out[j]; out[j] = tmp;
  }
  loadCards = out;
}
let loadTimer = null, loadIdx = 0;
function renderTip(){
  if(!loadCards.length) return;
  const card = loadCards[loadIdx % loadCards.length];
  $("tipLabel").className = "tiplabel " + card.cls;
  $("tipLabel").textContent = card.k;
  $("tipText").textContent = card.t;
}
function startLoading(){
  if(!loadCards.length) buildLoadCards();
  loadIdx = 0;
  $("tipCard").classList.remove("fade");
  renderTip();
  clearInterval(loadTimer);
  loadTimer = setInterval(()=>{
    const tc = $("tipCard");
    tc.classList.add("fade");                      // fade out current
    setTimeout(()=>{ loadIdx++; renderTip(); tc.classList.remove("fade"); }, 300);
  }, 1500);                                         // rotate every ~1.5s
}
function stopLoading(){ clearInterval(loadTimer); }

function findRoute(){
  state.from = $("fromSel").value;
  state.to = $("toSel").value;
  if(state.from === state.to){
    const alt = LM.find(l=>l.name!==state.from);
    state.to = alt.name; fillSelect($("toSel"), state.to);
  }
  refreshClock();
  go("loading"); startLoading();
  setTimeout(()=>{ stopLoading(); showVerdict(); }, LOAD_MS);
}

// ---------- verdict ----------
function pct(x){ return Math.round(x*100) + "%"; }

function showVerdict(){
  const key = state.from + "||" + state.to + "||" + state.chip;
  const e = DATA.dataset[key];
  state.entry = e;
  const route = e.route_corridors.join(" → ");
  let html = '<p class="route">' + e.from + " &rarr; " + e.to +
             " &middot; leaving <b>" + e.depart_hhmm + "</b></p>" +
             '<p class="route">via <b>' + route + "</b></p>";

  if(e.red_zone){
    $("verdictTitle").textContent = "Busy corridor in the peak";
    html += '<div class="banner red">Heads up: this trip rides a targeted ' +
      'corridor during the morning peak. Here are three nudges &mdash; each ' +
      'earns a reward.</div>';
    html += '<div class="card"><div class="opt-ty" style="font-weight:700">' +
      'Baseline (no change)</div><div class="desc" style="font-size:12.5px;color:var(--muted)">' +
      'Ride as planned at ' + e.depart_hhmm + ' on the busy corridor &mdash; no reward.</div></div>';
    html += "<h3>Pick a nudge</h3>";
    e.alternatives.forEach((a,i)=>{
      const r = a.reward;
      let pills = '<span class="pill green">⭐ ' + r.pulse_points + ' pts</span>';
      if(r.green_bonus>0) pills += '<span class="pill green">\u{1F331} +' + r.green_bonus + ' green</span>';
      if(r.lottery_entry) pills += '<span class="pill">\u{1F39F} lottery entry</span>';
      const benefit = "≈" + pct(a.crowding_reduction) + " less peak crowding &middot; ≈"
        + pct(a.carbon_reduction) + " lower carbon";
      html += '<div class="opt" onclick="choose(' + i + ')">' +
        '<div class="top"><span class="ty">' + altLabel(a.type) +
        '</span><span class="pts">+' + r.total_points + ' pts</span></div>' +
        '<div class="desc">' + a.description + '<br>' + benefit + '</div>' +
        '<div class="tagrow">' + pills + '</div></div>';
    });
  } else {
    $("verdictTitle").textContent = "You're already in the clear";
    const why = e.in_peak
      ? "Your route avoids the targeted corridors, so you're clear of the busy arterials."
      : "You're travelling off-peak, so the network's already quiet.";
    html += '<div class="banner ok">' + why +
      ' Confirm your trip and we&rsquo;ll still thank you with a small reward.</div>';
    html += '<div class="opt" onclick="choose(\'confirm\')">' +
      '<div class="top"><span class="ty">Confirm this trip</span>' +
      '<span class="pts">+' + DATA.confirmPoints + ' pts</span></div>' +
      '<div class="desc">No change needed &mdash; tap to confirm your low-impact trip.</div>' +
      '<div class="tagrow"><span class="pill green">⭐ ' + DATA.confirmPoints +
      ' pts</span><span class="pill">\u{1F39F} lottery entry</span></div></div>';
  }
  $("verdictBody").innerHTML = html;
  go("verdict");
}

function altLabel(t){
  return ({RETIME:"Leave earlier", REROUTE:"Take a quieter route",
           GREEN:"Travel in the green hour"})[t] || t;
}

// ---------- choose + map animation ----------
function project(lon, lat){
  let mnx=180, mxx=-180, mny=90, mxy=-90;
  LM.forEach(l=>{ mnx=Math.min(mnx,l.lon); mxx=Math.max(mxx,l.lon);
                  mny=Math.min(mny,l.lat); mxy=Math.max(mxy,l.lat); });
  const pad=28, W=300, H=200;
  const x = pad + (lon-mnx)/((mxx-mnx)||1)*(W-2*pad);
  const y = pad + (mxy-lat)/((mxy-mny)||1)*(H-2*pad); // flip lat
  return {x, y};
}
function lmByName(n){ return LM.find(l=>l.name===n); }

function choose(which){
  const e = state.entry;
  let reward, label, curveDir, type, offpeak, corridor, co2g;
  if(which==="confirm"){
    reward = {pulse_points:DATA.confirmPoints, green_bonus:0,
      total_points:DATA.confirmPoints, lottery_entry:true,
      partner_value:DATA.confirmPoints + " Pulse Points ≈ EUR " +
        (DATA.confirmPoints/100).toFixed(2) + " partner perk credit (illustrative)."};
    label = "Trip confirmed"; curveDir = 0;
    type = "CONFIRM"; offpeak = !e.in_peak; corridor = null; co2g = DATA.co2Confirm;
  } else {
    const a = e.alternatives[which];
    reward = a.reward; label = altLabel(a.type); type = a.type;
    curveDir = (a.type==="REROUTE") ? 1 : (a.type==="GREEN" ? -1 : 0);
    // RETIME and GREEN move the rider out of the peak; REROUTE keeps the time.
    offpeak = (a.type==="RETIME" || a.type==="GREEN");
    corridor = (e.targeted_corridors && e.targeted_corridors.length)
      ? e.targeted_corridors.join(" & ")
      : (e.route_corridors[0] || "the corridor");
    co2g = Math.round(a.carbon_reduction * DATA.co2Base);
  }
  state.pending = reward.total_points;
  state.pendingReward = reward;
  state.pendingMeta = {type, offpeak, corridor, co2g};

  $("chooseTitle").textContent = label;
  $("chooseSub").innerHTML = "Following your chosen route&hellip;";
  drawAndAnimate(e.from, e.to, curveDir, ()=> showArrival());
  go("choose");
}

function drawAndAnimate(fromName, toName, curveDir, onDone){
  const svg = $("map");
  const o = project(lmByName(fromName).lon, lmByName(fromName).lat);
  const d = project(lmByName(toName).lon, lmByName(toName).lat);
  // control point: midpoint pushed perpendicular for reroute/green curves
  const mx=(o.x+d.x)/2, my=(o.y+d.y)/2;
  const dx=d.x-o.x, dy=d.y-o.y, len=Math.hypot(dx,dy)||1;
  const off = curveDir*42;
  const cx = mx + (-dy/len)*off, cy = my + (dx/len)*off;

  // faint context dots for all landmarks
  let ctx = "";
  LM.forEach(l=>{ const p=project(l.lon,l.lat);
    ctx += '<circle cx="'+p.x+'" cy="'+p.y+'" r="2.4" fill="#2f3a45"/>'; });

  const pathD = "M "+o.x+" "+o.y+" Q "+cx+" "+cy+" "+d.x+" "+d.y;
  svg.innerHTML = ctx +
    '<path d="'+pathD+'" fill="none" stroke="'+ACCENT+'" stroke-width="3" ' +
      'stroke-dasharray="5 5" opacity="0.55"/>' +
    '<circle cx="'+o.x+'" cy="'+o.y+'" r="6" fill="#fff"/>' +
    '<circle cx="'+d.x+'" cy="'+d.y+'" r="6" fill="'+ACCENT+'"/>' +
    '<text x="'+o.x+'" y="'+(o.y-10)+'" fill="#e6edf3" font-size="9" ' +
      'text-anchor="middle">'+esc(fromName)+'</text>' +
    '<text x="'+d.x+'" y="'+(d.y+18)+'" fill="'+ACCENT+'" font-size="9" ' +
      'text-anchor="middle">'+esc(toName)+'</text>' +
    '<circle id="dot" r="6.5" fill="'+ACCENT+'" stroke="#fff" stroke-width="2"/>';

  const dot = $("dot");
  function bez(t){ const u=1-t;
    return { x:u*u*o.x + 2*u*t*cx + t*t*d.x, y:u*u*o.y + 2*u*t*cy + t*t*d.y }; }
  const DUR = 1700; let start=null, done=false;
  function step(ts){
    if(start===null) start=ts;
    let t = Math.min((ts-start)/DUR, 1);
    const p = bez(t); dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y);
    if(t<1){ requestAnimationFrame(step); }
    else if(!done){ done=true; setTimeout(onDone, 380); }
  }
  requestAnimationFrame(step);
}
function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// ---------- arrival: bank the reward, then celebrate ----------
function showArrival(){
  const r = state.pendingReward, m = state.pendingMeta || {};
  state.wallet += state.pending;
  state.trips += 1;
  if(r.lottery_entry) state.entries += 1;
  if(m.offpeak) state.offpeak += 1;
  // badges (session-only, illustrative)
  state.badges.first = true;                              // First Shift: any trip
  if(m.type === "GREEN") state.badges.green = true;       // Green Streak
  if(m.type === "RETIME" || (m.offpeak && m.type !== "CONFIRM"))
    state.badges.peak = true;                             // Peak Breaker

  $("walletPts").textContent = state.wallet;
  $("arrivePts").textContent = "+" + state.pending;
  $("arriveWallet").textContent = state.wallet;

  let rows = '<div class="reward-line"><span>Pulse Points</span><b>+' +
    r.pulse_points + '</b></div>';
  if(r.green_bonus>0) rows += '<div class="reward-line"><span>Green-hour bonus</span><b>+' +
    r.green_bonus + '</b></div>';
  rows += '<div class="reward-line"><span>Prize-draw entry</span><b>' +
    (r.lottery_entry ? "Yes \u{1F39F}" : "No") + '</b></div>';
  rows += '<div class="reward-line"><span>Total</span><b>+' + r.total_points +
    ' pts</b></div>';
  rows += '<div class="partner">' + r.partner_value + '</div>';
  $("arriveCard").innerHTML = rows;

  // animated lottery ticket
  const tk = $("arriveTicket");
  if(r.lottery_entry){
    tk.style.display = "flex";
    tk.innerHTML = '<div class="tk-ico">\u{1F39F}️</div><div>' +
      '<div class="tk-main">Prize-draw entry added</div>' +
      '<div class="tk-sub">You now hold ' + state.entries + ' entr' +
      (state.entries===1?"y":"ies") + ' &mdash; illustrative weekly draw</div></div>';
  } else { tk.style.display = "none"; }

  // illustrative corridor relief + CO2 line
  const co2 = $("arriveCo2");
  if(m.corridor){
    co2.innerHTML = 'Helped relieve <b>' + esc(m.corridor) + '</b><br>~' +
      m.co2g + ' g CO₂ saved <small>(illustrative)</small>';
  } else {
    co2.innerHTML = 'Kept an off-peak trip off-peak<br>~' + m.co2g +
      ' g CO₂ saved <small>(illustrative)</small>';
  }

  go("arrival");
  launchConfetti();
}

// ---------- confetti burst (pure JS, no libraries) ----------
function launchConfetti(){
  const host = $("confetti"); if(!host) return;
  host.innerHTML = "";
  const colors = [ACCENT,"#27d99a","#ffd166","#7ee0bd","#79c0ff","#ff8fa3"];
  for(let i=0;i<90;i++){
    const p = document.createElement("div");
    p.className = "confetti-pc";
    p.style.left = (Math.random()*100) + "%";
    p.style.background = colors[i % colors.length];
    p.style.width = (6 + Math.random()*5) + "px";
    p.style.height = (10 + Math.random()*8) + "px";
    p.style.animationDuration = (1.6 + Math.random()*1.4) + "s";
    p.style.animationDelay = (Math.random()*0.5) + "s";
    host.appendChild(p);
  }
  setTimeout(()=>{ host.innerHTML = ""; }, 2700);
}

// ---------- rewards screen ----------
function openRewards(){ renderRewards(); go("rewards"); }

function renderRewards(){
  $("rwWallet").textContent = state.wallet;
  $("rwEntries").textContent = state.entries;
  $("rwTrips").textContent = state.trips;

  // weekly challenge: take 3 off-peak trips
  const goal = 3, n = Math.min(state.offpeak, goal);
  $("rwChallengeTxt").textContent = state.offpeak + "/" + goal;
  $("rwChallengeFill").style.width = (n/goal*100) + "%";
  $("rwChallengeDone").style.display = (state.offpeak>=goal) ? "inline" : "none";

  // badges
  const defs = [
    {key:"first", ico:"\u{1F687}", label:"First Shift"},
    {key:"green", ico:"\u{1F331}", label:"Green Streak"},
    {key:"peak",  ico:"⚡",    label:"Peak Breaker"},
  ];
  $("rwBadges").innerHTML = defs.map(d =>
    '<div class="badge' + (state.badges[d.key] ? " earned" : "") + '">' +
    '<div class="bi">' + d.ico + '</div><div class="bl">' + d.label + '</div></div>'
  ).join("");

  // illustrative partner perks
  $("rwPerks").innerHTML = (DATA.perks || []).map(p => {
    const ok = state.wallet >= p.cost;
    const tag = ok
      ? '<span class="pk-state ok">Redeemable</span>'
      : '<span class="pk-state locked">' + (p.cost - state.wallet) + ' more</span>';
    return '<div class="perk"><div><div class="pk-name">' + esc(p.name) + '</div>' +
      '<div class="pk-cost">' + p.cost + ' pts</div></div>' + tag + '</div>';
  }).join("");

  // prize draw
  $("rwLottery").innerHTML = state.entries>0
    ? '\u{1F39F}️ <b>' + state.entries + '</b> entr' +
      (state.entries===1?"y":"ies") +
      ' in this week’s illustrative prize draw. Friday could be your day.'
    : 'No entries yet &mdash; complete a recommended shift to earn a prize-draw entry.';
}

// ---------- init ----------
fillSelect($("fromSel"), state.from);
fillSelect($("toSel"), state.to);
buildChips();
buildLoadCards();
refreshClock();
</script>
"""


with tab7:
    import json as _json
    import streamlit.components.v1 as _components

    import rider_engine

    st.title("Berlin Pulse Rider")
    st.caption(
        "A phone-style demo of the Section 5.3 five-step incentive logic from one "
        "rider's point of view: plan a morning trip, see if it lands on a busy "
        "targeted corridor in the peak, pick a nudge, and bank the reward. Every "
        "tap happens inside the demo — nothing is sent anywhere."
    )
    st.warning(
        "Demo only. Berlin is simulated, conditions are synthetic, and all rewards "
        "are illustrative figures from the paper's equations — not measured "
        "ridership, emissions, or real loyalty points. No network calls, no "
        "accounts, no language model."
    )

    # ---- STEP 2: precompute every trip outcome from rider_engine ----------------
    # For every ordered landmark pair and three representative departure chips we
    # call the pure engine (plan_trip -> generate_alternatives -> compute_reward)
    # and freeze the whole outcome into a JSON blob. The component never calls
    # back to Python: it just looks the answer up by "FROM||TO||CHIP".
    _corridor_index = rider_engine._load_corridor_index()

    # Three representative departure times. "Now / peak" sits inside the simulated
    # 07:30-09:00 morning peak (so a targeted route becomes a red zone); the other
    # two are off-peak, where the engine yields no nudge and we instead reward the
    # rider for confirming an already-quiet trip.
    RIDER_CHIPS = [
        {"id": "now",     "label": "Now / peak", "minute": 8 * 60 + 15},
        {"id": "earlier", "label": "Earlier",    "minute": 7 * 60},
        {"id": "later",   "label": "Later",      "minute": 10 * 60},
    ]
    for _c in RIDER_CHIPS:
        _c["hhmm"] = rider_engine._minute_to_hhmm(_c["minute"])

    # A small fixed reward for confirming an already-off-peak / off-corridor trip
    # (the engine offers no behavioural nudge there, but we still thank the rider).
    RIDER_CONFIRM_POINTS = 20

    _landmarks_payload = [
        {
            "name": lm["name"],
            "lon": lm["lon"],
            "lat": lm["lat"],
            "corridor": lm["corridor"],
            "targeted": bool(_corridor_index.get(lm["corridor"], {}).get("targeted", False)),
        }
        for lm in rider_engine.LANDMARKS
    ]
    _names = [lm["name"] for lm in rider_engine.LANDMARKS]

    _rider_dataset = {}
    for _o in _names:
        for _d in _names:
            if _o == _d:
                continue
            for _chip in RIDER_CHIPS:
                _trip = rider_engine.plan_trip(_o, _d, _chip["minute"])
                _alts_out = []
                for _alt in rider_engine.generate_alternatives(_trip):
                    _rw = rider_engine.compute_reward(_alt)
                    _alts_out.append({
                        "type": _alt["type"],
                        "description": _alt["description"],
                        "new_depart_hhmm": _alt["new_depart_hhmm"],
                        "route_corridors": _alt["route_corridors"],
                        "crowding_reduction": _alt["crowding_reduction"],
                        "carbon_reduction": _alt["carbon_reduction"],
                        "reward": {
                            "pulse_points": _rw["pulse_points"],
                            "green_bonus": _rw["green_bonus"],
                            "total_points": _rw["total_points"],
                            "lottery_entry": _rw["lottery_entry"],
                            "partner_value": _rw["partner_value"],
                        },
                    })
                _rider_dataset["%s||%s||%s" % (_o, _d, _chip["id"])] = {
                    "from": _o,
                    "to": _d,
                    "chip": _chip["id"],
                    "depart_hhmm": _trip["depart_hhmm"],
                    "in_peak": _trip["in_peak"],
                    "uses_targeted": _trip["uses_targeted"],
                    "red_zone": _trip["red_zone"],
                    "route_corridors": _trip["route_corridors"],
                    "targeted_corridors": _trip["targeted_corridors"],
                    "targeted_intensity": _trip["targeted_intensity"],
                    "alternatives": _alts_out,
                }

    # Loading-screen content (cycled every ~1.5s). All illustrative; FACTS are
    # rounded, public, non-operational figures used only to set the scene.
    RIDER_TIPS = [
        "Off-peak trips earn double Pulse Points.",
        "Green hour: travel when the grid is cleanest for a bonus.",
        "Small shifts add up. A 20-minute change can cool a whole corridor.",
        "Recommended shifts earn a draw entry. Friday could be your day.",
        "Targeted corridors relieve fast; the whole city moves slowly. "
        "That's the point.",
    ]
    RIDER_FACTS = [
        "Berlin drivers lose around 60 hours a year to congestion.",
        "Berlin's grid runs over three-fifths renewable in a typical year.",
        "Depot charging is the most controllable flexible load in the city.",
    ]
    RIDER_JOKES = [
        "Why did the U-Bahn get promoted? It was always on the right track.",
        "The bus tried meditation. Now it's much more centred at the stop.",
        "I told my route to relax. It said it couldn't, too much on its plate.",
        "Off-peak travel: because being early is the new being on time.",
    ]

    # Illustrative, fictional partner perks (no real brands, no real redemption).
    RIDER_PERKS = [
        {"name": "Example local cafe — free filter coffee", "cost": 100},
        {"name": "Example bike-share — day pass", "cost": 150},
        {"name": "Example cinema ticket", "cost": 250},
    ]

    # Illustrative CO2 conversion for the arrival card: grams "saved" =
    # carbon_reduction (engine fraction) * RIDER_CO2_BASE_G. NOT a measurement.
    RIDER_CO2_BASE_G = 1500       # GREEN(0.30)->450g, RETIME(0.08)->120g, etc.
    RIDER_CO2_CONFIRM_G = 60      # token figure for confirming an off-peak trip

    _rider_payload = {
        "accent": "#1d9e75",
        "landmarks": _landmarks_payload,
        "chips": RIDER_CHIPS,
        "dataset": _rider_dataset,
        "confirmPoints": RIDER_CONFIRM_POINTS,
        "tips": RIDER_TIPS,
        "facts": RIDER_FACTS,
        "jokes": RIDER_JOKES,
        "perks": RIDER_PERKS,
        "co2Base": RIDER_CO2_BASE_G,
        "co2Confirm": RIDER_CO2_CONFIRM_G,
    }

    # ---- STEP 3: the self-contained phone component -----------------------------
    _rider_html = _RIDER_COMPONENT_TEMPLATE.replace(
        "__RIDER_DATA__", _json.dumps(_rider_payload))
    _components.html(_rider_html, height=820, scrolling=False)

    st.caption(
        "Pulse Points live only in the demo's memory and reset when you reload — "
        "exactly right for a prototype. The trip outcomes and rewards shown here "
        "are the same numbers `rider_engine.py` produces and `test_rider.py` locks."
    )

    # ---- STEP 4: THE REVEAL -- "What if everyone shifted?" -----------------------
    # Zoom out from one rider to the whole city. The compliance slider (share of
    # riders who accept a recommended shift) maps to the scenario engine's
    # active_share; rider_engine.aggregate_effect rolls it up into the SAME
    # network/corridor figures the Scenario Explorer reports (registered share and
    # peak-shift stay at the Section 6 MEDIUM operating point, the aggregate_effect
    # defaults). We reuse the Tab 6 corridor geometry + redirection split so the
    # map recolours at the SAME operating point: targeted corridors visibly relieve
    # while the network barely moves -- the paper's core point.
    import pydeck as _pdk
    from prototype_engine import simulate_redirection as _simulate_redirection

    st.divider()
    st.header("What if everyone shifted?")
    st.caption(
        "The phone shows one rider. Drag the slider to roll many riders up into the "
        "validated Section 6 model and watch *where* the relief lands."
    )

    reveal_compliance_pct = st.slider(
        "Share of riders who accept a recommended shift (%)",
        0.0, 100.0, 45.0, step=1.0, key="reveal_compliance",
        help="Maps to the scenario engine's active_share. Registered share (20%) "
             "and peak-shift (7.5%) stay at the Section 6 MEDIUM operating point, "
             "exactly as rider_engine.aggregate_effect defaults.")
    reveal_compliance = reveal_compliance_pct / 100.0

    # Headline figures: rolled up by rider_engine.aggregate_effect -- reported
    # VERBATIM from scenario_engine.compute_scenario, so they match the Scenario
    # Explorer tab exactly at registered 20% / active <slider> / peak-shift 7.5%.
    reveal_agg = rider_engine.aggregate_effect(reveal_compliance)

    # Spatial split over the SAME Tab 6 corridors at the SAME operating point, so
    # the per-corridor colours reconcile with the headline above (both wrap
    # compute_scenario with identical inputs).
    reveal_sim = _simulate_redirection(
        registered_share=reveal_agg["registered_share"],
        active_share=reveal_compliance,
        peak_shift_share=reveal_agg["peak_shift_share"],
        corridor_share=reveal_agg["corridor_share"],
    )

    # Colour each corridor by its AFTER load on a fixed BEFORE scale (Tab 6 rule):
    # high load -> red, relieved -> green. Targeted arterials slide toward green;
    # the rest barely move.
    _reveal_max = max((c["before_peak"] for c in reveal_sim["per_corridor"]),
                      default=1.0) or 1.0

    def _reveal_color(load):
        t = min(max(load / _reveal_max, 0.0), 1.0)
        return [int(220 * t + 30), int(200 * (1.0 - t) + 30), 45]

    def _reveal_lines(coords):
        if not coords:
            return []
        first = coords[0]
        if (isinstance(first, (list, tuple)) and len(first) == 2
                and all(isinstance(v, (int, float)) for v in first)):
            return [coords]
        out = []
        for sub in coords:
            out.extend(_reveal_lines(sub))
        return out

    reveal_rows = []
    for c in reveal_sim["per_corridor"]:
        color = _reveal_color(c["after_peak"])
        for line in _reveal_lines(c["coords"]):
            reveal_rows.append({
                "name": c["name"],
                "path": [[float(x), float(y)] for x, y in line],
                "color": color,
                "before_peak": int(round(c["before_peak"])),
                "after_peak": int(round(c["after_peak"])),
                "relief_pct": round(c["relief_pct"], 1),
                "targeted": "targeted" if c["targeted"] else "non-targeted",
            })

    reveal_deck = _pdk.Deck(
        layers=[_pdk.Layer(
            "PathLayer", data=reveal_rows, get_path="path",
            get_color="color", get_width=30,
            width_min_pixels=3, width_max_pixels=9,
            pickable=True, auto_highlight=True,
        )],
        initial_view_state=_pdk.ViewState(
            latitude=52.52, longitude=13.405, zoom=11, pitch=0),
        map_provider="carto", map_style="road",
        tooltip={
            "html": "<b>{name}</b> ({targeted})<br/>"
                    "Before peak: {before_peak}<br/>After peak: {after_peak}<br/>"
                    "Relief: {relief_pct}%",
            "style": {"backgroundColor": "rgba(20,20,30,0.9)", "color": "white"},
        },
    )
    st.pydeck_chart(reveal_deck, use_container_width=True)

    rv1, rv2 = st.columns(2)
    rv1.metric("Network-wide peak reduction",
               f"{reveal_agg['network_peak_reduction_pct']:.2f} %")
    reveal_capped = (reveal_agg["corridor_relief_display_pct"]
                     < reveal_agg["corridor_relief_pct"])
    rv2.metric(
        "Corridor relief",
        f"{reveal_agg['corridor_relief_display_pct']:.2f} %",
        help=(f"Raw arithmetic value is {reveal_agg['corridor_relief_pct']:.2f} %; "
              f"capped to the paper's {SCENARIO_BAND[0]:.0f}–{SCENARIO_BAND[1]:.0f}% "
              "reported band for display, exactly as the Scenario Explorer tab does."
              if reveal_capped else None))

    st.caption(
        "This reuses the validated Section 6 model. Corridor effect is real; "
        "network effect is small."
    )
    st.caption(
        "Red = high peak load, green = relieved. Geometry is the real Berlin "
        "arterial network (the same corridors as the Network Prototype tab); the "
        "peak loads are synthetic. The headline figures come straight from "
        "`rider_engine.aggregate_effect`, which reports the Scenario Explorer's "
        "numbers verbatim — set that tab to registered 20%, active "
        f"{reveal_compliance_pct:.0f}%, peak-shift 7.5% to read the identical values."
    )

with tab8:
    from plotly.subplots import make_subplots as _make_subplots

    # ---- STEP 2: run the cross-grid comparison once per session -----------------
    @st.cache_data(show_spinner="Running the depot optimizer across five European grids...")
    def _load_crossgrid():
        return compare_grids()

    cg_results = _load_crossgrid()
    cg_by_code = {r["code"]: r for r in cg_results}

    # Fixed x-axis order required by the spec: DE, FR, PL, NO2, ES.
    CG_ORDER = ["DE", "FR", "PL", "NO2", "ES"]
    CG_WINDOW_HOURS = [22, 23, 0, 1, 2, 3, 4]
    cg_ordered = [cg_by_code[c] for c in CG_ORDER]

    # ---- STEP 3: intro ----------------------------------------------------------
    st.title("Cross-Grid Comparison")
    st.markdown(
        "This tab runs the **same validated depot-charging optimizer** — the one "
        "behind the Berlin result — on five real European grids (ENTSO-E "
        "Transparency Platform data, 2025), with **Germany as the paper's "
        "validated SMARD anchor**, to test whether the Berlin finding generalises. "
        "Results are reported as **percentage savings**, which are "
        "fleet-independent and therefore directly comparable across cities. Real "
        "data, real findings."
    )

    # ---- STEP 4: grouped savings bar chart --------------------------------------
    st.subheader("Carbon and cost savings by grid")
    cg_labels = [
        f"{r['label']} (anchor)" if r["code"] == "DE" else r["label"]
        for r in cg_ordered
    ]
    cg_carbon = [r["carbon_saving_pct"] for r in cg_ordered]
    cg_cost = [r["cost_saving_pct"] for r in cg_ordered]
    # Germany (the validated anchor) gets a distinct dark border on both bars.
    # marker.line takes per-bar arrays (one Line with list-valued color/width),
    # not a list of Line dicts.
    cg_border_color = [
        "rgb(20,20,20)" if r["code"] == "DE" else "rgba(0,0,0,0)"
        for r in cg_ordered
    ]
    cg_border_width = [3 if r["code"] == "DE" else 0 for r in cg_ordered]
    cg_border = dict(color=cg_border_color, width=cg_border_width)

    fig_cg = go.Figure()
    fig_cg.add_trace(go.Bar(
        x=cg_labels, y=cg_carbon, name="Carbon saving (%)",
        marker=dict(color="rgba(0,204,150,0.80)", line=cg_border),
        text=[f"{v:.2f}%" for v in cg_carbon], textposition="outside",
    ))
    fig_cg.add_trace(go.Bar(
        x=cg_labels, y=cg_cost, name="Cost saving (%)",
        marker=dict(color="rgba(99,110,250,0.80)", line=cg_border),
        text=[f"{v:.2f}%" for v in cg_cost], textposition="outside",
    ))
    fig_cg.update_layout(
        title="Percentage Savings — Same Optimizer, Five Grids "
              "(Germany = validated SMARD anchor)",
        xaxis_title="Grid",
        yaxis=dict(title="Saving vs naive charging (%)",
                   range=[0, max(cg_carbon + cg_cost) * 1.2]),
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
        margin=dict(t=60, b=60),
        height=440,
    )
    st.plotly_chart(fig_cg, use_container_width=True)
    st.caption(
        "Germany is outlined in black — it is the paper's validated SMARD anchor "
        "(2.39% carbon / 4.85% cost); the other four grids run the identical "
        "optimizer on ENTSO-E data."
    )

    # ---- STEP 5: overnight intensity small-multiples ----------------------------
    st.subheader("Why the percentages differ: overnight intensity profiles")
    cg_hour_labels = [f"{h:02d}:00" for h in CG_WINDOW_HOURS]
    fig_sm = _make_subplots(
        rows=1, cols=5, shared_yaxes=False,
        subplot_titles=[r["label"] for r in cg_ordered],
        horizontal_spacing=0.04,
    )
    for i, r in enumerate(cg_ordered, start=1):
        prof = r["overnight_profile"]
        yvals = [prof[h] for h in CG_WINDOW_HOURS]
        fig_sm.add_trace(go.Scatter(
            x=cg_hour_labels, y=yvals, mode="lines+markers",
            line=dict(color="rgb(99,110,250)", width=2),
            marker=dict(size=4), showlegend=False,
        ), row=1, col=i)
        fig_sm.update_xaxes(tickangle=-45, row=1, col=i)
    fig_sm.update_yaxes(title_text="g CO₂/kWh", row=1, col=1)
    fig_sm.update_layout(
        title="Average Overnight Carbon-Intensity Profile Across the Charging "
              "Window (22:00 → 04:00)",
        margin=dict(t=70, b=50),
        height=340,
    )
    st.plotly_chart(fig_sm, use_container_width=True)
    st.caption(
        "Each mini-chart shows that grid's mean carbon intensity across the "
        "charging-window hours 22, 23, 0, 1, 2, 3, 4. The carbon saving "
        "percentage tracks how much this profile *varies* overnight — flat "
        "profiles leave the optimizer little to exploit, regardless of how clean "
        "the grid is on average."
    )

    # ---- STEP 6: summary table --------------------------------------------------
    st.subheader("Summary table")
    cg_table = pd.DataFrame([
        {
            "Country": r["label"],
            "Mean Intensity (gCO2/kWh)": round(r["mean_intensity"], 2),
            "Carbon Saving (%)": round(r["carbon_saving_pct"], 2),
            "Cost Saving (%)": round(r["cost_saving_pct"], 2),
        }
        for r in cg_ordered
    ])
    st.dataframe(cg_table, hide_index=True, use_container_width=True)

    # ---- STEP 7: honest finding -------------------------------------------------
    st.info(
        "**What the real numbers show.**\n\n"
        "- **The scheduling method transfers everywhere on cost.** Every grid "
        "posts a positive cost saving, and it is substantial for four of the "
        "five — France 13.81%, Spain 10.50%, Germany 4.85%, Poland 3.11%. Norway "
        "is the lone exception at just 0.81%, because its near-total-hydro grid "
        "has an almost flat overnight price curve, leaving little to optimise. "
        "The day-ahead-price logic that worked in Berlin works across Europe.\n\n"
        "- **The carbon saving percentage measures overnight intensity "
        "*variability*, not average grid cleanliness.** France's high 12.97% "
        "comes from nuclear output ramping across the night (a swingy overnight "
        "profile), while Poland's low 1.16% comes from a coal-dominated grid that "
        "stays almost flat hour to hour — both follow from the shape of the "
        "profile, not from how clean or dirty the grid is.\n\n"
        "- **In absolute terms, clean grids avoid far less CO₂ per bus per "
        "year.** A large percentage on a clean grid is a large slice of a tiny "
        "number: Norway (mean ≈ 1.6 gCO₂/kWh) and France (≈ 17 gCO₂/kWh) avoid "
        "only a few kilograms of CO₂ per bus annually, whereas coal-heavy Poland "
        "(≈ 587 gCO₂/kWh) avoids far more real CO₂ per bus even though its "
        "percentage looks small.\n\n"
        "- **Spain data note (transparency).** The April 2025 Iberian Peninsula "
        "blackout left **35 hours of missing telemetry** in the Spain ENTSO-E "
        "series. Those hours were gap-filled using the adjacent valid generation "
        "mix (see `crossgrid_engine._repair_missing_hours`); it is documented "
        "here so the Spain figures are not mistaken for fully continuous data."
    )

    # ---- STEP 8: data attribution ----------------------------------------------
    st.caption(
        "Generation and price data: ENTSO-E Transparency Platform (2025). "
        "Germany: validated SMARD 2025 dataset (paper anchor, 2.3884% carbon / "
        "4.8482% cost saving)."
    )


# ============================================================================
# TAB 9 — Forecast Recovery (Appendix E)
# ============================================================================
with tab9:

    @st.cache_data(show_spinner="Computing forecast recovery (Appendix E)…")
    def _forecast_recovery():
        """Compute the Appendix E forecast-recovery result once and cache it.

        Reproduce-before-perturb lives inside forecast_engine: the five Appendix
        B bookends are reproduced and asserted before any forecast is built.
        """
        return forecast_engine.compute_forecast_recovery()

    fr = _forecast_recovery()
    bk = fr["bookends"]
    op = fr["operational"]
    nv = fr["naive"]

    # Pull the headline figures (recovery values are fractions -> percent).
    q4_oracle_pct = bk["carbon_oracle_q4_pct"]            # 3.76% — the carbon prize
    deployable_cost_pct = bk["deployable_cost_fy_pct"]    # 8.76% — the cost prize
    share_q4_recovery_pct = op["share"]["q4_recovery"] * 100.0   # 90.5%
    share_q4_saving_pct = op["share"]["q4_saving_pct"]           # 3.40%

    FR_ACCENT = "rgb(99,110,250)"          # app accent — operational forecasts
    FR_GREY = "rgba(150,150,150,0.75)"     # naive forecasts
    FR_NEUTRAL = "rgba(110,110,110,0.45)"  # bookends (blind / oracle)
    FR_NULL = "rgba(239,85,59,0.55)"       # random-order null

    # ---- Title + plain-English framing (Appendix E.5 order) --------------------
    st.title("Forecast Recovery")
    st.markdown(
        "**The big prize needs no forecast.** The deployable day-ahead-price "
        f"rule already captures a **{deployable_cost_pct:.2f}% cost saving** with "
        "zero foresight — it ranks overnight hours by published next-day prices, "
        "so nothing here has to be predicted. The **carbon** channel is the one "
        "where forecasting could help, and it is real but much smaller: with "
        f"perfect hindsight the most carbon a smarter night schedule can avoid is "
        f"only **{q4_oracle_pct:.2f}%** (Q4, out of sample). Against that small "
        f"prize, the operational *share* predictor recovers "
        f"**{share_q4_recovery_pct:.1f}%** of the oracle — which, because the "
        f"prize is small, is a realized carbon saving of just "
        f"**{share_q4_saving_pct:.2f}%**. This tab resolves the Appendix B "
        "question of whether the carbon gap can be closed with a real forecast: "
        "mostly yes in *relative* terms, but the absolute carbon prize stays "
        "small, while the cost prize — which needs no forecast at all — remains "
        "the larger result."
    )

    # ---- STEP 1: recovery ladder ----------------------------------------------
    st.subheader("Recovery ladder — how much of the carbon oracle each method recovers (Q4)")

    ladder = [
        ("Deployable blind",       bk["blind_recovery"] * 100.0,                FR_NEUTRAL),
        ("Persistence",            nv["persistence"]["q4_recovery"] * 100.0,    FR_GREY),
        ("Climatology",            nv["climatology"]["q4_recovery"] * 100.0,    FR_GREY),
        ("Random null",            fr["random_null"]["q4_recovery"] * 100.0,    FR_NULL),
        ("Share predictor",        op["share"]["q4_recovery"] * 100.0,          FR_ACCENT),
        ("Direct predictor",       op["direct"]["q4_recovery"] * 100.0,         FR_ACCENT),
        ("Merit-order predictor",  op["merit_order"]["q4_recovery"] * 100.0,    FR_ACCENT),
        ("Oracle",                 bk["oracle_recovery"] * 100.0,               FR_NEUTRAL),
    ]
    ladder_x = [name for name, _, _ in ladder]
    ladder_y = [val for _, val, _ in ladder]
    ladder_c = [col for _, _, col in ladder]
    null_pct = fr["random_null"]["q4_recovery"] * 100.0

    fig_ladder = go.Figure()
    # Shaded band for the random-order null floor (chance recovery).
    fig_ladder.add_hrect(
        y0=min(0.0, null_pct), y1=max(0.0, null_pct),
        fillcolor="rgba(239,85,59,0.10)", line_width=0, layer="below",
        annotation_text="random-order null floor", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="rgba(239,85,59,0.9)",
    )
    fig_ladder.add_trace(go.Bar(
        x=ladder_x, y=ladder_y, marker_color=ladder_c,
        text=[f"{v:.1f}%" for v in ladder_y], textposition="outside",
        hovertemplate="%{x}<br>Q4 recovery: %{y:.1f}%<extra></extra>",
    ))
    fig_ladder.update_layout(
        xaxis_title="Forecast method",
        yaxis=dict(title="Q4 recovery of carbon oracle (%)",
                   range=[min(0.0, null_pct) - 8, max(ladder_y) * 1.18]),
        margin=dict(t=30, b=80),
        height=460,
        showlegend=False,
    )
    st.plotly_chart(fig_ladder, use_container_width=True)
    st.caption(
        "Grey = naive forecasts · accent = operational forecasts · neutral = "
        "blind/oracle bookends · the red bar and shaded band mark the random-"
        "order null floor (what a coin-flip schedule recovers by chance)."
    )

    # ---- STEP 2: headline metrics ---------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Q4 carbon oracle (the prize)", f"{q4_oracle_pct:.2f} %",
              help="Most carbon a perfect-foresight night schedule can avoid, "
                   "Q4 out of sample. This is the ceiling forecasts chase.")
    m2.metric("Share predictor — Q4 recovery", f"{share_q4_recovery_pct:.1f} %",
              help="Share of the carbon oracle recovered by the operational "
                   "renewable-share predictor (relative).")
    m3.metric("Share predictor — realized saving", f"{share_q4_saving_pct:.2f} %",
              help="The recovery applied to the small prize: actual Q4 carbon "
                   "saving vs naive Strategy A.")
    m4.metric("Deployable cost saving", f"{deployable_cost_pct:.2f} %",
              help="Full-year cost saving from the deployable day-ahead-price "
                   "rule — needs no forecast at all (the larger prize).")

    # ---- STEP 3: full method table --------------------------------------------
    st.subheader("All forecast methods")
    rows = []
    method_labels = [
        ("persistence", "Persistence (naive)", nv),
        ("climatology", "Climatology (naive)", nv),
        ("share", "Share predictor (operational)", op),
        ("direct", "Direct predictor (operational)", op),
        ("merit_order", "Merit-order predictor (operational)", op),
    ]
    for key, label, src in method_labels:
        m = src[key]
        rows.append({
            "Forecast method": label,
            "Q4 recovery": f"{m['q4_recovery'] * 100.0:.1f}%",
            "Full-year recovery": f"{m['fy_recovery'] * 100.0:.1f}%",
            "Within-night rank corr (Q4)": f"{m['rank_corr']:.3f}",
        })
    st.table(pd.DataFrame(rows).set_index("Forecast method"))
    st.caption(
        "Recovery = realized carbon saving under the forecast schedule ÷ saving "
        "under the perfect-foresight oracle, on the same night set. Rank "
        "correlation is the within-night Spearman of forecast vs realized "
        "intensity, averaged over Q4 nights."
    )

    # ---- STEP 4: scope notes ---------------------------------------------------
    with st.expander("What this means"):
        st.markdown(
            f"- **The big prize is cost, and it needs no forecast.** The "
            f"deployable day-ahead-price rule already books a "
            f"{deployable_cost_pct:.2f}% cost saving using only published "
            f"next-day prices — no prediction of the future is required. "
            f"Forecasting changes nothing about this larger result.\n\n"
            f"- **The carbon prize is real but small, and it is what foresight "
            f"buys.** Even perfect hindsight avoids only {q4_oracle_pct:.2f}% "
            f"more carbon (Q4). So the {share_q4_recovery_pct:.1f}% the share "
            f"predictor recovers is an impressive slice of a small pie — a "
            f"realized carbon saving of {share_q4_saving_pct:.2f}%. A large "
            f"recovery percentage is not a large amount of carbon.\n\n"
            f"- **What is being measured is within-night ranking, out of "
            f"sample.** The metric scores how well a forecast orders the hours "
            f"*inside each Q4 night* against realized intensity — never how it "
            f"chooses which nights to charge. Naive forecasts barely clear the "
            f"random-order null floor; the operational forecasts recover most "
            f"of the (small) carbon prize, which is the honest answer to the "
            f"Appendix B gap."
        )

    # ---- STEP 5: pointer ------------------------------------------------------
    st.markdown(
        "← See **Deployability Gap** tab for the Appendix B result this resolves."
    )
