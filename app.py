import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import run_simulation, run_simulation_deployable, run_sensitivity, load_dataset
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Depot Optimizer", "Scenario Explorer", "Deployability Gap",
    "Robustness (Monte Carlo)", "Unified Model", "Network Prototype",
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
the sidebar, then press **Run** to see the headline savings, intensity profiles,
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
