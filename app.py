import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from engine import run_simulation, run_simulation_deployable, run_sensitivity

# ── Colour palette ───────────────────────────────────────────────────────────
TEAL = "#1d9e75"
TEAL_DARK = "#0f6e56"
CORAL = "#d85a30"
AMBER = "#ef9f27"
BLUE = "#378add"

st.set_page_config(page_title="Berlin Pulse Depot Charging Optimizer", layout="wide")

# ── Global CSS: fixed light theme on top of Streamlit's light mode ────────────
st.markdown("""
<style>
/* ── Section headers ────────────────────────────────────────── */
.bp-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.6rem 0 1rem 0;
}
.bp-section-header span {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    white-space: nowrap;
}
.bp-section-header hr {
    flex: 1;
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 0;
}

/* ── Oracle warning bar ─────────────────────────────────────── */
.bp-oracle-warning {
    background: rgba(239, 159, 39, 0.10);
    border-left: 4px solid """ + AMBER + """;
    border-radius: 6px;
    padding: 10px 16px;
    color: #b47a1a;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Pill toggle (style native Streamlit radio) ────────────── */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    gap: 8px;
    flex-direction: row !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label {
    padding: 6px 18px !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1.5px solid #d1d5db !important;
    color: #6b7280 !important;
    background: transparent !important;
    white-space: nowrap;
    margin: 0 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    border-color: """ + TEAL + """ !important;
    color: """ + TEAL + """ !important;
    background: rgba(29, 158, 117, 0.08) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {
    display: none;
}
div[data-testid="stRadio"] > label {
    display: none;
}

/* ── Sidebar styling ────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    background-color: """ + TEAL_DARK + """ !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: """ + TEAL + """ !important;
}
.bp-sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 4px;
    margin-top: 6px;
    padding-top: 18px;
    border-top: 1px solid #e5e7eb;
}
.bp-sidebar-label-first {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 2px;
}
.bp-sidebar-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 18px 0 14px 0;
}

/* ── Title accent ──────────────────────────────────────────── */
h1 {
    border-left: 4px solid """ + TEAL + """;
    padding-left: 14px !important;
}

/* ── Chart container borders ───────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border: 0.5px solid rgba(0,0,0,0.08);
    border-radius: 8px;
    overflow: hidden;
}

/* ── Tab strip & panel borders ─────────────────────────────── */
div[data-testid="stTabs"] {
    border: 0.5px solid rgba(0,0,0,0.08);
    border-radius: 8px;
    overflow: hidden;
    padding: 0 8px 8px 8px;
}
div[data-testid="stTabs"] > div[role="tablist"] {
    border-bottom: 1px solid #e5e7eb;
}

/* ── About expander interior ────────────────────────────────── */
div[data-testid="stExpander"] details {
    border-color: #e5e7eb !important;
}
div[data-testid="stExpander"] .stMarkdown {
    font-size: 0.88rem;
    line-height: 1.65;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)

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

# ── Mode toggle (pill buttons) ───────────────────────────────────────────────
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
    st.markdown('<p class="bp-sidebar-label-first">Fleet parameters</p>', unsafe_allow_html=True)
    n_buses = st.number_input("Fleet size (buses)", min_value=1, value=277, step=1)
    kwh_per_bus = st.number_input(
        "Energy per bus per night (kWh)", min_value=1.0, value=240.0, step=10.0,
        help="Total energy each bus needs to charge overnight.",
    )
    st.markdown('<p class="bp-sidebar-label">Charger settings</p>', unsafe_allow_html=True)
    charger_kw = st.number_input(
        "Charger power (kW)", min_value=1.0, value=50.0, step=5.0,
        help="Rated power of each charger. Determines how many hours are needed per bus.",
    )
    st.markdown('<p class="bp-sidebar-label">Window</p>', unsafe_allow_html=True)
    window_label = st.selectbox("Charging window", list(WINDOW_PRESETS.keys()))
    st.markdown('<hr class="bp-sidebar-divider">', unsafe_allow_html=True)
    run_btn = st.button("Run", type="primary", use_container_width=True)

# ── Input validation ─────────────────────────────────────────────────────────

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


# ── Helper: animated metric cards via custom HTML component ──────────────────

def render_metric_cards(cards, footnote=""):
    """Render a row of metric cards with fade-in / slide-up / count-up animations.

    Each card dict: {label, value, suffix, subtitle, color, prefix}
    """
    card_html_parts = []
    for i, c in enumerate(cards):
        delay = i * 120
        accent = c.get("color", TEAL)
        prefix = c.get("prefix", "")
        suffix = c.get("suffix", "")
        # Determine numeric value for count-up
        raw = c["value"]
        # Format: we pass the numeric value as data attr, display formatted
        try:
            num_val = float(str(raw).replace(",", "").replace("€", "").replace("%", "").strip())
        except (ValueError, TypeError):
            num_val = 0

        decimals = 0
        if "." in str(raw):
            decimals = len(str(raw).split(".")[-1])

        card_html_parts.append(f"""
        <div class="mc" style="--accent:{accent}; animation-delay:{delay}ms">
            <div class="mc-label">{c['label']}</div>
            <div class="mc-value">
                <span class="mc-prefix">{prefix}</span><span class="mc-num" data-target="{num_val}" data-decimals="{decimals}">0</span><span class="mc-suffix">{suffix}</span>
            </div>
            <div class="mc-subtitle">{c.get('subtitle', '')}</div>
        </div>
        """)

    cards_grid = "\n".join(card_html_parts)
    footnote_html = f'<div class="mc-footnote">{footnote}</div>' if footnote else ""

    html = f"""
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .mc-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            padding: 4px 0;
        }}
        .mc {{
            background: #ffffff;
            border: 0.5px solid rgba(0,0,0,0.08);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 16px 20px;
            opacity: 0;
            transform: translateY(12px);
            animation: fadeSlide 0.5s ease-out forwards;
        }}
        @keyframes fadeSlide {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .mc-label {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 6px;
        }}
        .mc-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
        }}
        .mc-prefix, .mc-suffix {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #6b7280;
        }}
        .mc-subtitle {{
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 4px;
        }}
        .mc-footnote {{
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 10px;
            padding-left: 2px;
        }}
    </style>
    <div class="mc-grid">
        {cards_grid}
    </div>
    {footnote_html}
    <script>
        function easeOut(t) {{ return 1 - Math.pow(1 - t, 3); }}
        function countUp(el) {{
            const target = parseFloat(el.dataset.target);
            const decimals = parseInt(el.dataset.decimals) || 0;
            const duration = 800;
            const start = performance.now();
            function tick(now) {{
                const t = Math.min((now - start) / duration, 1);
                const val = target * easeOut(t);
                el.textContent = val.toLocaleString('en-US', {{
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals
                }});
                if (t < 1) requestAnimationFrame(tick);
            }}
            requestAnimationFrame(tick);
        }}
        document.querySelectorAll('.mc-num').forEach(function(el, i) {{
            setTimeout(function() {{ countUp(el); }}, i * 120);
        }});
    </script>
    """
    n_cards = len(cards)
    height = 160 if n_cards <= 3 else 320
    if footnote:
        height += 30
    components.html(html, height=height)


def section_header(text):
    st.markdown(
        f'<div class="bp-section-header"><span>{text}</span><hr></div>',
        unsafe_allow_html=True,
    )


# ── Plotly theme helper ──────────────────────────────────────────────────────

def apply_chart_theme(fig, title="", xaxis_title="", yaxis_title="", **extra):
    defaults = dict(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color="#111827"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            yanchor="top", y=0.99, xanchor="left", x=0.01,
        ),
        margin=dict(t=40, b=40),
        height=380,
    )
    defaults.update(extra)
    fig.update_layout(**defaults)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")


# ── Main results ─────────────────────────────────────────────────────────────

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

    # ── Headline results ─────────────────────────────────────────────────────
    section_header("Headline Results")

    if is_oracle:
        st.markdown(
            '<div class="bp-oracle-warning">'
            '<strong>Upper bound only.</strong> Theoretical maximum assuming perfect '
            'foresight; the deployable carbon saving is much smaller.'
            '</div>',
            unsafe_allow_html=True,
        )

    carbon_label = "CO₂ saved (upper bound)" if is_oracle else "CO₂ saved (deployable)"
    mode_note = "oracle / perfect foresight" if is_oracle else "deployable / frozen day-ahead ranking"

    render_metric_cards([
        dict(label=carbon_label, value=f"{res['carbon_saving_pct']:.2f}",
             suffix=" %", color=TEAL,
             subtitle="Percentage reduction in fleet carbon emissions"),
        dict(label="Cost saved", value=f"{res['cost_saving_pct']:.2f}",
             suffix=" %", color=AMBER,
             subtitle="Percentage reduction in fleet electricity cost"),
    ])

    render_metric_cards([
        dict(label="Fleet CO₂ saved / year",
             value=f"{res['fleet_co2_saved_tonnes']:.1f}",
             suffix=" tonnes", color=TEAL,
             subtitle="Absolute annual carbon reduction"),
        dict(label="Fleet cost saved / year",
             value=f"{res['fleet_cost_saved_eur']:,.0f}",
             prefix="€ ", color=AMBER,
             subtitle="Absolute annual cost reduction"),
    ])

    render_metric_cards([
        dict(label="CO₂ saved per bus / year",
             value=f"{res['per_bus']['co2_saved_kg']:.1f}",
             suffix=" kg", color=BLUE,
             subtitle="Carbon saving per individual bus"),
        dict(label="Cost saved per bus / year",
             value=f"{res['per_bus']['cost_saved_eur']:.2f}",
             prefix="€ ", color=BLUE,
             subtitle="Cost saving per individual bus"),
    ], footnote=f"Based on {res['n_nights']} complete nights, window {window_label}, mode: {mode_note}.")

    # ── Chart tabs ─────────────────────────────────────────────────────────────
    tab_carbon, tab_night, tab_cumul, tab_sens = st.tabs([
        "Carbon Profile", "Night Schedule", "Cumulative CO₂", "Sensitivity",
    ])

    # ── Tab 1: Carbon-intensity profile ─────────────────────────────────────
    with tab_carbon:
        section_header("Grid Carbon-Intensity Profile")
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
            fill="tonexty", fillcolor="rgba(29,158,117,0.15)",
            name="10th–90th percentile",
        ))
        fig1.add_trace(go.Scatter(
            x=hour_labels, y=means, mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(color=TEAL, size=6),
            name="Nightly mean",
        ))
        apply_chart_theme(fig1,
            title="Mean Grid Carbon Intensity Across Charging Window",
            xaxis_title="Hour (Berlin time)",
            yaxis_title="Production intensity (g CO₂ / kWh)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Tab 2: Example night schedule ────────────────────────────────────────
    with tab_night:
        strategy_label = "Carbon-optimal oracle" if is_oracle else "Deployable (frozen ranking)"
        section_header(f"Example Night — Naive vs {strategy_label}")
        ex = res["example_night"]
        ex_hours = ex["hours"]
        ex_labels = [f"{h:02d}:00" for h in ex_hours]
        n_hours = len(ex_hours)

        naive_charging = [1 if i in ex["a_slots"] else 0 for i in range(n_hours)]
        optimal_charging = [1 if i in ex["b_slots"] else 0 for i in range(n_hours)]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=ex_labels, y=[v * 0.9 for v in naive_charging],
            name="Naive (Strategy A)", marker_color="rgba(216,90,48,0.75)",
            width=0.35, offset=-0.2,
        ))
        fig2.add_trace(go.Bar(
            x=ex_labels, y=[v * 0.9 for v in optimal_charging],
            name=f"{strategy_label} (Strategy B)", marker_color="rgba(29,158,117,0.75)",
            width=0.35, offset=0.15,
        ))
        fig2.add_trace(go.Scatter(
            x=ex_labels, y=ex["intensity"], mode="lines+markers",
            name="Intensity (g CO₂/kWh)", yaxis="y2",
            line=dict(color=TEAL, width=2, dash="dot"),
            marker=dict(size=6, color=TEAL),
        ))
        apply_chart_theme(fig2,
            title=f"Charging Schedule — Night of {ex['night_id']} (median saving)",
            xaxis_title="Hour (Berlin time)",
            yaxis=dict(title="Charging (on / off)", range=[0, 1.1],
                       tickvals=[0, 1], ticktext=["Off", "On"]),
            yaxis2=dict(title="Intensity (g CO₂ / kWh)", overlaying="y",
                        side="right"),
            barmode="group",
            legend=dict(yanchor="top", y=1.12, xanchor="left", x=0.0,
                        orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Cumulative CO₂ saved ──────────────────────────────────────────
    with tab_cumul:
        section_header("Cumulative CO₂ Saved Across the Year")
        if is_oracle:
            st.markdown(
                '<div class="bp-oracle-warning">'
                'Theoretical maximum assuming perfect foresight; '
                'the deployable carbon saving is much smaller.'
                '</div>',
                unsafe_allow_html=True,
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
            line=dict(color=TEAL, width=2),
            fillcolor="rgba(29,158,117,0.12)",
            name="Cumulative CO₂ saved",
        ))
        apply_chart_theme(fig3,
            title=f"Cumulative Fleet CO₂ Saved — {'Oracle (upper bound)' if is_oracle else 'Deployable'}",
            xaxis_title="Date",
            yaxis_title="Cumulative CO₂ saved (tonnes)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: Sensitivity tornado ───────────────────────────────────────────
    with tab_sens:
        section_header("Sensitivity Analysis")
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

        tornado_colors = {
            "carbon_pcts": (TEAL, "CO₂ Saving (%)"),
            "cost_pcts": (AMBER, "Cost Saving (%)"),
        }

        for metric_key, (bar_color, metric_title) in tornado_colors.items():
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
                marker_color=bar_color,
                opacity=0.7,
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
                marker=dict(symbol="diamond", size=10, color=CORAL,
                            line=dict(width=1, color="white")),
                name="Default",
                hovertext=[f"Default: {b['default_val']:.3f}%" for b in bars],
                hoverinfo="text",
            ))

            apply_chart_theme(fig_t,
                title=f"Tornado — {metric_title}",
                xaxis_title=metric_title,
                yaxis=dict(autorange="reversed"),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99,
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=40, b=40, l=160),
                height=300,
            )
            st.plotly_chart(fig_t, use_container_width=True)
