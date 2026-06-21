"""
Berlin Pulse, Appendix D, Chat 1 baseline.
Depot-charging simulation: Strategy A (naive) vs Strategy B (carbon-optimal,
perfect foresight), reproducing the Section 6.5 / Appendix B baseline.

The Strategy A and Strategy B functions are factored out so later chats
(Monte Carlo, consumption-based) can import them without rewriting the
simulation logic:

    from baseline_simulation import strategy_A, strategy_B, simulate_year

Data convention (verified against Appendix B Table B.1 to the decimal):
  - SMARD English CSV "Start date" is Europe/Berlin local civil time.
  - Semicolon-delimited, German number format (',' thousands, '.' decimal),
    missing values as '-'.
  - Localized to Europe/Berlin with DST handling: the spring-forward 02:00
    hour is absent; the fall-back 02:00 hour appears twice (first=DST,
    second=standard).
  - Charging window 22:00-05:00 -> in-window start-hours {22,23,0,1,2,3,4}.
  - Nights indexed by the civil date on which 22:00 falls; early-morning
    hours (<=4) belong to the previous civil date.
  - The two data-edge nights (2024-12-31 morning-only, 2025-12-31 truncated)
    are dropped, leaving 364 complete nights.
"""

import numpy as np
import pandas as pd

# ---- Configuration -----------------------------------------------------------

EMISSION_FACTORS = {            # g CO2 / kWh, direct combustion
    "lignite": 1054.0,
    "hard_coal": 884.0,
    "fossil_gas": 401.0,
    "other_conventional": 700.0,
    "biomass": 0.0,
    "hydropower": 0.0,
    "wind_offshore": 0.0,
    "wind_onshore": 0.0,
    "photovoltaics": 0.0,
    "other_renewable": 0.0,
    "nuclear": 0.0,
    "pumped_storage": 0.0,
}

GEN_COLUMN_MAP = {
    "Biomass [MWh] Calculated resolutions": "biomass",
    "Hydropower [MWh] Calculated resolutions": "hydropower",
    "Wind offshore [MWh] Calculated resolutions": "wind_offshore",
    "Wind onshore [MWh] Calculated resolutions": "wind_onshore",
    "Photovoltaics [MWh] Calculated resolutions": "photovoltaics",
    "Other renewable [MWh] Calculated resolutions": "other_renewable",
    "Nuclear [MWh] Calculated resolutions": "nuclear",
    "Lignite [MWh] Calculated resolutions": "lignite",
    "Hard coal [MWh] Calculated resolutions": "hard_coal",
    "Fossil gas [MWh] Calculated resolutions": "fossil_gas",
    "Hydro pumped storage [MWh] Calculated resolutions": "pumped_storage",
    "Other conventional [MWh] Calculated resolutions": "other_conventional",
}
GEN_TYPES = list(GEN_COLUMN_MAP.values())
PRICE_COLUMN = "Germany/Luxembourg [\u20ac/MWh] Calculated resolutions"

WINDOW_HOURS = [22, 23, 0, 1, 2, 3, 4]   # 22:00-05:00 inclusive of the 04:00-05:00 hour
EDGE_NIGHTS = {"2024-12-31", "2025-12-31"}

DEFAULTS = dict(n_buses=277, kwh_per_bus=240.0, charger_kw=50.0)


# ---- Loading -----------------------------------------------------------------

def _de_num(series):
    """Parse SMARD German-formatted numbers; '-' and '' -> NaN."""
    return pd.to_numeric(
        series.astype(str).str.strip()
              .replace({"-": np.nan, "": np.nan})
              .str.replace(",", "", regex=False),
        errors="coerce",
    )


def _localize_berlin(start_strings):
    """Localize naive 'Start date' strings to Europe/Berlin with DST handling."""
    naive = pd.to_datetime(start_strings, format="%b %d, %Y %I:%M %p")
    ambiguous = np.ones(len(naive), dtype=bool)          # first fall-back hour = DST
    ambiguous[naive.duplicated(keep="first").values] = False  # second = standard
    return pd.DatetimeIndex(naive).tz_localize(
        "Europe/Berlin", ambiguous=ambiguous, nonexistent="shift_forward"
    )


def load_smard(generation_csv, prices_csv):
    """Load and align the two SMARD CSVs into one hourly Europe/Berlin frame."""
    gen = pd.read_csv(generation_csv, sep=";", dtype=str, encoding="utf-8-sig")
    prc = pd.read_csv(prices_csv, sep=";", dtype=str, encoding="utf-8-sig")

    for raw, clean in GEN_COLUMN_MAP.items():
        gen[clean] = _de_num(gen[raw]).fillna(0.0)
    prc_price = _de_num(prc[PRICE_COLUMN])

    gen_idx = _localize_berlin(gen["Start date"])
    prc_idx = _localize_berlin(prc["Start date"])

    g = gen[GEN_TYPES].copy()
    g.index = gen_idx
    p = pd.DataFrame({"dayahead_price": prc_price.values}, index=prc_idx)

    df = g.join(p, how="outer").sort_index()
    return df


def gap_report(df):
    """Return a dict describing gaps and duplicate hours on a UTC continuous grid."""
    utc = df.index.tz_convert("UTC")
    full = pd.date_range(utc.min(), utc.max(), freq="h", tz="UTC")
    return {
        "n_rows": len(df),
        "expected_hours": len(full),
        "missing_hours": len(full.difference(utc)),
        "duplicate_utc_hours": int(utc.duplicated(keep=False).sum()),
        "nan_prices": int(df["dayahead_price"].isna().sum()),
    }


# ---- Intensity & night structure --------------------------------------------

def add_intensity(df, factors=EMISSION_FACTORS):
    df = df.copy()
    df[GEN_TYPES] = df[GEN_TYPES].fillna(0.0)
    df["total_generation"] = df[GEN_TYPES].sum(axis=1)
    weighted = sum(df[g] * factors[g] for g in GEN_TYPES)
    df["production_intensity"] = weighted / df["total_generation"]   # g CO2 / kWh
    return df


def add_window_and_nights(df):
    df = df.copy()
    hr = df.index.hour
    df["in_window"] = np.isin(hr, WINDOW_HOURS)
    local_date = pd.Series(df.index.date, index=df.index)
    shift_back = pd.Series(np.where(hr <= 4, 1, 0), index=df.index)
    night = pd.to_datetime(local_date) - pd.to_timedelta(shift_back, unit="D")
    nid = night.dt.strftime("%Y-%m-%d")
    df["night_id"] = np.where(df["in_window"], nid.values, None)
    return df


def complete_nights(df):
    """List of night_ids that are complete (i.e. not data-edge nights)."""
    w = df[df["in_window"]]
    nights = [n for n in pd.unique(w["night_id"].dropna()) if n not in EDGE_NIGHTS]
    return sorted(nights)


# ---- Charging strategies -----------------------------------------------------

def slot_energies(kwh_per_bus=DEFAULTS["kwh_per_bus"], charger_kw=DEFAULTS["charger_kw"]):
    """Per-bus energy delivered in each used in-window hour, in priority order.

    240 kWh at 50 kW -> four full 50 kWh hours + one 40 kWh partial hour.
    """
    full = int(kwh_per_bus // charger_kw)
    remainder = kwh_per_bus - full * charger_kw
    slots = [charger_kw] * full
    if remainder > 1e-9:
        slots.append(remainder)
    return slots


def _evaluate(intensity, price, slots, order):
    """Place each slot k into hour order[k]; return (gCO2, EUR) for one bus-night."""
    co2_g = 0.0
    cost_eur = 0.0
    for k, e_kwh in enumerate(slots):
        h = order[k]
        co2_g += e_kwh * intensity[h]            # kWh * g/kWh = g CO2
        cost_eur += (e_kwh / 1000.0) * price[h]  # MWh * EUR/MWh = EUR
    return co2_g, cost_eur


def strategy_A(intensity, price, slots):
    """Naive: charge from window open, earliest in-window hours first."""
    order = list(range(len(intensity)))
    return _evaluate(intensity, price, slots, order)


def strategy_B(intensity, price, slots):
    """Carbon-optimal, perfect foresight: fill the cleanest in-window hours.

    Ranks the night's hours by that night's realised carbon intensity.
    Identical total energy to Strategy A by construction.
    """
    order = list(np.argsort(intensity, kind="stable"))
    return _evaluate(intensity, price, slots, order)


# ---- Annual simulation -------------------------------------------------------

def simulate_year(df, intensity_col="production_intensity",
                  n_buses=DEFAULTS["n_buses"],
                  kwh_per_bus=DEFAULTS["kwh_per_bus"],
                  charger_kw=DEFAULTS["charger_kw"]):
    """Run Strategy A and B over all complete nights; return a results dict.

    Percentage savings are fleet-size invariant; absolute savings scale by n_buses.
    """
    slots = slot_energies(kwh_per_bus, charger_kw)
    nights = complete_nights(df)
    w = df[df["in_window"] & df["night_id"].isin(nights)]

    A_co2 = B_co2 = A_cost = B_cost = 0.0
    for _, g in w.groupby("night_id"):
        g = g.sort_index()
        e = g[intensity_col].to_numpy()
        p = g["dayahead_price"].to_numpy()
        aC, aK = strategy_A(e, p, slots)
        bC, bK = strategy_B(e, p, slots)
        A_co2 += aC; B_co2 += bC
        A_cost += aK; B_cost += bK

    carbon_saving_pct = (A_co2 - B_co2) / A_co2 * 100.0
    cost_saving_pct = (A_cost - B_cost) / A_cost * 100.0

    return {
        "n_nights": len(nights),
        "slots_kwh": slots,
        "carbon_saving_pct": carbon_saving_pct,
        "cost_saving_pct": cost_saving_pct,
        "fleet_co2_saved_tonnes": (A_co2 - B_co2) * n_buses / 1e6,  # g -> t
        "fleet_cost_saved_eur": (A_cost - B_cost) * n_buses,
        "per_bus_A_co2_kg": A_co2 / 1000.0,
        "per_bus_B_co2_kg": B_co2 / 1000.0,
        "per_bus_A_cost_eur": A_cost,
        "per_bus_B_cost_eur": B_cost,
    }


def build_validated_dataset(df):
    """Assemble the deliverable frame with the required columns."""
    out = df.copy()
    out.index.name = "timestamp_berlin"
    out = out.reset_index()
    cols = (["timestamp_berlin"] + GEN_TYPES +
            ["total_generation", "production_intensity",
             "dayahead_price", "in_window", "night_id"])
    return out[cols]


if __name__ == "__main__":
    import sys
    gen_csv = sys.argv[1] if len(sys.argv) > 1 else "smard_generation_2025.csv"
    prc_csv = sys.argv[2] if len(sys.argv) > 2 else "smard_dayahead_prices_2025.csv"

    df = load_smard(gen_csv, prc_csv)
    print("Gap report:", gap_report(df))
    df = add_intensity(df)
    df = add_window_and_nights(df)
    res = simulate_year(df)
    print(f"Complete nights: {res['n_nights']}")
    print(f"Carbon saving: {res['carbon_saving_pct']:.4f}%  (target 2.402 / headline 2.41)")
    print(f"Cost saving:   {res['cost_saving_pct']:.4f}%  (target 4.877 / headline 4.85)")
    print(f"Fleet CO2 saved: {res['fleet_co2_saved_tonnes']:.1f} t/yr (manuscript ~215.5)")
    print(f"Fleet cost saved: {res['fleet_cost_saved_eur']:.0f} EUR/yr (manuscript ~106,385)")
