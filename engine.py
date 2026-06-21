"""
Streamlit-facing engine for the Berlin Pulse Depot Charging Optimizer.

This module is a thin wrapper. It does NOT re-implement any simulation logic:
the Strategy A (naive) and Strategy B (carbon-optimal, perfect foresight)
primitives, the slot-energy split, the complete-night selection and the annual
accumulation are all imported UNCHANGED from baseline_simulation.py and
gate.py's proven data path (the validated parquet in data/).

Public API:

    run_simulation(n_buses, kwh_per_bus, charger_kw, window_hours) -> dict

The default case (277 buses, 240 kWh/bus, 50 kW, window 22:00->05:00) reuses
the parquet's pre-computed in_window / night_id columns, so the headline
numbers are bit-identical to gate.py / simulate_year and reproduce the
manuscript bookends (~2.40% carbon, ~4.88% cost).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- reused, unmodified simulation logic --------------------------------------
from baseline_simulation import (
    simulate_year,           # annual A-vs-B accumulator (reused as-is)
    complete_nights,         # data-edge-night exclusion (reused as-is)
    slot_energies,           # per-bus kWh slot split (reused as-is)
    strategy_A,              # naive charging (reused as-is)
    strategy_B,              # carbon-optimal charging (reused as-is)
    _evaluate,              # order-based evaluation (reused from gate.py pattern)
    WINDOW_HOURS,            # default 22:00-05:00 clock order [22,23,0,1,2,3,4]
    DEFAULTS,
)
from gate import frozen_orders, night_sets, order_from_clock_priority, CLOCK

DATA_DIR = Path(__file__).resolve().parent / "data"
PARQUET = DATA_DIR / "berlin_pulse_validated_dataset.parquet"


# ---- window handling ---------------------------------------------------------

def _expand_window(window_hours):
    """Normalise the window argument to a clock-ordered list of start-hours.

    Accepted forms:
      * (start, end) tuple  -> e.g. (22, 5) means 22:00 inclusive .. 05:00
        exclusive, i.e. start-hours [22, 23, 0, 1, 2, 3, 4]. Wraps midnight
        when start > end.
      * an explicit iterable of hour integers -> used verbatim, e.g.
        [22, 23, 0, 1, 2, 3, 4].

    The default (22, 5) reproduces baseline_simulation.WINDOW_HOURS exactly.
    """
    if isinstance(window_hours, tuple) and len(window_hours) == 2:
        start, end = int(window_hours[0]), int(window_hours[1])
        if start <= end:
            hours = list(range(start, end))          # same-day window
        else:
            hours = list(range(start, 24)) + list(range(0, end))  # wraps midnight
        return hours
    # explicit list/range of hours
    return [int(h) for h in window_hours]


def _apply_window(df, window):
    """Recompute in_window / night_id for a custom window.

    Generalises baseline_simulation.add_window_and_nights: post-midnight
    in-window hours (those numerically below the window start) belong to the
    previous civil date. For the default window this is identical to the
    parquet's pre-computed columns, so this path is only taken for non-default
    windows.
    """
    out = df.copy()
    hr = out.index.hour
    out["in_window"] = np.isin(hr, window)
    start = window[0]
    local_date = pd.Series(out.index.date, index=out.index)
    shift_back = pd.Series(np.where(hr < start, 1, 0), index=out.index)
    night = pd.to_datetime(local_date) - pd.to_timedelta(shift_back, unit="D")
    nid = night.dt.strftime("%Y-%m-%d")
    out["night_id"] = np.where(out["in_window"], nid.values, None)
    return out


# ---- data loading ------------------------------------------------------------

@st.cache_data
def load_dataset():
    """Load the validated hourly dataset (Europe/Berlin indexed)."""
    df = pd.read_parquet(PARQUET).set_index("timestamp_berlin")
    return df.sort_index()


# ---- main entry point --------------------------------------------------------

def run_simulation(n_buses=DEFAULTS["n_buses"],
                   kwh_per_bus=DEFAULTS["kwh_per_bus"],
                   charger_kw=DEFAULTS["charger_kw"],
                   window_hours=(22, 5)):
    """Run Strategy A vs Strategy B over all complete nights and return results.

    Parameters
    ----------
    n_buses : int          Fleet size (scales absolute savings; pct invariant).
    kwh_per_bus : float    Nightly energy required per bus.
    charger_kw : float     Charger power; sets the slot split via slot_energies.
    window_hours : tuple   Charging window as (start_hour, end_hour), default
                           (22, 5) == 22:00->05:00. Also accepts an explicit
                           list of start-hours.

    Returns
    -------
    dict with carbon_saving_pct, cost_saving_pct, fleet_co2_saved_tonnes,
    fleet_cost_saved_eur, per-bus figures, and per-night detail for charts.
    """
    window = _expand_window(window_hours)
    df = load_dataset()

    # Default window -> reuse the parquet's gate-proven in_window/night_id
    # columns unchanged. Custom window -> re-derive membership only.
    if window == list(WINDOW_HOURS):
        work = df
    else:
        work = _apply_window(df, window)

    slots = slot_energies(kwh_per_bus, charger_kw)
    nights = complete_nights(work)
    w = work[work["in_window"] & work["night_id"].isin(nights)]

    # Per-night accumulation, mirroring simulate_year's loop exactly but also
    # recording each night so the caller can plot time series. Strategy A and B
    # are the imported, unmodified primitives.
    A_co2 = B_co2 = A_cost = B_cost = 0.0
    per_night = []
    intensity_by_hour = {h: [] for h in window}
    example_night_detail = None
    n_slots = len(slots)

    for night_id, g in w.groupby("night_id"):
        g = g.sort_index()
        e = g["production_intensity"].to_numpy()
        p = g["dayahead_price"].to_numpy()
        hours = list(g.index.hour)
        aC, aK = strategy_A(e, p, slots)
        bC, bK = strategy_B(e, p, slots)
        A_co2 += aC; B_co2 += bC
        A_cost += aK; B_cost += bK

        for i, h in enumerate(hours):
            if h in intensity_by_hour:
                intensity_by_hour[h].append(e[i])

        per_night.append({
            "night_id": night_id,
            "a_co2_kg": aC / 1000.0,
            "b_co2_kg": bC / 1000.0,
            "co2_saved_kg": (aC - bC) / 1000.0,
            "a_cost_eur": aK,
            "b_cost_eur": bK,
            "cost_saved_eur": aK - bK,
            "hours": hours,
            "intensity": e.tolist(),
        })

    median_idx = int(np.argsort([r["co2_saved_kg"] for r in per_night])[len(per_night) // 2])
    ex = per_night[median_idx]
    ex_e = np.array(ex["intensity"])
    order_a = list(range(len(ex_e)))
    order_b = list(np.argsort(ex_e, kind="stable"))
    example_night_detail = {
        "night_id": ex["night_id"],
        "hours": ex["hours"],
        "intensity": ex["intensity"],
        "a_slots": order_a[:n_slots],
        "b_slots": order_b[:n_slots],
    }

    intensity_profile = {}
    for h in window:
        vals = np.array(intensity_by_hour[h])
        if len(vals) > 0:
            intensity_profile[h] = {
                "mean": float(vals.mean()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
        else:
            intensity_profile[h] = {"mean": 0, "min": 0, "max": 0, "p10": 0, "p90": 0}

    carbon_saving_pct = (A_co2 - B_co2) / A_co2 * 100.0
    cost_saving_pct = (A_cost - B_cost) / A_cost * 100.0

    # Internal guard: the headline numbers must equal simulate_year's, proving
    # the per-night loop did not alter the reused logic.
    ref = simulate_year(work, n_buses=n_buses,
                        kwh_per_bus=kwh_per_bus, charger_kw=charger_kw)
    assert abs(carbon_saving_pct - ref["carbon_saving_pct"]) < 1e-9, \
        (carbon_saving_pct, ref["carbon_saving_pct"])
    assert abs(cost_saving_pct - ref["cost_saving_pct"]) < 1e-9, \
        (cost_saving_pct, ref["cost_saving_pct"])

    return {
        "inputs": {
            "n_buses": n_buses,
            "kwh_per_bus": kwh_per_bus,
            "charger_kw": charger_kw,
            "window_hours": window,
        },
        "n_nights": len(nights),
        "slots_kwh": slots,
        "carbon_saving_pct": carbon_saving_pct,
        "cost_saving_pct": cost_saving_pct,
        "fleet_co2_saved_tonnes": (A_co2 - B_co2) * n_buses / 1e6,   # g -> t
        "fleet_cost_saved_eur": (A_cost - B_cost) * n_buses,
        "per_bus": {
            "a_co2_kg": A_co2 / 1000.0,
            "b_co2_kg": B_co2 / 1000.0,
            "co2_saved_kg": (A_co2 - B_co2) / 1000.0,
            "a_cost_eur": A_cost,
            "b_cost_eur": B_cost,
            "cost_saved_eur": A_cost - B_cost,
        },
        "per_night": per_night,
        "intensity_profile": intensity_profile,
        "example_night": example_night_detail,
    }


def run_simulation_deployable(n_buses=DEFAULTS["n_buses"],
                              kwh_per_bus=DEFAULTS["kwh_per_bus"],
                              charger_kw=DEFAULTS["charger_kw"],
                              window_hours=(22, 5)):
    """Run deployable (day-ahead) simulation using frozen training-window ranking.

    Uses the same approach as gate.py: learn the average carbon-intensity rank
    of each clock-hour from Jan-Sep (training window), then apply that fixed
    order to all nights. This represents what is achievable with only day-ahead
    information — no per-night perfect foresight.
    """
    window = _expand_window(window_hours)
    df = load_dataset()

    if window == list(WINDOW_HOURS):
        work = df
    else:
        work = _apply_window(df, window)

    slots = slot_energies(kwh_per_bus, charger_kw)
    nights = complete_nights(work)
    _, train, _ = night_sets(work)
    carbon_clocks, price_clocks, _, _ = frozen_orders(work, train)
    w = work[work["in_window"] & work["night_id"].isin(nights)]

    A_co2 = D_co2 = A_cost = D_cost = 0.0
    per_night = []
    intensity_by_hour = {h: [] for h in window}
    n_slots = len(slots)

    for night_id, g in w.groupby("night_id"):
        g = g.sort_index()
        e = g["production_intensity"].to_numpy()
        p = g["dayahead_price"].to_numpy()
        hours = list(g.index.hour)
        aC, aK = strategy_A(e, p, slots)
        order = order_from_clock_priority(hours, carbon_clocks)
        dC, dK = _evaluate(e, p, slots, order)
        A_co2 += aC; D_co2 += dC
        A_cost += aK; D_cost += dK

        for i, h in enumerate(hours):
            if h in intensity_by_hour:
                intensity_by_hour[h].append(e[i])

        per_night.append({
            "night_id": night_id,
            "a_co2_kg": aC / 1000.0,
            "b_co2_kg": dC / 1000.0,
            "co2_saved_kg": (aC - dC) / 1000.0,
            "a_cost_eur": aK,
            "b_cost_eur": dK,
            "cost_saved_eur": aK - dK,
            "hours": hours,
            "intensity": e.tolist(),
        })

    median_idx = int(np.argsort([r["co2_saved_kg"] for r in per_night])[len(per_night) // 2])
    ex = per_night[median_idx]
    ex_e = np.array(ex["intensity"])
    order_a = list(range(len(ex_e)))
    order_d = order_from_clock_priority(ex["hours"], carbon_clocks)
    example_night_detail = {
        "night_id": ex["night_id"],
        "hours": ex["hours"],
        "intensity": ex["intensity"],
        "a_slots": order_a[:n_slots],
        "b_slots": order_d[:n_slots],
    }

    intensity_profile = {}
    for h in window:
        vals = np.array(intensity_by_hour[h])
        if len(vals) > 0:
            intensity_profile[h] = {
                "mean": float(vals.mean()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
        else:
            intensity_profile[h] = {"mean": 0, "min": 0, "max": 0, "p10": 0, "p90": 0}

    carbon_saving_pct = (A_co2 - D_co2) / A_co2 * 100.0 if A_co2 != 0 else 0.0
    cost_saving_pct = (A_cost - D_cost) / A_cost * 100.0 if A_cost != 0 else 0.0

    return {
        "inputs": {
            "n_buses": n_buses,
            "kwh_per_bus": kwh_per_bus,
            "charger_kw": charger_kw,
            "window_hours": window,
        },
        "n_nights": len(nights),
        "slots_kwh": slots,
        "carbon_saving_pct": carbon_saving_pct,
        "cost_saving_pct": cost_saving_pct,
        "fleet_co2_saved_tonnes": (A_co2 - D_co2) * n_buses / 1e6,
        "fleet_cost_saved_eur": (A_cost - D_cost) * n_buses,
        "per_bus": {
            "a_co2_kg": A_co2 / 1000.0,
            "b_co2_kg": D_co2 / 1000.0,
            "co2_saved_kg": (A_co2 - D_co2) / 1000.0,
            "a_cost_eur": A_cost,
            "b_cost_eur": D_cost,
            "cost_saved_eur": A_cost - D_cost,
        },
        "per_night": per_night,
        "intensity_profile": intensity_profile,
        "example_night": example_night_detail,
    }


def _quick_sim(work, kwh_per_bus, charger_kw, is_oracle):
    """Fast inner loop: returns (carbon_saving_pct, cost_saving_pct).

    Skips per-night detail, intensity profiles, and example-night selection
    to keep the sensitivity sweep fast.
    """
    slots = slot_energies(kwh_per_bus, charger_kw)
    nights = complete_nights(work)
    w = work[work["in_window"] & work["night_id"].isin(nights)]

    if not is_oracle:
        _, train, _ = night_sets(work)
        carbon_clocks, _, _, _ = frozen_orders(work, train)

    A_co2 = S_co2 = A_cost = S_cost = 0.0
    for _, g in w.groupby("night_id"):
        g = g.sort_index()
        e = g["production_intensity"].to_numpy()
        p = g["dayahead_price"].to_numpy()
        hours = list(g.index.hour)
        night_slots = slots[:len(e)]
        aC, aK = strategy_A(e, p, night_slots)
        if is_oracle:
            sC, sK = strategy_B(e, p, night_slots)
        else:
            order = order_from_clock_priority(hours, carbon_clocks)
            sC, sK = _evaluate(e, p, night_slots, order)
        A_co2 += aC; S_co2 += sC
        A_cost += aK; S_cost += sK

    carbon_pct = (A_co2 - S_co2) / A_co2 * 100.0 if A_co2 else 0.0
    cost_pct = (A_cost - S_cost) / A_cost * 100.0 if A_cost else 0.0
    return carbon_pct, cost_pct


def run_sensitivity(is_oracle=True, base_kwh=240.0, base_kw=50.0,
                    base_window=(22, 5), base_n_buses=277):
    """Sweep one parameter at a time around defaults; return tornado data.

    Each parameter is varied across a small range while the others stay at
    their base values.  Fleet size is included for completeness (pct savings
    are fleet-invariant, so the bar will be zero-width).

    Returns a list of dicts, one per parameter, each with:
        label, values, default, carbon_pcts, cost_pcts
    """
    df = load_dataset()
    base_window_list = _expand_window(base_window)

    if base_window_list == list(WINDOW_HOURS):
        base_work = df
    else:
        base_work = _apply_window(df, base_window_list)

    sweeps = []

    # 1. Charger power (kW)
    kw_vals = [30.0, 40.0, 50.0, 75.0, 100.0]
    carbon_kw, cost_kw = [], []
    for kw in kw_vals:
        c, k = _quick_sim(base_work, base_kwh, kw, is_oracle)
        carbon_kw.append(c); cost_kw.append(k)
    sweeps.append(dict(label="Charger power (kW)", values=kw_vals,
                       default=base_kw, carbon_pcts=carbon_kw, cost_pcts=cost_kw))

    # 2. Energy per bus (kWh)
    kwh_vals = [150.0, 200.0, 240.0, 300.0, 350.0]
    carbon_kwh, cost_kwh = [], []
    for kwh in kwh_vals:
        c, k = _quick_sim(base_work, kwh, base_kw, is_oracle)
        carbon_kwh.append(c); cost_kwh.append(k)
    sweeps.append(dict(label="Energy per bus (kWh)", values=kwh_vals,
                       default=base_kwh, carbon_pcts=carbon_kwh, cost_pcts=cost_kwh))

    # 3. Charging window length (hours)
    window_specs = [
        ((23, 4), "5 h"),
        ((23, 5), "6 h"),
        ((22, 5), "7 h"),
        ((22, 6), "8 h"),
        ((21, 6), "9 h"),
    ]
    carbon_win, cost_win = [], []
    win_labels = []
    default_win_label = None
    for spec, lbl in window_specs:
        wl = _expand_window(spec)
        if wl == list(WINDOW_HOURS):
            w_work = df
        else:
            w_work = _apply_window(df, wl)
        c, k = _quick_sim(w_work, base_kwh, base_kw, is_oracle)
        carbon_win.append(c); cost_win.append(k)
        win_labels.append(lbl)
        if wl == base_window_list:
            default_win_label = lbl
    sweeps.append(dict(label="Window length", values=win_labels,
                       default=default_win_label or "7 h",
                       carbon_pcts=carbon_win, cost_pcts=cost_win))

    # 4. Fleet size (pct savings invariant — included to show that)
    bus_vals = [50, 150, 277, 400, 600]
    c_base, k_base = _quick_sim(base_work, base_kwh, base_kw, is_oracle)
    sweeps.append(dict(label="Fleet size (buses)", values=bus_vals,
                       default=base_n_buses,
                       carbon_pcts=[c_base] * len(bus_vals),
                       cost_pcts=[k_base] * len(bus_vals)))

    return sweeps


if __name__ == "__main__":
    res = run_simulation()
    print(f"Nights: {res['n_nights']}  slots(kWh): {res['slots_kwh']}")
    print(f"Carbon saving: {res['carbon_saving_pct']:.4f}%")
    print(f"Cost saving:   {res['cost_saving_pct']:.4f}%")
