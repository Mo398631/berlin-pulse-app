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

# --- reused, unmodified simulation logic --------------------------------------
from baseline_simulation import (
    simulate_year,           # annual A-vs-B accumulator (reused as-is)
    complete_nights,         # data-edge-night exclusion (reused as-is)
    slot_energies,           # per-bus kWh slot split (reused as-is)
    strategy_A,              # naive charging (reused as-is)
    strategy_B,              # carbon-optimal charging (reused as-is)
    WINDOW_HOURS,            # default 22:00-05:00 clock order [22,23,0,1,2,3,4]
    DEFAULTS,
)

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
    for night_id, g in w.groupby("night_id"):
        g = g.sort_index()
        e = g["production_intensity"].to_numpy()
        p = g["dayahead_price"].to_numpy()
        aC, aK = strategy_A(e, p, slots)
        bC, bK = strategy_B(e, p, slots)
        A_co2 += aC; B_co2 += bC
        A_cost += aK; B_cost += bK
        per_night.append({
            "night_id": night_id,
            "a_co2_kg": aC / 1000.0,            # per bus
            "b_co2_kg": bC / 1000.0,
            "co2_saved_kg": (aC - bC) / 1000.0,
            "a_cost_eur": aK,                   # per bus
            "b_cost_eur": bK,
            "cost_saved_eur": aK - bK,
        })

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
    }


if __name__ == "__main__":
    res = run_simulation()
    print(f"Nights: {res['n_nights']}  slots(kWh): {res['slots_kwh']}")
    print(f"Carbon saving: {res['carbon_saving_pct']:.4f}%")
    print(f"Cost saving:   {res['cost_saving_pct']:.4f}%")
