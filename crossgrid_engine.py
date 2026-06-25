"""Tab 8 runtime engine: cross-grid comparison.

Runs the Appendix D depot-charging simulation across five European grids and
returns per-grid savings, mean carbon intensity, and an overnight intensity
profile. The physics (intensity, charging window, Strategy A/B, annual
simulation) is REUSED from baseline_simulation -- nothing is reimplemented here.

Germany comes from the committed validated dataset (already carries the
intensity / window / night columns). The four ENTSO-E grids come from the
small normalized parquets under crossgrid_data/ and are passed through
add_intensity / add_window_and_nights so they match the German frame's schema.

Light libraries only (pandas, numpy + baseline_simulation).
"""

import os

import pandas as pd

from baseline_simulation import (
    add_intensity,
    add_window_and_nights,
    simulate_year,
    GEN_TYPES,
    WINDOW_HOURS,
)

DE_DATASET = os.path.join("data", "berlin_pulse_validated_dataset.parquet")
CROSSGRID_DIR = "crossgrid_data"

# Fixed spectrum order: Germany (validated anchor) first, then the four
# ENTSO-E grids FR, PL, NO2, ES.
GRIDS = [
    {"code": "DE", "label": "Germany"},
    {"code": "FR", "label": "France"},
    {"code": "PL", "label": "Poland"},
    {"code": "NO2", "label": "Norway"},
    {"code": "ES", "label": "Spain"},
]

_LABELS = {g["code"]: g["label"] for g in GRIDS}


def _repair_missing_hours(df):
    """Carry the adjacent valid hour's generation mix into missing-telemetry hours.

    A handful of hours in the ENTSO-E exports report zero across all 12 fuels
    -- these are missing-data hours (e.g. France 2025-12-10 gaps, Spain's
    2025-04-28/29 Iberian blackout), not genuine zero-output hours. Their
    carbon intensity (sum(gen*factor) / total_generation) is 0/0 = NaN, which
    would propagate through the carbon-optimal schedule. We forward/back-fill
    the full generation block from the nearest valid hour so total_generation
    stays positive and intensity stays defined. Returns the repaired frame.
    """
    df = df.copy()
    zero_total = df[GEN_TYPES].sum(axis=1) == 0
    if zero_total.any():
        df.loc[zero_total, GEN_TYPES] = pd.NA
        df[GEN_TYPES] = df[GEN_TYPES].ffill().bfill()
    return df


def load_grid(code):
    """Return a frame ready for simulate_year, indexed on the local timestamp.

    Germany is read from the validated dataset and kept as-is. The four
    ENTSO-E grids are read from crossgrid_data/<code>.parquet, have any
    missing-telemetry hours repaired, then run through add_intensity +
    add_window_and_nights to derive the simulation columns.
    """
    if code == "DE":
        df = pd.read_parquet(DE_DATASET)
        return df.set_index("timestamp_berlin")

    df = pd.read_parquet(os.path.join(CROSSGRID_DIR, f"{code}.parquet"))
    df = df.set_index("timestamp_local")
    df = _repair_missing_hours(df)
    df = add_intensity(df)
    df = add_window_and_nights(df)
    return df


def _overnight_profile(df):
    """Mean production_intensity per window clock-hour over all in-window rows."""
    w = df[df["in_window"]]
    hour = w.index.hour
    profile = {}
    for h in WINDOW_HOURS:
        vals = w["production_intensity"][hour == h]
        profile[h] = float(vals.mean())
    return profile


def run_grid(code):
    """Run the simulation for one grid and return its summary metrics."""
    df = load_grid(code)
    res = simulate_year(df)
    return {
        "carbon_saving_pct": res["carbon_saving_pct"],
        "cost_saving_pct": res["cost_saving_pct"],
        "mean_intensity": float(df["production_intensity"].mean()),
        "overnight_profile": _overnight_profile(df),
    }


def compare_grids():
    """Run all five grids in spectrum order; return a list of result dicts."""
    out = []
    for g in GRIDS:
        code = g["code"]
        res = run_grid(code)
        res["code"] = code
        res["label"] = _LABELS[code]
        out.append(res)
    return out


if __name__ == "__main__":
    for r in compare_grids():
        print(
            f"{r['code']:<4} {r['label']:<8} "
            f"carbon={r['carbon_saving_pct']:.4f}%  "
            f"cost={r['cost_saving_pct']:.4f}%  "
            f"mean_intensity={r['mean_intensity']:.1f}"
        )
