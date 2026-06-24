"""
Pure-logic engine for the Berlin Pulse map prototype (the sixth Streamlit tab).

This module turns the Section 6 congestion scenario into a *spatial* picture: it
spreads the network-wide peak-shift across the named arterial corridors drawn on
the map, so the user can see WHERE the relief lands. Like engine.py and
scenario_engine.py it is a *pure* module -- the only I/O is reading the small,
already-processed prototype_data/corridors.geojson with the standard-library
`json` reader (NO osmnx / geopandas at runtime; those are dev-only, see
build_prototype_data.py). Everything else is light arithmetic over numpy arrays,
so test_prototype.py can lock the numbers the way test_engine.py locks the
optimizer.

RECONCILIATION with scenario_engine (the binding contract)
----------------------------------------------------------
The aggregate figures this engine reports are taken *directly* from
scenario_engine.compute_scenario, so the map can never drift away from the
headline scenario tab:

    network_peak_reduction_pct  <- compute_scenario(...)["network_peak_reduction_pct"]
    corridor_relief_pct         <- compute_scenario(...)["corridor_relief_pct"]

The per-corridor effects are then chosen so their energy-weighted (peak-load
weighted) aggregate reduction reproduces network_peak_reduction_pct *exactly*.

Concentration. The paper's corridor relief is the network shift concentrated by
the factor 1/corridor_share onto the targeted arterials (scenario_engine
docstring). We keep that RATIO: each targeted corridor relieves 1/corridor_share
times as much (per peak trip) as a non-targeted one. We deliberately do NOT
paint the headline corridor_relief_pct onto every targeted corridor: in the
synthetic data the targeted arterials carry far more than corridor_share (~0.25)
of peak trips, so doing that would break demand conservation. Preserving the
concentration ratio and the exact network aggregate is the honest reconciliation;
the headline corridor_relief_pct remains available as the (stylized 25%-share)
aggregate figure.

Conservation. Trips are retimed / rerouted, not deleted. We track total_before
and total_after PEAK load and return them; the difference is absorbed off-peak
(see `note`). No corridor's after_peak can go negative.

All `*_share` arguments are fractions in [0, 1]; every *_pct figure is a
percentage. These are illustrative what-if figures driven by the paper's
equations over SYNTHETIC corridor demand, NOT forecasts.

Public API:

    load_corridors(path=None) -> list[dict]
    simulate_redirection(registered_share, active_share, peak_shift_share,
                         corridor_share=0.25, path=None) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scenario_engine

# Default location of the processed, committed corridor geometry.
HERE = Path(__file__).resolve().parent
DEFAULT_CORRIDORS_PATH = HERE / "prototype_data" / "corridors.geojson"

# Off-peak is where the shifted peak trips go; surfaced in the returned `note`.
CONSERVATION_NOTE = (
    "Peak trips are retimed/rerouted, not deleted: total_before - total_after "
    "is absorbed off-peak, so network demand is conserved."
)


def load_corridors(path=None):
    """Read prototype_data/corridors.geojson with the stdlib json reader.

    Returns a list of corridors, each a dict with:
        name          : str   corridor display name
        coordinates   : list  raw GeoJSON geometry coordinates (lon/lat)
        baseline_peak : int    synthetic peak-hour trip count (seeded, see note)
        targeted      : bool   whether the redirection policy acts on it
    No osmnx / geopandas -- this is a plain JSON parse of already-built data.
    """
    path = Path(path) if path is not None else DEFAULT_CORRIDORS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        fc = json.load(fh)

    corridors = []
    for feat in fc.get("features", []):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        corridors.append({
            "name": props.get("name"),
            "coordinates": geom.get("coordinates"),
            "baseline_peak": props.get("baseline_peak"),
            "targeted": bool(props.get("targeted", False)),
        })
    return corridors


def simulate_redirection(registered_share, active_share, peak_shift_share,
                         corridor_share=scenario_engine.DEFAULT_CORRIDOR_SHARE,
                         path=None):
    """Spread the Section 6 network peak-shift across the map's corridors.

    The aggregate network_peak_reduction_pct and corridor_relief_pct come
    straight from scenario_engine.compute_scenario. The per-corridor reductions
    concentrate on the targeted arterials by the paper's 1/corridor_share factor,
    scaled so their energy-weighted aggregate reproduces network_peak_reduction_pct
    exactly. Peak demand is conserved (shift absorbed off-peak).

    Returns
    -------
    dict with:
        per_corridor : list of {name, coords, before_peak, after_peak,
                                relief_pct, targeted}
        network_peak_reduction_pct : aggregate (== compute_scenario), %
        corridor_relief_pct        : aggregate (== compute_scenario), %
        total_before, total_after  : aggregate PEAK load before/after the shift
        realized_network_reduction_pct : reduction implied by the totals (%),
                                         equals network_peak_reduction_pct
        note : conservation statement (off-peak absorbs the difference)
    """
    if corridor_share <= 0:
        raise ValueError("corridor_share must be positive.")

    # --- aggregate figures: taken verbatim from the scenario engine ----------
    agg = scenario_engine.compute_scenario(
        registered_share, active_share, peak_shift_share,
        corridor_share=corridor_share,
    )
    network_frac = agg["network_peak_reduction_pct"] / 100.0   # whole-network shift

    corridors = load_corridors(path)
    peaks = np.array([c["baseline_peak"] for c in corridors], dtype=float)
    targeted = np.array([c["targeted"] for c in corridors], dtype=bool)

    total_before = float(peaks.sum())

    # --- concentrate the shift on targeted corridors -------------------------
    # Targeted corridors relieve (1/corridor_share)x as much per peak trip as
    # non-targeted ones (the paper's concentration factor). Solve the common
    # base rate so the peak-weighted average reduction == network_frac exactly:
    #
    #   B_T * (base/corridor_share) + B_N * base = network_frac * total_before
    #
    # where B_T, B_N are the peak loads carried by targeted / non-targeted
    # corridors. Then weighted_drop / total_before == network_frac by construction.
    weight = np.where(targeted, 1.0 / corridor_share, 1.0)   # per-corridor concentration
    weighted_capacity = float((peaks * weight).sum())        # == B_T/cs + B_N

    base = 0.0 if weighted_capacity == 0 else network_frac * total_before / weighted_capacity
    relief_frac = base * weight                              # per-corridor reduction fraction

    after = peaks * (1.0 - relief_frac)
    after = np.maximum(after, 0.0)                           # never negative

    per_corridor = []
    for c, before_i, after_i, r_i, t_i in zip(
            corridors, peaks, after, relief_frac, targeted):
        per_corridor.append({
            "name": c["name"],
            "coords": c["coordinates"],
            "before_peak": float(before_i),
            "after_peak": float(after_i),
            "relief_pct": float(r_i * 100.0),
            "targeted": bool(t_i),
        })

    total_after = float(after.sum())
    realized = 0.0 if total_before == 0 else (total_before - total_after) / total_before * 100.0

    return {
        "per_corridor": per_corridor,
        "network_peak_reduction_pct": agg["network_peak_reduction_pct"],
        "corridor_relief_pct": agg["corridor_relief_pct"],
        "total_before": total_before,
        "total_after": total_after,
        "realized_network_reduction_pct": realized,
        "note": CONSERVATION_NOTE,
    }


if __name__ == "__main__":
    for name, preset in scenario_engine.PRESETS.items():
        res = simulate_redirection(
            preset["registered_share"], preset["active_share"],
            preset["peak_shift_share"],
        )
        tgt = [c for c in res["per_corridor"] if c["targeted"]]
        non = [c for c in res["per_corridor"] if not c["targeted"]]
        print(f"{name:7} network={res['network_peak_reduction_pct']:.4f}%  "
              f"corridor(agg)={res['corridor_relief_pct']:.4f}%  "
              f"targeted relief~{tgt[0]['relief_pct']:.3f}%  "
              f"non-targeted relief~{non[0]['relief_pct']:.3f}%  "
              f"realized={res['realized_network_reduction_pct']:.4f}%")
