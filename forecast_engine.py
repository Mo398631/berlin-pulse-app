"""
Berlin Pulse, Appendix E: Forecast Recovery engine (Session 02).

Reproduce-before-perturb. This module FIRST reproduces the five Appendix B
bookends through the reused Appendix D / gate primitives and asserts them, and
ONLY THEN builds and scores the forecast-driven charging schedules of Appendix E.

No physics is rewritten. Every charging evaluation reuses the imported
primitives verbatim (strategy_A, _evaluate, slot_energies) and every night-set /
frozen-order / ranking convention is the one gate.py already uses. The only new
logic here is (a) the Appendix E.3 forecast -> intensity mappings, (b) the
Appendix E.2 naive forecasts, and (c) the recovery-fraction scoring. None of it
touches realized data when building a forecast; realized intensity and price are
used only to *score* a schedule, never to *choose* it.

The Oct-26 midnight forecast gap (Session 01 filled forecast_wind_onshore with
0.0 at 2025-10-26 00:00 +02:00) is repaired here per Appendix E by linear
time-interpolation of the three day-ahead component columns at that one cell,
using day-ahead values only. No realized data is used for the fill.

Public API:
    compute_forecast_recovery() -> dict          # everything the tab UI needs

Does not modify any existing file or any parquet on disk.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Reused primitives -- imported, never rewritten.
from baseline_simulation import (
    strategy_A, _evaluate, slot_energies, complete_nights, DEFAULTS,
)
from gate import (
    night_sets, frozen_orders, order_from_clock_priority, run, CLOCK,
)

ROOT = Path(__file__).resolve().parent
VALIDATED_PARQUET = ROOT / "data" / "berlin_pulse_validated_dataset.parquet"
FORECAST_PARQUET = ROOT / "data" / "berlin_pulse_forecast_dataset.parquet"

# The single Session-01 gap cell, repaired by day-ahead time-interpolation.
GAP_CELL_TS = pd.Timestamp("2025-10-26 00:00:00", tz="Europe/Berlin")
FORECAST_COMPONENTS = ["forecast_wind_onshore", "forecast_wind_offshore", "forecast_solar"]

# Predictor design, pre-designated (Appendix E.3) -- NOT chosen after seeing test data.
OPERATIONAL_PREDICTORS = {
    "share": "forecast_renewable_share",          # E.3 primary
    "direct": "forecast_renewable_mwh",           # alt: renewable MWh, no share
    "merit_order": "forecast_fossil_residual",    # alt: total - renewable (fossil residual)
}

TRAIN_CUTOFF = "2025-10-01"      # nights < cutoff are training (Q1-Q3); >= are test (Q4)
RANDOM_NULL_TRIALS = 4000
RANDOM_NULL_SEED = 20251026

# Bookend acceptance (Appendix B / golden numbers), reproduce-before-perturb.
_BOOKEND_TOL = {
    "complete_nights": 364,
    "carbon_oracle_q4": (3.756, 0.014),
    "deployable_carbon_q4": (0.000, 1e-6),
    "deployable_cost_fy": (8.756, 0.01),
    "cost_oracle_fy": (9.144, 0.02),
}


# ---- Loading -----------------------------------------------------------------

def load_validated():
    """Validated parquet indexed by timestamp_berlin (the index authority)."""
    return pd.read_parquet(VALIDATED_PARQUET).set_index("timestamp_berlin")


def load_forecast():
    """Load the forecast parquet and repair the Oct-26 midnight gap cell.

    Appendix E: the Session-01 gap (forecast_wind_onshore filled with 0.0 at the
    2025-10-26 00:00 +02:00 cell) must instead be filled by linear time-
    interpolation of the three day-ahead component columns at that one cell.
    forecast_renewable_mwh and forecast_renewable_share are recomputed at that
    row. No realized data is used.

    Returns
    -------
    (fc, gap_flag)
        fc : DataFrame indexed by timestamp_berlin with the component columns,
             forecast_total_generation_mwh, forecast_renewable_mwh,
             forecast_renewable_share and forecast_fossil_residual.
        gap_flag : dict describing the repaired cell (for the gap report).
    """
    fc = pd.read_parquet(FORECAST_PARQUET).set_index("timestamp_berlin")

    before = {c: float(fc.at[GAP_CELL_TS, c]) for c in FORECAST_COMPONENTS}

    # Linear time-interpolation of the three day-ahead component columns at the
    # one gap cell. NaN only that cell, then interpolate on the time index so the
    # fill is the day-ahead neighbours' linear midpoint -- no realized data.
    for c in FORECAST_COMPONENTS:
        s = fc[c].copy()
        s.loc[GAP_CELL_TS] = np.nan
        fc[c] = s.interpolate(method="time")

    after = {c: float(fc.at[GAP_CELL_TS, c]) for c in FORECAST_COMPONENTS}

    # Recompute the renewable sum (and share / fossil residual) everywhere; the
    # gap row now reflects the interpolated components rather than the 0.0 fill.
    fc["forecast_renewable_mwh"] = fc[FORECAST_COMPONENTS].sum(axis=1)
    fc["forecast_renewable_share"] = (
        fc["forecast_renewable_mwh"] / fc["forecast_total_generation_mwh"]
    )
    fc["forecast_fossil_residual"] = (
        fc["forecast_total_generation_mwh"] - fc["forecast_renewable_mwh"]
    )

    gap_flag = {
        "timestamp": str(GAP_CELL_TS),
        "method": "linear time-interpolation of day-ahead components (no realized data)",
        "before": before,                                   # 0.0-filled values
        "after": after,                                     # interpolated values
        "forecast_renewable_mwh": float(fc.at[GAP_CELL_TS, "forecast_renewable_mwh"]),
        "forecast_renewable_share": float(fc.at[GAP_CELL_TS, "forecast_renewable_share"]),
        "interpolated": True,
    }
    return fc, gap_flag


# ---- STEP 1: reproduce the Appendix B bookends (assert before any forecast) ---

def _reproduce_bookends(df):
    """Reproduce-before-perturb: rebuild and assert the five Appendix B bookends
    through the reused gate primitives, returning the reproduced values."""
    full, train, q4 = night_sets(df)
    slots = slot_energies()
    carbon_clocks, price_clocks, _, _ = frozen_orders(df, train)

    # Same rankers gate.main() builds, on the reused argsort / order convention.
    rank_carbon_oracle = lambda e, p, h: list(np.argsort(e, kind="stable"))
    rank_price_oracle = lambda e, p, h: list(np.argsort(p, kind="stable"))
    rank_frozen_carbon = lambda e, p, h: order_from_clock_priority(h, carbon_clocks)
    rank_frozen_price = lambda e, p, h: order_from_clock_priority(h, price_clocks)

    carbon_oracle_q4, _ = run(df, q4, rank_carbon_oracle, slots)
    carbon_oracle_fy, _ = run(df, full, rank_carbon_oracle, slots)
    deployable_carbon_q4, _ = run(df, q4, rank_frozen_carbon, slots)
    _, deployable_cost_fy = run(df, full, rank_frozen_price, slots)
    _, cost_oracle_fy = run(df, full, rank_price_oracle, slots)

    n_complete = len(complete_nights(df))
    assert n_complete == _BOOKEND_TOL["complete_nights"], n_complete
    assert len(train) == 273 and len(q4) == 91 and len(full) == 364, (
        len(train), len(q4), len(full))

    def _assert(name, value):
        target, tol = _BOOKEND_TOL[name]
        assert abs(value - target) <= tol + 1e-12, (name, value, target, tol)

    _assert("carbon_oracle_q4", carbon_oracle_q4)
    _assert("deployable_carbon_q4", deployable_carbon_q4)
    _assert("deployable_cost_fy", deployable_cost_fy)
    _assert("cost_oracle_fy", cost_oracle_fy)

    return {
        "carbon_oracle_q4_pct": carbon_oracle_q4,
        "carbon_oracle_fy_pct": carbon_oracle_fy,
        "deployable_carbon_q4_pct": deployable_carbon_q4,
        "deployable_cost_fy_pct": deployable_cost_fy,
        "cost_oracle_fy_pct": cost_oracle_fy,
    }


# ---- per-night realized arrays (built once) ----------------------------------

def _night_arrays(df, nights):
    """Map night_id -> realized arrays for the in-window hours of that night.

    e = realized production_intensity, p = realized day-ahead price, hours = the
    within-night clock hours, idx = the row index (for forecast lookups), and
    weekday = the night's civil weekday (for climatology).
    """
    w = df[df["in_window"] & df["night_id"].isin(nights)]
    out = {}
    for nid, g in w.groupby("night_id"):
        g = g.sort_index()
        out[nid] = {
            "e": g["production_intensity"].to_numpy(),
            "p": g["dayahead_price"].to_numpy(),
            "hours": list(g.index.hour),
            "idx": g.index,
            "weekday": pd.Timestamp(nid).weekday(),
        }
    return out


# ---- STEP 2: operational forecast intensity (Appendix E.3) -------------------

def _fit_linear(df, feature, train):
    """Freeze a 1-D linear mapping realized intensity ~ feature on TRAINING,
    in-window hours only. Returns (slope, intercept). Never refit on test."""
    m = df["in_window"] & df["night_id"].isin(train)
    x = df.loc[m, feature].to_numpy()
    y = df.loc[m, "production_intensity"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _operational_predictions(df, train):
    """Build forecast_intensity for share / direct / merit_order over ALL nights
    from frozen training coefficients. Returns dict name -> Series on df.index."""
    preds = {}
    coeffs = {}
    for name, feature in OPERATIONAL_PREDICTORS.items():
        slope, intercept = _fit_linear(df, feature, train)
        coeffs[name] = (slope, intercept)
        preds[name] = slope * df[feature] + intercept     # frozen mapping, all nights
    return preds, coeffs


# ---- STEP 3: naive forecasts (Appendix E.2) ----------------------------------

def _clock_intensity(night):
    """Mean realized intensity per clock-hour for one night (handles the DST
    fall-back duplicate 02:00 by averaging)."""
    m = {}
    for h, ev in zip(night["hours"], night["e"]):
        m.setdefault(h, []).append(ev)
    return {h: float(np.mean(v)) for h, v in m.items()}


def _persistence_predictions(nd, full):
    """Persistence (E.2): each clock-hour's predicted intensity = that clock-
    hour's realized intensity on the PREVIOUS available night. The first test
    night is thereby seeded from the last training night (they are consecutive in
    the full ordering); the very first night of the year, having no predecessor,
    is seeded from itself."""
    clock = {n: _clock_intensity(nd[n]) for n in full}
    preds = {}
    for i, nid in enumerate(full):
        prev = full[i - 1] if i > 0 else nid
        prev_clock = clock[prev]
        self_clock = clock[nid]
        d = nd[nid]
        # Fall back to this night's own clock value only if a clock-hour is
        # absent from the predecessor (DST length change); never realized of a
        # future night.
        preds[nid] = np.array(
            [prev_clock.get(h, self_clock[h]) for h in d["hours"]], dtype=float
        )
    return preds


def _climatology_predictions(nd, train, full):
    """Climatology (E.2): training-window mean intensity per (weekday, clock-
    hour), frozen, then read back for every night. Fitted on training only."""
    rows = []
    for n in train:
        d = nd[n]
        for h, ev in zip(d["hours"], d["e"]):
            rows.append((d["weekday"], h, ev))
    table = pd.DataFrame(rows, columns=["weekday", "clock", "intensity"])
    means = table.groupby(["weekday", "clock"])["intensity"].mean()
    global_mean = float(table["intensity"].mean())
    preds = {}
    for n in full:
        d = nd[n]
        preds[n] = np.array(
            [float(means.get((d["weekday"], h), global_mean)) for h in d["hours"]],
            dtype=float,
        )
    return preds


# ---- STEP 4: scoring (recovery fraction) -------------------------------------

def _forecast_order(values):
    """Within-night schedule: rank hours by forecast intensity ascending
    (cleanest forecast first). Reused argsort convention."""
    return list(np.argsort(np.asarray(values), kind="stable"))


def _score(pred_by_night, nd, nights):
    """Score one forecast over a night set.

    For each night, place the 5 slots into the hours ranked by forecast
    intensity, then evaluate with REALIZED intensity and price via the reused
    _evaluate. Returns (saving_pct, recovery_fraction).

    recovery_fraction = realized saving under the forecast schedule
                        / realized saving under the perfect-foresight oracle,
    on the same night set; saving_pct is vs naive Strategy A.
    """
    slots = slot_energies()
    A_co2 = S_co2 = O_co2 = 0.0
    for nid in nights:
        d = nd[nid]
        e, p = d["e"], d["p"]
        a_c, _ = strategy_A(e, p, slots)                              # naive baseline
        s_c, _ = _evaluate(e, p, slots, _forecast_order(pred_by_night[nid]))
        o_c, _ = _evaluate(e, p, slots, _forecast_order(e))           # carbon oracle
        A_co2 += a_c
        S_co2 += s_c
        O_co2 += o_c
    saving = A_co2 - S_co2
    oracle_saving = A_co2 - O_co2
    saving_pct = saving / A_co2 * 100.0
    recovery = saving / oracle_saving if oracle_saving != 0 else float("nan")
    return saving_pct, recovery


def _spearman(a, b):
    """Spearman rank correlation (no scipy): Pearson on ranks."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_rank_corr(pred_by_night, nd, nights):
    """Within-night Spearman (forecast vs realized intensity), averaged."""
    cs = [_spearman(pred_by_night[n], nd[n]["e"]) for n in nights]
    return float(np.nanmean(cs))


def _metrics(pred_by_night, nd, q4, full):
    return {
        "q4_saving_pct": _score(pred_by_night, nd, q4)[0],
        "fy_saving_pct": _score(pred_by_night, nd, full)[0],
        "q4_recovery": _score(pred_by_night, nd, q4)[1],
        "fy_recovery": _score(pred_by_night, nd, full)[1],
        "rank_corr": _avg_rank_corr(pred_by_night, nd, q4),
        "rank_corr_fy": _avg_rank_corr(pred_by_night, nd, full),
    }


def _random_null_q4_recovery(nd, q4):
    """Mean Q4 recovery fraction over RANDOM_NULL_TRIALS random within-night
    shuffles (seeded for reproducibility). The null floor for the metric."""
    slots = slot_energies()
    rng = np.random.default_rng(RANDOM_NULL_SEED)
    A_co2 = O_co2 = 0.0
    arrays = []
    for nid in q4:
        d = nd[nid]
        e, p = d["e"], d["p"]
        A_co2 += strategy_A(e, p, slots)[0]
        O_co2 += _evaluate(e, p, slots, _forecast_order(e))[0]
        arrays.append((e, p))
    oracle_saving = A_co2 - O_co2
    recs = np.empty(RANDOM_NULL_TRIALS)
    for t in range(RANDOM_NULL_TRIALS):
        s_co2 = 0.0
        for e, p in arrays:
            s_co2 += _evaluate(e, p, slots, list(rng.permutation(len(e))))[0]
        recs[t] = (A_co2 - s_co2) / oracle_saving
    return float(recs.mean())


# ---- STEP 5: public API ------------------------------------------------------

def compute_forecast_recovery():
    """Compute every figure the Forecast Recovery tab needs.

    Reproduce-before-perturb: the five Appendix B bookends are reproduced and
    asserted first; only then are the Appendix E forecasts built and scored.
    """
    df = load_validated()

    # STEP 1 -- reproduce and assert the bookends before any forecast logic.
    bookends = _reproduce_bookends(df)

    # Repair the Oct-26 gap and attach forecast features to the validated index.
    fc, gap_flag = load_forecast()
    feature_cols = ["forecast_renewable_share", "forecast_renewable_mwh",
                    "forecast_fossil_residual"]
    df = df.join(fc[feature_cols])

    full, train, q4 = night_sets(df)
    nd = _night_arrays(df, full)

    # STEP 2 -- operational forecast intensities (share / direct / merit_order).
    op_preds, op_coeffs = _operational_predictions(df, train)
    op_by_night = {
        name: {nid: op_preds[name].loc[nd[nid]["idx"]].to_numpy() for nid in full}
        for name in OPERATIONAL_PREDICTORS
    }

    # STEP 3 -- naive forecasts (persistence, climatology).
    naive_by_night = {
        "persistence": _persistence_predictions(nd, full),
        "climatology": _climatology_predictions(nd, train, full),
    }

    # STEP 4 -- score everything.
    operational = {name: _metrics(op_by_night[name], nd, q4, full)
                   for name in OPERATIONAL_PREDICTORS}
    for name in operational:
        operational[name]["coefficients"] = {
            "slope": op_coeffs[name][0], "intercept": op_coeffs[name][1]}

    naive = {name: _metrics(naive_by_night[name], nd, q4, full)
             for name in naive_by_night}

    random_null_q4 = _random_null_q4_recovery(nd, q4)

    # Recovery bookends, by construction: blind == 0%, oracle == 100%.
    blind_by_night = {nid: nd[nid]["e"] * 0.0 + 1.0 for nid in full}  # constant -> clock order
    blind_q4_recovery = _score(blind_by_night, nd, q4)[1]
    oracle_by_night = {nid: nd[nid]["e"] for nid in full}
    oracle_q4_recovery = _score(oracle_by_night, nd, q4)[1]
    assert abs(blind_q4_recovery - 0.0) < 1e-9, blind_q4_recovery
    assert abs(oracle_q4_recovery - 1.0) < 1e-4, oracle_q4_recovery

    return {
        "bookends": {
            "carbon_oracle_q4_pct": bookends["carbon_oracle_q4_pct"],
            "carbon_oracle_fy_pct": bookends["carbon_oracle_fy_pct"],
            "deployable_carbon_q4_pct": bookends["deployable_carbon_q4_pct"],
            "deployable_cost_fy_pct": bookends["deployable_cost_fy_pct"],
            "cost_oracle_fy_pct": bookends["cost_oracle_fy_pct"],
            "blind_recovery": blind_q4_recovery,     # 0% bookend (by construction)
            "oracle_recovery": oracle_q4_recovery,   # 100% bookend (by construction)
        },
        "naive": naive,
        "operational": operational,
        "random_null": {"q4_recovery": random_null_q4},
        "gap_cell": gap_flag,
        "training_nights": len(train),
        "test_nights": len(q4),
        "full_nights": len(full),
    }


if __name__ == "__main__":
    import json

    r = compute_forecast_recovery()

    def f(x):
        return f"{x:+.4f}" if isinstance(x, float) else x

    print("=" * 78)
    print("APPENDIX E - FORECAST RECOVERY  (reproduce-before-perturb)")
    print("=" * 78)
    b = r["bookends"]
    print(f"Nights: full {r['full_nights']}  train {r['training_nights']}  "
          f"Q4 {r['test_nights']}")
    print(f"Bookends  Q4 carbon oracle {b['carbon_oracle_q4_pct']:.4f}  "
          f"FY carbon oracle {b['carbon_oracle_fy_pct']:.4f}  "
          f"deployable carbon Q4 {b['deployable_carbon_q4_pct']:.4f}")
    print(f"          blind recovery {b['blind_recovery']:.4f}  "
          f"oracle recovery {b['oracle_recovery']:.4f}")
    print("-" * 78)
    print(f"{'forecast':14}{'Q4 save%':>10}{'Q4 recov':>10}"
          f"{'FY save%':>10}{'FY recov':>10}{'rankcorr':>10}")
    print("-" * 78)
    for pillar in ("operational", "naive"):
        for name, m in r[pillar].items():
            print(f"{name:14}{m['q4_saving_pct']:10.4f}{m['q4_recovery']:10.4f}"
                  f"{m['fy_saving_pct']:10.4f}{m['fy_recovery']:10.4f}"
                  f"{m['rank_corr']:10.4f}")
    print("-" * 78)
    print(f"random null Q4 recovery: {r['random_null']['q4_recovery']:.4f}")
    print(f"gap cell {r['gap_cell']['timestamp']}: "
          f"renewable_mwh {r['gap_cell']['forecast_renewable_mwh']:.2f}  "
          f"share {r['gap_cell']['forecast_renewable_share']:.4f}  "
          f"(interpolated={r['gap_cell']['interpolated']})")
    print("=" * 78)
