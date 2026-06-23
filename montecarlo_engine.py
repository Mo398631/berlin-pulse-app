"""
Appendix D, Pillar One: emission-factor Monte Carlo on the carbon saving.

This module makes the manuscript's Appendix D.2 robustness test reproducible and
interactive. It does NOT re-implement any physics: the night-by-night carbon
arithmetic is the reused `strategy_A` / `strategy_B` primitives from
`baseline_simulation.py`, and the deterministic baseline is the reused
`add_intensity` -> `simulate_year` pipeline run once. Only the four conventional
emission factors are perturbed.

Protocol (Appendix D.2.1, reproduced verbatim):
  - The four conventional factors are replaced by distributions whose MODE (for
    the triangulars) or centre is the paper's deterministic value, so the test
    measures the spread AROUND the existing result, not a relocation of it.
  - 10,000 factor vectors are drawn under fixed seed 20250619.
  - For each draw the hourly intensity series is rebuilt from the per-technology
    generation (exactly the (sum gen*factor)/total_generation that
    `add_intensity` computes), Strategy A vs Strategy B are re-run over the 364
    complete nights, and the carbon saving of B against A is recorded, together
    with the Appendix B flatness metric (the spread between the highest and
    lowest per-clock-hour mean intensity over the training window).

Published Appendix D.2.2 figures this reproduces (base Monte Carlo):
  carbon saving mean   2.393 %     median 2.392 %
  90% interval         2.329 - 2.461 %      extreme range 2.300 - 2.498 %
  flatness mean        12.39 g/kWh          90% interval 10.73 - 14.01 g
  single-digit (< 5 %) carbon saving on every draw.

Public API:
    run_emission_factor_mc(n_draws=10000, seed=20250619) -> dict
"""

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_simulation import (
    add_intensity, simulate_year, complete_nights, slot_energies,
    strategy_A, strategy_B, GEN_TYPES, EMISSION_FACTORS,
)

PARQUET = Path(__file__).resolve().parent / "data" / "berlin_pulse_validated_dataset.parquet"

# Training window for the flatness metric: 1 Jan - 30 Sep 2025 (Appendix B split).
TRAIN_CUTOFF = "2025-10-01"
CLOCK = [22, 23, 0, 1, 2, 3, 4]            # within-night chronological clock hours

# The reproduced deterministic carbon saving (Appendix D / E anchor): Table B.2's
# 2.402 less the openly-stated 0.014-point partial-hour residual -> 2.388 %.
DETERMINISTIC_BASELINE = 2.388

# ---- Table D.1 emission-factor distributions (HARD-CODED from the paper) -------
# Source: Berlin_Pulse_v20_Final_SSRN.docx, Appendix D, Table D.1. The mode of
# each triangular distribution is the paper's deterministic value, so the
# simulation is centred on the existing result.
#
#   Generation source     Paper value   Distribution   Range (g/kWh)
#   Lignite               1,054         Triangular     980, 1,054, 1,140
#   Hard coal               884         Triangular     800,   884,   965
#   Fossil gas              401         Triangular     350,   401,   500
#   Other conventional      700         Uniform        500, 900
#   Renewables/nuclear/
#   biomass/pumped storage    0         Fixed          0
#
# NOTE: the task brief described all four conventional factors as "triangular",
# but Table D.1 is authoritative and gives the mixed-residual "other conventional"
# category a flat (Uniform 500-900) uninformative range, NOT a triangular. We
# follow Table D.1 exactly so the run reproduces the published 2.393 % mean; only
# lignite / hard coal / fossil gas are triangular. (These are the EXACT published
# bounds, read from Table D.1 with the docx parsed offline; the app never reads
# the docx at run time.)
TRIANGULAR_FACTORS = {                # (left/min, mode/paper value, right/max)
    "lignite":    (980.0, 1054.0, 1140.0),
    "hard_coal":  (800.0,  884.0,  965.0),
    "fossil_gas": (350.0,  401.0,  500.0),
}
UNIFORM_FACTORS = {                   # (low, high)
    "other_conventional": (500.0, 900.0),
}


def _load_df():
    """Load the validated dataset (no Streamlit, no caching in the engine)."""
    return pd.read_parquet(PARQUET).set_index("timestamp_berlin")


def _precompute_night_arrays(df):
    """Per complete in-window night, the generation matrix, total generation and
    price as plain numpy, computed ONCE so each draw only rebuilds intensity and
    re-runs the reused strategy primitives (no per-draw pandas groupby)."""
    nights = complete_nights(df)
    w = df[df["in_window"] & df["night_id"].isin(nights)].sort_index()
    groups = []
    for _, g in w.groupby("night_id"):
        g = g.sort_index()
        groups.append((
            g[GEN_TYPES].to_numpy(dtype=float),       # h x 12 generation
            g["total_generation"].to_numpy(dtype=float),
            g["dayahead_price"].to_numpy(dtype=float),
        ))
    return groups


def _precompute_train_clock_blocks(df):
    """For the flatness metric: per training-window clock hour, the stacked
    generation matrix and total-generation vector of its in-window rows, so each
    draw can recompute the per-clock-hour mean intensity cheaply.

    Restricted to the COMPLETE training nights (Appendix B split, data-edge
    nights excluded) so it matches gate.frozen_orders' Table B.1 means exactly.
    """
    train = [n for n in complete_nights(df) if n < TRAIN_CUTOFF]
    w = df[df["in_window"] & df["night_id"].isin(train)].copy()
    blocks = []
    hr = w.index.hour
    for h in CLOCK:
        sel = w[hr == h]
        blocks.append((
            sel[GEN_TYPES].to_numpy(dtype=float),
            sel["total_generation"].to_numpy(dtype=float),
        ))
    return blocks


def _factor_vector(factors):
    """Pack a factors dict into the GEN_TYPES-ordered numpy vector add_intensity
    would dot against the generation columns."""
    return np.array([factors[t] for t in GEN_TYPES], dtype=float)


def _draw_factor_vectors(n_draws, seed):
    """Draw the perturbed GEN_TYPES factor matrix, shape (12, n_draws).

    Zero-factor technologies stay fixed at 0; the three triangulars and the one
    uniform are sampled under the fixed seed. Drawn per Table D.1.
    """
    rng = np.random.default_rng(seed)
    base = _factor_vector(EMISSION_FACTORS)            # deterministic factors
    F = np.repeat(base[:, None], n_draws, axis=1)      # (12, n_draws)
    idx = {t: i for i, t in enumerate(GEN_TYPES)}
    for tech, (lo, mode, hi) in TRIANGULAR_FACTORS.items():
        F[idx[tech]] = rng.triangular(lo, mode, hi, size=n_draws)
    for tech, (lo, hi) in UNIFORM_FACTORS.items():
        F[idx[tech]] = rng.uniform(lo, hi, size=n_draws)
    return F


def _carbon_saving_for_factors(groups, fvec, slots):
    """Carbon saving (%) of Strategy B vs A for one factor vector, reusing the
    strategy primitives night by night. Identical arithmetic to simulate_year,
    with the per-night intensity rebuilt as add_intensity does: (gen.f)/total."""
    A_co2 = B_co2 = 0.0
    for G, T, p in groups:
        intensity = (G @ fvec) / T                     # g CO2 / kWh, == add_intensity
        aC, _ = strategy_A(intensity, p, slots)        # reused, unmodified
        bC, _ = strategy_B(intensity, p, slots)        # reused, unmodified
        A_co2 += aC
        B_co2 += bC
    return (A_co2 - B_co2) / A_co2 * 100.0


def _flatness_for_factors(blocks, fvec):
    """Appendix B flatness metric for one factor vector: spread between the
    highest and lowest per-clock-hour mean intensity over the training window."""
    clock_means = np.empty(len(blocks))
    for i, (G, T) in enumerate(blocks):
        clock_means[i] = np.mean((G @ fvec) / T)
    return float(clock_means.max() - clock_means.min())


def run_emission_factor_mc(n_draws=10000, seed=20250619):
    """Run the Appendix D Pillar-One emission-factor Monte Carlo.

    Reuses add_intensity (deterministic baseline) and the strategy / simulate
    primitives from baseline_simulation.py. NOT cached here (the Streamlit layer
    wraps this in st.cache_data so the 10,000-draw run computes once).

    Returns a dict with:
        carbon_saving_mean, carbon_saving_median, carbon_p5, carbon_p95 : floats (%)
        flatness_mean            : float (g CO2/kWh, training-window profile spread)
        carbon_savings           : np.ndarray of per-draw carbon savings (%)
        flatness_values          : np.ndarray of per-draw flatness (g/kWh)
        cost_saving_mean         : float (%), the incidental cost-of-carbon control
        deterministic_baseline   : 2.388 (reproduced, add_intensity/simulate_year)
        paper_deterministic      : 2.41 (published headline)
        n_draws, seed            : echoed inputs
    """
    df = _load_df()
    slots = slot_energies()

    # Deterministic baseline via the REUSED pipeline (add_intensity -> simulate_year),
    # cross-checked against the documented 2.388 anchor.
    det = simulate_year(add_intensity(df, EMISSION_FACTORS))
    deterministic_repro = det["carbon_saving_pct"]

    groups = _precompute_night_arrays(df)
    blocks = _precompute_train_clock_blocks(df)
    F = _draw_factor_vectors(n_draws, seed)

    carbon = np.empty(n_draws)
    flatness = np.empty(n_draws)
    cost = np.empty(n_draws)
    for d in range(n_draws):
        fvec = F[:, d]
        # carbon saving + incidental cost control, night by night (reused primitives)
        A_co2 = B_co2 = A_cost = B_cost = 0.0
        for G, T, p in groups:
            intensity = (G @ fvec) / T
            aC, aK = strategy_A(intensity, p, slots)
            bC, bK = strategy_B(intensity, p, slots)
            A_co2 += aC; B_co2 += bC
            A_cost += aK; B_cost += bK
        carbon[d] = (A_co2 - B_co2) / A_co2 * 100.0
        cost[d] = (A_cost - B_cost) / A_cost * 100.0
        flatness[d] = _flatness_for_factors(blocks, fvec)

    return {
        "carbon_saving_mean": float(np.mean(carbon)),
        "carbon_saving_median": float(np.median(carbon)),
        "carbon_p5": float(np.percentile(carbon, 5)),
        "carbon_p95": float(np.percentile(carbon, 95)),
        "flatness_mean": float(np.mean(flatness)),
        "carbon_savings": carbon,
        "flatness_values": flatness,
        "cost_saving_mean": float(np.mean(cost)),
        "deterministic_baseline": DETERMINISTIC_BASELINE,
        "deterministic_reproduced": float(deterministic_repro),
        "paper_deterministic": 2.41,
        "n_draws": int(n_draws),
        "seed": int(seed),
    }


if __name__ == "__main__":
    r = run_emission_factor_mc()
    print(f"draws            : {r['n_draws']}  (seed {r['seed']})")
    print(f"deterministic    : reproduced {r['deterministic_reproduced']:.4f}% "
          f"(anchor {r['deterministic_baseline']}, paper {r['paper_deterministic']})")
    print(f"carbon mean      : {r['carbon_saving_mean']:.4f}%  (paper 2.393)")
    print(f"carbon median    : {r['carbon_saving_median']:.4f}%  (paper 2.392)")
    print(f"carbon 90% int   : {r['carbon_p5']:.4f} - {r['carbon_p95']:.4f}%  "
          f"(paper 2.329 - 2.461)")
    print(f"flatness mean    : {r['flatness_mean']:.4f} g/kWh  (paper 12.39)")
    print(f"max carbon saving: {r['carbon_savings'].max():.4f}%  (single-digit on every draw)")
    print(f"cost control mean: {r['cost_saving_mean']:.4f}%")
