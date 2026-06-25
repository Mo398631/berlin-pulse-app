"""Tab 8 cross-grid validation.

Checks, in order:
  ANCHOR   - Germany reproduces the validated baseline savings (hard gate).
  VALIDITY - the four ENTSO-E grids load clean (shape, columns, signs, totals).
  SPECTRUM - mean carbon intensity ranks PL > DE > ES > FR > NO2.
  BOUNDS   - every grid's savings sit in plausible ranges.

Prints a results table and a clear PASS / FAIL, exits non-zero on failure.
"""

import sys

import pandas as pd

import crossgrid_engine
from baseline_simulation import GEN_TYPES

ENTSOE_CODES = ["FR", "PL", "NO2", "ES"]

ANCHOR_CARBON = 2.3884
ANCHOR_COST = 4.8482
ANCHOR_TOL = 0.02


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


# ---- ANCHOR (hard) -----------------------------------------------------------

de = crossgrid_engine.run_grid("DE")
if (abs(de["carbon_saving_pct"] - ANCHOR_CARBON) > ANCHOR_TOL
        or abs(de["cost_saving_pct"] - ANCHOR_COST) > ANCHOR_TOL):
    print("ANCHOR FAIL")
    print(f"  carbon_saving_pct = {de['carbon_saving_pct']:.4f} "
          f"(expected {ANCHOR_CARBON} +/- {ANCHOR_TOL})")
    print(f"  cost_saving_pct   = {de['cost_saving_pct']:.4f} "
          f"(expected {ANCHOR_COST} +/- {ANCHOR_TOL})")
    sys.exit(1)
print(f"ANCHOR OK: DE carbon={de['carbon_saving_pct']:.4f}%  "
      f"cost={de['cost_saving_pct']:.4f}%")


# ---- VALIDITY (four ENTSO-E grids) -------------------------------------------

for code in ENTSOE_CODES:
    try:
        df = crossgrid_engine.load_grid(code)
    except Exception as exc:                       # noqa: BLE001
        fail(f"{code} failed to load: {exc!r}")

    n = len(df)
    if not (8758 <= n <= 8761):
        fail(f"{code} has {n} rows, expected 8758-8761")

    missing = [c for c in GEN_TYPES if c not in df.columns]
    if missing:
        fail(f"{code} missing generation columns: {missing}")

    neg = int((df[GEN_TYPES] < 0).to_numpy().sum())
    if neg:
        fail(f"{code} has {neg} negative generation values")

    if not (df["total_generation"] > 0).all():
        bad = int((df["total_generation"] <= 0).sum())
        fail(f"{code} has {bad} rows with total_generation <= 0")

    print(f"VALIDITY OK: {code} rows={n}, all 12 gen cols present, "
          f"no negatives, total_generation > 0 everywhere")


# ---- Run every grid ----------------------------------------------------------

results = crossgrid_engine.compare_grids()
by_code = {r["code"]: r for r in results}


# ---- SPECTRUM ----------------------------------------------------------------

mi = {c: by_code[c]["mean_intensity"] for c in by_code}
if not (mi["PL"] > mi["DE"] > mi["ES"] > mi["FR"] > mi["NO2"]):
    fail("mean_intensity spectrum violated; expected PL > DE > ES > FR > NO2, "
         f"got " + ", ".join(f"{c}={mi[c]:.1f}" for c in ["PL", "DE", "ES", "FR", "NO2"]))
print("SPECTRUM OK: PL > DE > ES > FR > NO2  ("
      + " > ".join(f"{c} {mi[c]:.1f}" for c in ["PL", "DE", "ES", "FR", "NO2"]) + ")")


# ---- SANITY BOUNDS -----------------------------------------------------------

for r in results:
    if not (0 <= r["carbon_saving_pct"] <= 15):
        fail(f"{r['code']} carbon_saving_pct={r['carbon_saving_pct']:.4f} out of [0, 15]")
    if not (0 <= r["cost_saving_pct"] <= 30):
        fail(f"{r['code']} cost_saving_pct={r['cost_saving_pct']:.4f} out of [0, 30]")
print("BOUNDS OK: carbon in [0,15] and cost in [0,30] for all five grids")


# ---- Results table -----------------------------------------------------------

print()
print("=" * 60)
print("CROSS-GRID RESULTS")
print("=" * 60)
print(f"{'code':<5}{'label':<9}{'carbon%':>10}{'cost%':>10}{'mean gCO2/kWh':>16}")
print("-" * 60)
for r in results:
    print(f"{r['code']:<5}{r['label']:<9}"
          f"{r['carbon_saving_pct']:>10.4f}{r['cost_saving_pct']:>10.4f}"
          f"{r['mean_intensity']:>16.1f}")
print("=" * 60)
print()
print("PASS")
