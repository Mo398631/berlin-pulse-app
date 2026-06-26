"""
Golden-numbers safety guard (Appendix E, Session 00).

A standalone tripwire that re-asserts the Appendix B / Appendix D bookends BEFORE
any forecasting work begins. It rewrites no physics: it runs gate.py's main()
verbatim and re-reads the deployability table through gate.deployability_results,
both of which reuse baseline_simulation unchanged.

Run from the repo root:

    python scripts_check/golden_numbers.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable when launched as scripts_check/golden_numbers.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gate                                          # noqa: E402
from gate import PARQUET, main, deployability_results  # noqa: E402
from baseline_simulation import complete_nights        # noqa: E402


def approx(value, target, tol):
    return abs(value - target) <= tol + 1e-12


def check(label, value, target, tol):
    ok = approx(value, target, tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:36} "
          f"{value:11.6f}  (target {target:.3f} +/- {tol})")
    assert ok, f"{label}: {value} not within {tol} of {target}"


def run():
    print("=" * 78)
    print("GOLDEN NUMBERS SAFETY GUARD  (Appendix E, Session 00)")
    print("=" * 78)

    # 1. The full Appendix E gate must pass exactly as shipped.
    print("\n[1] Running gate.main() ...")
    print("-" * 78)
    gate_ok = main()
    print("-" * 78)
    # main() returns a numpy bool_; assert by truth value, not identity.
    assert bool(gate_ok) is True, "gate.main() did not return True"
    print("  [PASS] gate.main() returned True")

    # Re-read the validated parquet the same way gate.main() does, then pull the
    # deployability table through the reused (unmodified) gate harness.
    df = pd.read_parquet(PARQUET).set_index("timestamp_berlin")
    nights = complete_nights(df)
    results = deployability_results(df)

    print("\n[2..7] Re-asserting golden numbers ...")
    print("-" * 78)

    # 2. Night count.
    n = len(nights)
    print(f"  [{'PASS' if n == 364 else 'FAIL'}] complete_nights count"
          f"{'':17}{n:11d}  (target 364)")
    assert n == 364, f"complete_nights count {n} != 364"

    full = results["full"]
    q4 = results["test"]   # Q4 == the 'test' window in deployability_results

    # 3. Carbon oracle, full year.
    check("carbon oracle (full year)", full["carbon"]["oracle"], 2.39, 0.014)

    # 4. Carbon oracle, Q4.
    check("carbon oracle (Q4)", q4["carbon"]["oracle"], 3.76, 0.014)

    # 5. Deployable carbon, Q4 (frozen order == clock order -> zero saving).
    check("deployable carbon (Q4)", q4["carbon"]["deployable"], 0.000, 1e-6)

    # 6. Deployable cost (frozen price rule), full year.
    check("deployable cost (full year)", full["cost"]["deployable"], 8.756, 0.01)

    # 7. Cost oracle (price oracle), full year.
    check("cost oracle (full year)", full["cost"]["oracle"], 9.144, 0.02)

    print("-" * 78)
    print("\nALL GOLDEN NUMBERS PASS")
    print("=" * 78)


if __name__ == "__main__":
    run()
