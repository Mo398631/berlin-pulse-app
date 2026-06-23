"""
Smoke test for gate.deployability_results — the Appendix B Table B.2 figures.

Locks the deployability-gap numbers the way test_engine.py locks the optimizer:
the deployable (blind / frozen training-window) rule vs the perfect-foresight
oracle, for the CARBON and COST channels, over train (Q1-Q3), test (Q4) and
the full year.

Published Table B.2 acceptance (tolerance 0.05 pp unless noted):
    Carbon deployable (blind): 0.000 on train, test, full   (tol 0.001)
        -> the frozen carbon order reduces to clock order, i.e. naive charging,
           because the overnight carbon profile is nearly flat.
    Carbon oracle:  full ~2.39 (2.37-2.41);  Q4 ~3.75 (3.70-3.80)
    Cost deployable (frozen price): Q4 ~8.77 (8.70-8.85); full ~8.80 (8.70-8.85)
    Cost oracle:  full ~9.14 (9.10-9.20)

If any value falls outside its band the reused physics has drifted; the test
fails loudly rather than papering over it.
"""

from pathlib import Path

import pandas as pd

from gate import deployability_results

PARQUET = Path(__file__).resolve().parent / "data" / "berlin_pulse_validated_dataset.parquet"

CARBON_BLIND_TOL = 0.001   # near-exact zero deployable carbon co-benefit


def test_table_b2():
    df = pd.read_parquet(PARQUET).set_index("timestamp_berlin")
    res = deployability_results(df)

    train, test, full = res["train"], res["test"], res["full"]

    print(f"{'Window':9}{'nights':>8}"
          f"{'carbon dep':>12}{'carbon orc':>12}"
          f"{'cost dep':>11}{'cost orc':>11}")
    print("-" * 63)
    for name in ("train", "test", "full"):
        r = res[name]
        print(f"{name:9}{r['n_nights']:>8}"
              f"{r['carbon']['deployable']:12.4f}{r['carbon']['oracle']:12.4f}"
              f"{r['cost']['deployable']:11.4f}{r['cost']['oracle']:11.4f}")
    print("-" * 63)

    checks = [
        # label, value, lo, hi
        ("Carbon deployable (blind) train", train["carbon"]["deployable"],
         -CARBON_BLIND_TOL, CARBON_BLIND_TOL),
        ("Carbon deployable (blind) test ", test["carbon"]["deployable"],
         -CARBON_BLIND_TOL, CARBON_BLIND_TOL),
        ("Carbon deployable (blind) full ", full["carbon"]["deployable"],
         -CARBON_BLIND_TOL, CARBON_BLIND_TOL),
        ("Carbon oracle full             ", full["carbon"]["oracle"], 2.37, 2.41),
        ("Carbon oracle Q4 (test)        ", test["carbon"]["oracle"], 3.70, 3.80),
        ("Cost deployable Q4 (test)      ", test["cost"]["deployable"], 8.70, 8.85),
        ("Cost deployable full           ", full["cost"]["deployable"], 8.70, 8.85),
        ("Cost oracle full               ", full["cost"]["oracle"], 9.10, 9.20),
    ]

    print(f"{'Check':33}{'value':>10}{'band':>16}  result")
    print("-" * 70)
    all_pass = True
    for label, val, lo, hi in checks:
        ok = lo <= val <= hi
        all_pass &= ok
        print(f"{label:33}{val:10.4f}   [{lo:.3f}, {hi:.3f}]  "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 70)

    # Hard assertions (mirror the printed table) so the test fails loudly.
    assert train["n_nights"] == 273, train["n_nights"]
    assert test["n_nights"] == 91, test["n_nights"]
    assert full["n_nights"] == 364, full["n_nights"]
    for label, val, lo, hi in checks:
        assert lo <= val <= hi, f"{label}: {val:.4f} outside [{lo}, {hi}]"

    assert all_pass
    print("\nPASS: deployability_results reproduces the Table B.2 figures.")


if __name__ == "__main__":
    test_table_b2()
