"""
Smoke test for montecarlo_engine.run_emission_factor_mc — the Appendix D Pillar
One emission-factor Monte Carlo. Locks the published Appendix D.2.2 figures the
way test_engine.py locks the optimizer and test_deployability.py locks Table B.2.

Published Appendix D.2.2 acceptance (10,000 draws, seed 20250619):
    carbon saving mean   2.393 %     -> assert in [2.36, 2.42]
    90% interval         2.329 - 2.461 %
                         -> assert carbon_p5 >= 2.30 and carbon_p95 <= 2.50
    flatness mean        12.39 g/kWh -> assert in [11.5, 13.5]
    single-digit saving on every draw  -> assert every per-draw saving < 5

If any value falls outside its band the reused physics or the Table D.1 bounds
have drifted; the test fails loudly rather than papering over it.
"""

from montecarlo_engine import run_emission_factor_mc


def test_emission_factor_mc():
    r = run_emission_factor_mc(n_draws=10000, seed=20250619)

    mean = r["carbon_saving_mean"]
    median = r["carbon_saving_median"]
    p5 = r["carbon_p5"]
    p95 = r["carbon_p95"]
    flat = r["flatness_mean"]
    max_saving = float(r["carbon_savings"].max())

    print(f"draws                : {r['n_draws']}  (seed {r['seed']})")
    print(f"deterministic baseline: {r['deterministic_baseline']} "
          f"(reproduced {r['deterministic_reproduced']:.4f}%, paper {r['paper_deterministic']})")
    print(f"carbon saving mean   : {mean:.4f} %   (paper 2.393, band [2.36, 2.42])")
    print(f"carbon saving median : {median:.4f} %   (paper 2.392)")
    print(f"carbon 90% interval  : {p5:.4f} - {p95:.4f} %   (paper 2.329 - 2.461)")
    print(f"flatness mean        : {flat:.4f} g/kWh   (paper 12.39, band [11.5, 13.5])")
    print(f"max carbon saving    : {max_saving:.4f} %   (must be < 5 on every draw)")
    print(f"cost control mean    : {r['cost_saving_mean']:.4f} %")
    print("-" * 60)

    checks = [
        ("carbon mean in [2.36, 2.42]", 2.36 <= mean <= 2.42),
        ("carbon_p5 >= 2.30", p5 >= 2.30),
        ("carbon_p95 <= 2.50", p95 <= 2.50),
        ("flatness mean in [11.5, 13.5]", 11.5 <= flat <= 13.5),
        ("every per-draw saving < 5 (single-digit)", max_saving < 5.0),
    ]
    all_pass = True
    for label, ok in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("-" * 60)

    # Hard assertions (mirror the printed checks) so the test fails loudly.
    assert 2.36 <= mean <= 2.42, f"carbon mean {mean:.4f} outside [2.36, 2.42]"
    assert p5 >= 2.30, f"carbon_p5 {p5:.4f} below 2.30"
    assert p95 <= 2.50, f"carbon_p95 {p95:.4f} above 2.50"
    assert 11.5 <= flat <= 13.5, f"flatness mean {flat:.4f} outside [11.5, 13.5]"
    assert max_saving < 5.0, f"a draw produced saving {max_saving:.4f} >= 5 (not single-digit)"

    assert all_pass
    print("\nPASS: emission-factor Monte Carlo reproduces the Appendix D.2.2 figures.")


if __name__ == "__main__":
    test_emission_factor_mc()
