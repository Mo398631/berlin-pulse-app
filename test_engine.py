"""
Smoke test for engine.run_simulation on the default case.

Default: 277 buses, 240 kWh/bus, 50 kW charger, window 22:00->05:00.
Expected (gate.py bookend, gate.py:123):
    carbon saving = 2.39 +/- 0.014   (Table B.2 2.402 less the 0.014 partial-hour
                                       residual -> realised 2.388; the 2.41 headline
                                       is the theoretical, not the reproduced, figure)
    cost saving   ~ 4.85 %           (App D 4.877; realised 4.848)

The carbon band matches gate.py's own documented +/-0.014 tolerance. If the value
falls outside this the refactor altered the reused logic; the test fails loudly
rather than papering over it.
"""

from engine import run_simulation

CARBON_TARGET, CARBON_TOL = 2.39, 0.014           # gate.py:123 documented residual
CARBON_LO, CARBON_HI = CARBON_TARGET - CARBON_TOL, CARBON_TARGET + CARBON_TOL
COST_LO, COST_HI = 4.80, 4.90


def test_default_case():
    res = run_simulation(n_buses=277, kwh_per_bus=240.0,
                         charger_kw=50.0, window_hours=(22, 5))

    carbon = res["carbon_saving_pct"]
    cost = res["cost_saving_pct"]

    print(f"Nights simulated : {res['n_nights']}")
    print(f"Slots (kWh)      : {res['slots_kwh']}")
    print(f"Carbon saving    : {carbon:.4f}%   (expect 2.39 +/- 0.014)")
    print(f"Cost saving      : {cost:.4f}%   (expect ~4.85)")
    print(f"Fleet CO2 saved  : {res['fleet_co2_saved_tonnes']:.1f} t/yr")
    print(f"Fleet cost saved : {res['fleet_cost_saved_eur']:,.0f} EUR/yr")
    print(f"Per-bus CO2 saved: {res['per_bus']['co2_saved_kg']:.2f} kg/yr")
    print(f"Per-night rows   : {len(res['per_night'])}")

    assert CARBON_LO <= carbon <= CARBON_HI, (
        f"carbon saving {carbon:.4f}% outside {CARBON_TARGET} +/- {CARBON_TOL} "
        f"([{CARBON_LO:.3f}, {CARBON_HI:.3f}]) -- "
        "refactor likely altered the logic; STOP and investigate."
    )
    assert COST_LO <= cost <= COST_HI, (
        f"cost saving {cost:.4f}% outside [{COST_LO}, {COST_HI}] -- "
        "refactor likely altered the logic; STOP and investigate."
    )
    assert res["n_nights"] == 364, f"expected 364 complete nights, got {res['n_nights']}"
    assert len(res["per_night"]) == res["n_nights"]

    print("\nPASS: default case reproduces the carbon and cost bookends.")


if __name__ == "__main__":
    test_default_case()
