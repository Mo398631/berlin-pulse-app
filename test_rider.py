"""
Smoke test for rider_engine -- the Section 5.3 five-step incentive logic.

Locks the rider demo the way test_prototype.py locks the map prototype. The
binding contract is the SAME reconciliation: for every Section 6 preset,
aggregate_effect must equal scenario_engine.compute_scenario(...) to within
1e-6, so the rider rollup can never drift from the Scenario Explorer tab. The
rest asserts the per-rider logic (red-zone detection, alternatives, rewards).

Run:  python test_rider.py
"""

import rider_engine
from rider_engine import (
    LANDMARKS, plan_trip, generate_alternatives, compute_reward, aggregate_effect,
)
import scenario_engine

RECON_TOL = 1e-6      # aggregates must match compute_scenario to here
MIN_LANDMARKS = 6


def _valid_coord(lon, lat):
    return (isinstance(lon, (int, float)) and isinstance(lat, (int, float))
            and -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0)


def test_rider_engine():
    # --- LANDMARKS: >= 6 entries, each with valid coordinates ----------------
    print(f"Landmarks loaded : {len(LANDMARKS)}")
    coords_ok = True
    for lm in LANDMARKS:
        ok = _valid_coord(lm["lon"], lm["lat"])
        coords_ok = coords_ok and ok
        print(f"   - {lm['name']:<22} ({lm['lon']:.4f}, {lm['lat']:.4f})  "
              f"on {lm['corridor']:<18} {'ok' if ok else 'BAD'}")
    assert len(LANDMARKS) >= MIN_LANDMARKS, (
        f"need >= {MIN_LANDMARKS} landmarks, got {len(LANDMARKS)}")
    assert coords_ok, "a landmark has invalid coordinates"

    # --- red-zone detection: peak+targeted True; off-peak False --------------
    # Alexanderplatz (Karl-Marx-Allee) -> Kurfuerstendamm/Zoo (Kurfuerstendamm):
    # both endpoints sit on TARGETED corridors.
    peak_trip = plan_trip("Alexanderplatz", "Kurfuerstendamm/Zoo", 8 * 60 + 15)
    offpeak_trip = plan_trip("Alexanderplatz", "Kurfuerstendamm/Zoo", 10 * 60)
    print(f"\nPeak trip    {peak_trip['depart_hhmm']}  route={peak_trip['route_corridors']}  "
          f"in_peak={peak_trip['in_peak']}  uses_targeted={peak_trip['uses_targeted']}  "
          f"red_zone={peak_trip['red_zone']}")
    print(f"Off-peak trip {offpeak_trip['depart_hhmm']}  in_peak={offpeak_trip['in_peak']}  "
          f"red_zone={offpeak_trip['red_zone']}")
    assert peak_trip["uses_targeted"], "test trip should traverse a targeted corridor"
    assert peak_trip["red_zone"] is True, "peak trip over targeted corridor must be red_zone"
    assert offpeak_trip["red_zone"] is False, "off-peak trip must NOT be red_zone"

    # --- alternatives: >= 1 option with positive crowding_reduction ----------
    alts = generate_alternatives(peak_trip)
    print(f"\nAlternatives for the red-zone trip ({len(alts)}):")
    for a in alts:
        print(f"   {a['type']:7} depart={a['new_depart_hhmm']}  "
              f"crowding_reduction={a['crowding_reduction']:.4f}  "
              f"carbon_reduction={a['carbon_reduction']:.4f}")
    assert len(alts) >= 1, "a flexible red-zone trip must yield >= 1 alternative"
    assert any(a["crowding_reduction"] > 0 for a in alts), (
        "at least one alternative must have positive crowding_reduction")
    # a non-flexible (or off-peak) trip yields none
    assert generate_alternatives(offpeak_trip) == [], (
        "off-peak trip should yield no alternatives")
    stiff = plan_trip("Alexanderplatz", "Kurfuerstendamm/Zoo", 8 * 60 + 15,
                      flexible=False)
    assert generate_alternatives(stiff) == [], (
        "non-flexible trip should yield no alternatives")

    # --- rewards: bigger relief -> more points; lottery entry; GREEN bonus ---
    by_type = {a["type"]: a for a in alts}
    retime_rw = compute_reward(by_type["RETIME"])   # biggest crowd factor (0.30)
    reroute_rw = compute_reward(by_type["REROUTE"])  # smallest crowd factor (0.22)
    green_rw = compute_reward(by_type["GREEN"])
    print("\nRewards:")
    for rw in (retime_rw, reroute_rw, green_rw):
        print(f"   {rw['type']:7} pulse={rw['pulse_points']:>4}  "
              f"green_bonus={rw['green_bonus']:>3}  total={rw['total_points']:>4}  "
              f"lottery={rw['lottery_entry']}")
    print(f"   partner_value e.g.: {green_rw['partner_value']}")

    # MORE points for the bigger-relief shift
    assert retime_rw["pulse_points"] > reroute_rw["pulse_points"], (
        f"bigger-relief RETIME ({retime_rw['pulse_points']}) must out-point "
        f"smaller-relief REROUTE ({reroute_rw['pulse_points']})")
    # every recommended shift earns a lottery entry
    assert all(rw["lottery_entry"] is True
               for rw in (retime_rw, reroute_rw, green_rw)), (
        "every recommended shift must award a lottery_entry")
    # GREEN carries a green_bonus; non-green do not
    assert green_rw["green_bonus"] > 0, "GREEN option must carry a green_bonus"
    assert retime_rw["green_bonus"] == 0 and reroute_rw["green_bonus"] == 0, (
        "non-GREEN options must not carry a green_bonus")

    # --- reconciliation: aggregate_effect == compute_scenario for L/M/H ------
    print()
    header = (f"{'Preset':7}{'compliance':>12}{'network %':>12}"
              f"{'corridor %':>13}{'scen net %':>13}{'scen cor %':>13}")
    print(header)
    print("-" * len(header))
    for name, preset in scenario_engine.PRESETS.items():
        ref = scenario_engine.compute_scenario(
            preset["registered_share"], preset["active_share"],
            preset["peak_shift_share"])
        # compliance_rate maps to active_share; operating point = the preset
        res = aggregate_effect(
            preset["active_share"],
            registered_share=preset["registered_share"],
            peak_shift_share=preset["peak_shift_share"])
        print(f"{name:7}{preset['active_share']:12.4f}"
              f"{res['network_peak_reduction_pct']:12.4f}"
              f"{res['corridor_relief_pct']:13.4f}"
              f"{ref['network_peak_reduction_pct']:13.4f}"
              f"{ref['corridor_relief_pct']:13.4f}")

        d_net = abs(res["network_peak_reduction_pct"]
                    - ref["network_peak_reduction_pct"])
        d_cor = abs(res["corridor_relief_pct"] - ref["corridor_relief_pct"])
        assert d_net <= RECON_TOL, (
            f"{name}: network aggregate off by {d_net:.2e} -- reconciliation broken.")
        assert d_cor <= RECON_TOL, (
            f"{name}: corridor aggregate off by {d_cor:.2e} -- reconciliation broken.")
    print("-" * len(header))

    print("\nPASS: landmarks valid; red-zone detection correct; alternatives and\n"
          "      rewards behave; aggregate_effect reconciles with compute_scenario\n"
          "      on all presets to 1e-6.")


if __name__ == "__main__":
    test_rider_engine()
