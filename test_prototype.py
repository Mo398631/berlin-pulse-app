"""
Smoke test for prototype_engine -- the map prototype's redirection logic.

Locks the spatial engine the way test_scenario.py locks the scenario arithmetic.
The binding contract is RECONCILIATION: for every Section 6 preset the map's
aggregate network_peak_reduction_pct and corridor_relief_pct must equal
scenario_engine.compute_scenario(...) to within 1e-6, and the per-corridor
effects must be spatially sane (targeted corridors relieved more than the rest,
no corridor driven negative).

Run:  python test_prototype.py
"""

from prototype_engine import load_corridors, simulate_redirection
import scenario_engine

RECON_TOL = 1e-6      # aggregates must match compute_scenario to here
MIN_CORRIDORS = 5     # build_prototype_data guarantees at least this many


def _valid_coords(coords):
    """True if `coords` is a non-empty nest of [lon, lat] numeric pairs."""
    def walk(node):
        # a coordinate pair: [number, number]
        if (isinstance(node, (list, tuple)) and len(node) == 2
                and all(isinstance(v, (int, float)) for v in node)):
            lon, lat = node
            return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
        if isinstance(node, (list, tuple)) and node:
            return all(walk(child) for child in node)
        return False
    return isinstance(coords, (list, tuple)) and len(coords) > 0 and walk(coords)


def test_prototype_engine():
    # --- load_corridors: >= 5 corridors, each with valid coordinates ---------
    corridors = load_corridors()
    n = len(corridors)
    n_targeted = sum(c["targeted"] for c in corridors)
    coords_ok = all(_valid_coords(c["coordinates"]) for c in corridors)
    peaks_ok = all(isinstance(c["baseline_peak"], (int, float))
                   and c["baseline_peak"] > 0 for c in corridors)

    print(f"Corridors loaded : {n}  (targeted={n_targeted}, "
          f"non-targeted={n - n_targeted})")
    for c in corridors:
        flag = "TARGETED" if c["targeted"] else "        "
        npts = sum(1 for _ in _flatten_pairs(c["coordinates"]))
        print(f"   - {c['name']:<20} {flag}  peak={c['baseline_peak']:>5}  "
              f"pts={npts:>4}")
    print(f"Coords valid     : {coords_ok}")

    assert n >= MIN_CORRIDORS, f"need >= {MIN_CORRIDORS} corridors, got {n}"
    assert coords_ok, "a corridor has invalid/empty coordinates"
    assert peaks_ok, "a corridor has a non-positive/invalid baseline_peak"
    assert n_targeted >= 1 and n_targeted < n, (
        "need both targeted and non-targeted corridors for the relief contrast")

    # --- reconciliation + spatial sanity for Low / Medium / High -------------
    print()
    header = (f"{'Preset':7}{'network %':>12}{'corridor %':>13}"
              f"{'tgt relief':>12}{'non relief':>12}{'realized %':>12}")
    print(header)
    print("-" * len(header))

    for name, preset in scenario_engine.PRESETS.items():
        ref = scenario_engine.compute_scenario(
            preset["registered_share"], preset["active_share"],
            preset["peak_shift_share"])
        res = simulate_redirection(
            preset["registered_share"], preset["active_share"],
            preset["peak_shift_share"])

        tgt = [c["relief_pct"] for c in res["per_corridor"] if c["targeted"]]
        non = [c["relief_pct"] for c in res["per_corridor"] if not c["targeted"]]

        print(f"{name:7}{res['network_peak_reduction_pct']:12.4f}"
              f"{res['corridor_relief_pct']:13.4f}"
              f"{min(tgt):12.4f}{max(non):12.4f}"
              f"{res['realized_network_reduction_pct']:12.4f}")

        # aggregates match compute_scenario to 1e-6 (the reconciliation contract)
        d_net = abs(res["network_peak_reduction_pct"]
                    - ref["network_peak_reduction_pct"])
        d_cor = abs(res["corridor_relief_pct"] - ref["corridor_relief_pct"])
        assert d_net <= RECON_TOL, (
            f"{name}: network aggregate off by {d_net:.2e} "
            f"(map {res['network_peak_reduction_pct']} vs scenario "
            f"{ref['network_peak_reduction_pct']}) -- reconciliation broken.")
        assert d_cor <= RECON_TOL, (
            f"{name}: corridor aggregate off by {d_cor:.2e} "
            f"(map {res['corridor_relief_pct']} vs scenario "
            f"{ref['corridor_relief_pct']}) -- reconciliation broken.")

        # the realized (peak-weighted) reduction must also equal the aggregate
        assert abs(res["realized_network_reduction_pct"]
                   - res["network_peak_reduction_pct"]) <= RECON_TOL, (
            f"{name}: realized reduction {res['realized_network_reduction_pct']} "
            f"!= reported {res['network_peak_reduction_pct']} -- not conservative.")

        # every targeted corridor relieved MORE than every non-targeted one
        assert min(tgt) > max(non), (
            f"{name}: targeted relief {min(tgt):.4f}% not > non-targeted "
            f"{max(non):.4f}% -- concentration lost.")

        # no corridor's after_peak is negative; demand conserved downward
        assert all(c["after_peak"] >= 0.0 for c in res["per_corridor"]), (
            f"{name}: a corridor has negative after_peak.")
        assert res["total_after"] <= res["total_before"] + RECON_TOL, (
            f"{name}: total_after exceeds total_before -- trips created.")

    print("-" * len(header))
    print("\nPASS: map prototype reconciles with compute_scenario on all "
          "presets;\n      targeted corridors relieved more, demand conserved, "
          "no negatives.")


def _flatten_pairs(node):
    """Yield each [lon, lat] pair in an arbitrarily nested coordinate list."""
    if (isinstance(node, (list, tuple)) and len(node) == 2
            and all(isinstance(v, (int, float)) for v in node)):
        yield node
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _flatten_pairs(child)


if __name__ == "__main__":
    test_prototype_engine()
