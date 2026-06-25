"""
Pure-logic engine for the Berlin Pulse Rider demo (the seventh Streamlit tab).

This module makes the manuscript's Section 5.3 *five-step incentive logic*
interactive from a single rider's point of view: plan a morning trip, detect
whether it lands on a TARGETED corridor during the peak ("red zone"), offer the
three behavioural alternatives the paper proposes (retime / reroute / green
hour), reward a chosen shift with the paper's four instruments, and finally roll
many riders up into the SAME network/corridor figures the Scenario Explorer
reports.

Like engine.py, scenario_engine.py and prototype_engine.py this is a *pure*
module: no Streamlit, no global state. The only I/O is reading the small,
already-processed prototype_data/corridors.geojson with the standard-library
`json` reader (NO osmnx / geopandas at runtime; those are dev-only, see
build_prototype_data.py). Everything else is light arithmetic, so test_rider.py
can lock the numbers the way test_engine.py locks the optimizer.

EVERYTHING HERE IS SYNTHETIC / ILLUSTRATIVE. The landmark demand, crowding and
carbon reductions, and point awards are stylised figures derived from the
paper's equations and a corridor's targeted status -- they are NOT measured
ridership, emissions, or a forecast.

RECONCILIATION with scenario_engine (the binding contract)
----------------------------------------------------------
aggregate_effect reports its network and corridor figures *verbatim* from
scenario_engine.compute_scenario, exactly as prototype_engine (Tab 6) does, so
the rider rollup can never drift away from the headline Scenario Explorer tab.
The rider dial `compliance_rate` maps to the scenario's `active_share` (both are
"the fraction of registered riders who actively respond"); `registered_share`
and `peak_shift_share` are the operating point, defaulting to the MEDIUM preset.

Public API:

    LANDMARKS                                   -> list[dict]
    plan_trip(origin, dest, depart_minute, flexible=True) -> dict
    generate_alternatives(trip)                 -> list[dict]
    compute_reward(alternative)                 -> dict
    aggregate_effect(compliance_rate, corridor_share=0.25, ...) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import scenario_engine

# Default location of the processed, committed corridor geometry (reused from
# the map prototype -- same file Tab 6 reads).
HERE = Path(__file__).resolve().parent
DEFAULT_CORRIDORS_PATH = HERE / "prototype_data" / "corridors.geojson"

# --- the simulated morning peak and the low-carbon "green hour" ---------------
# Minutes are minute-of-day (0..1439). Peak = 07:30-09:00 inclusive of start,
# exclusive of end, per the demo brief.
PEAK_START_MIN = 7 * 60 + 30          # 07:30 = 450
PEAK_END_MIN = 9 * 60                  # 09:00 = 540

# A RETIME nudge leaves before the peak builds: ~25 min ahead of peak start.
RETIME_EARLIER_MIN = 25
RETIME_DEPART_MIN = PEAK_START_MIN - RETIME_EARLIER_MIN   # 07:05 = 425

# The "green hour": a midday, low grid-carbon window (more renewables on the
# Berlin grid). We pick a representative depart minute inside it.
GREEN_WINDOW = (11 * 60, 15 * 60)      # 11:00-15:00
GREEN_HOUR_MIN = 13 * 60               # 13:00 = 780


# --- curated Berlin landmarks, each pinned to a corridor in corridors.geojson -
# Coordinates are real (lon, lat); the `corridor` is the arterial each spot sits
# on / feeds, and MUST match a corridor name in prototype_data/corridors.geojson
# so targeted status and synthetic peak load can be looked up consistently.
LANDMARKS = [
    {"name": "Alexanderplatz",      "lon": 13.4132, "lat": 52.5219,
     "corridor": "Karl-Marx-Allee"},
    {"name": "Ostkreuz",            "lon": 13.4694, "lat": 52.5030,
     "corridor": "Frankfurter Allee"},
    {"name": "Warschauer Strasse",  "lon": 13.4490, "lat": 52.5055,
     "corridor": "Frankfurter Allee"},
    {"name": "Kurfuerstendamm/Zoo", "lon": 13.3330, "lat": 52.5073,
     "corridor": "Kurfuerstendamm"},
    {"name": "Hermannplatz",        "lon": 13.4244, "lat": 52.4869,
     "corridor": "Sonnenallee"},
    {"name": "Potsdamer Platz",     "lon": 13.3759, "lat": 52.5096,
     "corridor": "Potsdamer Strasse"},
    {"name": "Hauptbahnhof",        "lon": 13.3694, "lat": 52.5251,
     "corridor": "Leipziger Strasse"},
    {"name": "Tempelhof",           "lon": 13.3858, "lat": 52.4821,
     "corridor": "Tempelhofer Damm"},
]


# --- reward instrument constants (paper Section 5.3), all illustrative --------
PULSE_POINTS_SCALE = 1000.0   # pulse_points = round(scale * crowding_reduction)
GREEN_BONUS_SCALE = 500.0     # green_bonus  = round(scale * carbon_reduction)
POINTS_PER_EURO = 100.0       # 100 Pulse Points ~ EUR 1.00 partner perk

# Per-alternative-type yield factors (fraction of peak load relieved / carbon
# saved per unit of targeted intensity). RETIME removes the rider from the peak
# entirely (largest crowding relief); REROUTE keeps them in the peak but off the
# red corridor (smaller); GREEN shifts to the low-carbon hour (largest carbon).
CROWD_FACTOR = {"RETIME": 0.30, "REROUTE": 0.22, "GREEN": 0.25}
CARBON_FACTOR = {"RETIME": 0.08, "REROUTE": 0.05, "GREEN": 0.30}


# --- corridor lookup (targeted status + synthetic peak load) ------------------

def _load_corridor_index(path=None):
    """Return {corridor_name: {"targeted": bool, "baseline_peak": float}}.

    A plain stdlib-json parse of the same already-built corridors.geojson the
    map prototype uses -- NO osmnx / geopandas at runtime.
    """
    path = Path(path) if path is not None else DEFAULT_CORRIDORS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        fc = json.load(fh)
    index = {}
    for feat in fc.get("features", []):
        props = feat.get("properties", {}) or {}
        name = props.get("name")
        index[name] = {
            "targeted": bool(props.get("targeted", False)),
            "baseline_peak": float(props.get("baseline_peak", 0.0)),
        }
    return index


def _landmark(name):
    for lm in LANDMARKS:
        if lm["name"] == name:
            return lm
    raise KeyError(f"unknown landmark {name!r}; choose from "
                   f"{[lm['name'] for lm in LANDMARKS]}")


def is_peak(depart_minute):
    """True if depart_minute falls in the simulated morning peak (07:30-09:00)."""
    return PEAK_START_MIN <= depart_minute < PEAK_END_MIN


def _minute_to_hhmm(m):
    m = int(m) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


# --- step 1-2: plan the baseline trip and detect the red zone -----------------

def plan_trip(origin, dest, depart_minute, flexible=True, path=None):
    """Plan a rider's baseline trip and flag whether it is a peak "red zone".

    Parameters
    ----------
    origin, dest : str        Landmark names (see LANDMARKS).
    depart_minute : int       Departure as minute-of-day (0..1439).
    flexible : bool           Whether the rider can shift (gate for alternatives).
    path : str or None        Optional override of the corridors.geojson path.

    Returns
    -------
    dict with:
        origin, dest, depart_minute, depart_hhmm, flexible
        route_corridors      : list[str]  corridors the trip traverses
        targeted_corridors   : list[str]  the subset that the policy targets
        uses_targeted        : bool
        in_peak              : bool        depart_minute is in 07:30-09:00
        red_zone             : bool        in_peak AND uses_targeted
        targeted_intensity   : float       share of route PEAK load on targeted
                                            corridors (0..1); drives later relief
    """
    o, d = _landmark(origin), _landmark(dest)
    index = _load_corridor_index(path)

    # the route traverses the corridor each endpoint sits on (union, dedup,
    # order-stable: origin first)
    route_corridors = [o["corridor"]]
    if d["corridor"] != o["corridor"]:
        route_corridors.append(d["corridor"])

    targeted_corridors = [c for c in route_corridors
                          if index.get(c, {}).get("targeted", False)]
    uses_targeted = len(targeted_corridors) > 0

    total_peak = sum(index.get(c, {}).get("baseline_peak", 0.0)
                     for c in route_corridors)
    targeted_peak = sum(index.get(c, {}).get("baseline_peak", 0.0)
                        for c in targeted_corridors)
    targeted_intensity = 0.0 if total_peak == 0 else targeted_peak / total_peak

    in_peak = is_peak(depart_minute)
    red_zone = in_peak and uses_targeted

    return {
        "origin": origin,
        "dest": dest,
        "depart_minute": int(depart_minute),
        "depart_hhmm": _minute_to_hhmm(depart_minute),
        "flexible": bool(flexible),
        "route_corridors": route_corridors,
        "targeted_corridors": targeted_corridors,
        "uses_targeted": uses_targeted,
        "in_peak": in_peak,
        "red_zone": red_zone,
        "targeted_intensity": targeted_intensity,
    }


# --- step 3: generate the three behavioural alternatives ----------------------

def generate_alternatives(trip, path=None):
    """Offer the paper's three alternatives for a flexible, red-zone trip.

    Returns [] unless the trip is BOTH flexible AND in the red zone (only then
    is there a peak conflict worth nudging). Otherwise returns a list of three
    alternatives -- one RETIME, one REROUTE, one GREEN -- each carrying an
    illustrative `crowding_reduction` and `carbon_reduction` derived from the
    trip's targeted intensity and the per-type yield factors.

    Each alternative is a dict:
        type               : "RETIME" | "REROUTE" | "GREEN"
        description        : str
        new_depart_minute  : int   (RETIME / GREEN move the time; REROUTE keeps it)
        new_depart_hhmm    : str
        route_corridors    : list[str]  (REROUTE swaps to non-targeted parallels)
        crowding_reduction : float  fraction of the rider's peak crowding relieved
        carbon_reduction   : float  fraction of the trip's carbon relieved
    """
    if not (trip.get("flexible") and trip.get("red_zone")):
        return []

    intensity = trip["targeted_intensity"]
    index = _load_corridor_index(path)
    non_targeted = [name for name, meta in index.items()
                    if not meta["targeted"]]

    def _reductions(kind):
        return (CROWD_FACTOR[kind] * intensity, CARBON_FACTOR[kind] * intensity)

    alternatives = []

    # RETIME: leave ~25 min ahead of the peak, off-peak, same route.
    crowd, carbon = _reductions("RETIME")
    shift = trip["depart_minute"] - RETIME_DEPART_MIN
    alternatives.append({
        "type": "RETIME",
        "description": (f"Leave at {_minute_to_hhmm(RETIME_DEPART_MIN)} "
                        f"({shift} min earlier) to beat the peak."),
        "new_depart_minute": RETIME_DEPART_MIN,
        "new_depart_hhmm": _minute_to_hhmm(RETIME_DEPART_MIN),
        "route_corridors": list(trip["route_corridors"]),
        "crowding_reduction": crowd,
        "carbon_reduction": carbon,
    })

    # REROUTE: same time, parallel NON-targeted corridor(s).
    crowd, carbon = _reductions("REROUTE")
    alternatives.append({
        "type": "REROUTE",
        "description": ("Take a parallel non-targeted corridor "
                        "at the same time (similar travel time)."),
        "new_depart_minute": trip["depart_minute"],
        "new_depart_hhmm": trip["depart_hhmm"],
        "route_corridors": non_targeted,
        "crowding_reduction": crowd,
        "carbon_reduction": carbon,
    })

    # GREEN: depart in the low-carbon green hour.
    crowd, carbon = _reductions("GREEN")
    alternatives.append({
        "type": "GREEN",
        "description": (f"Travel in the green hour around "
                        f"{_minute_to_hhmm(GREEN_HOUR_MIN)} (low grid carbon)."),
        "new_depart_minute": GREEN_HOUR_MIN,
        "new_depart_hhmm": _minute_to_hhmm(GREEN_HOUR_MIN),
        "route_corridors": list(trip["route_corridors"]),
        "crowding_reduction": crowd,
        "carbon_reduction": carbon,
    })

    return alternatives


# --- step 4: reward a chosen shift with the four instruments -------------------

def compute_reward(alternative):
    """Reward a chosen alternative with the paper's four incentive instruments.

    Point formula (all illustrative):
        pulse_points = round(PULSE_POINTS_SCALE * crowding_reduction)
            -> the core relief reward; MORE points for bigger peak relief, so a
               RETIME (factor 0.30) beats a REROUTE (factor 0.22) at equal
               targeted intensity.
        green_bonus  = round(GREEN_BONUS_SCALE * carbon_reduction)  if GREEN
                     = 0                                            otherwise
            -> an extra carbon reward reserved for the low-carbon green hour.
        total_points = pulse_points + green_bonus
        lottery_entry = True for ANY recommended shift (every alternative earns
                        one prize-draw entry).
        partner_value = an illustrative points-to-perk note
                        (POINTS_PER_EURO = 100 Pulse Points ~ EUR 1.00 credit).

    Returns a dict with: type, crowding_reduction, carbon_reduction,
    pulse_points, green_bonus, total_points, lottery_entry, partner_value, formula.
    """
    kind = alternative["type"]
    crowd = float(alternative["crowding_reduction"])
    carbon = float(alternative["carbon_reduction"])

    pulse_points = int(round(PULSE_POINTS_SCALE * crowd))
    green_bonus = int(round(GREEN_BONUS_SCALE * carbon)) if kind == "GREEN" else 0
    total_points = pulse_points + green_bonus
    lottery_entry = True   # any recommended shift earns a prize-draw entry

    euro = total_points / POINTS_PER_EURO
    partner_value = (f"{total_points} Pulse Points ~ EUR {euro:.2f} partner perk "
                     f"credit (illustrative; {POINTS_PER_EURO:.0f} pts = EUR 1).")

    return {
        "type": kind,
        "crowding_reduction": crowd,
        "carbon_reduction": carbon,
        "pulse_points": pulse_points,
        "green_bonus": green_bonus,
        "total_points": total_points,
        "lottery_entry": lottery_entry,
        "partner_value": partner_value,
        "formula": ("pulse_points=round(1000*crowding_reduction); "
                    "green_bonus=round(500*carbon_reduction) if GREEN else 0; "
                    "total=pulse_points+green_bonus"),
    }


# --- step 5: roll many riders up into the Scenario Explorer figures -----------

def aggregate_effect(compliance_rate,
                     corridor_share=scenario_engine.DEFAULT_CORRIDOR_SHARE,
                     registered_share=None, peak_shift_share=None):
    """Roll rider compliance up into the network/corridor figures, RECONCILED.

    The rider dial `compliance_rate` is the share of registered riders who
    actually accept a recommended shift -- exactly scenario_engine's
    `active_share`. We pass it straight into compute_scenario together with the
    operating point (`registered_share`, `peak_shift_share`, defaulting to the
    MEDIUM preset), so the returned figures are taken VERBATIM from the Scenario
    Explorer's engine and can never drift from it.

    Returns
    -------
    dict with:
        compliance_rate, registered_share, peak_shift_share, corridor_share
        network_peak_reduction_pct : == compute_scenario(...) (%)
        corridor_relief_pct        : == compute_scenario(...) (%), uncapped
        corridor_relief_display_pct: corridor relief clamped to the paper's band
    """
    if registered_share is None:
        registered_share = scenario_engine.MEDIUM["registered_share"]
    if peak_shift_share is None:
        peak_shift_share = scenario_engine.MEDIUM["peak_shift_share"]

    agg = scenario_engine.compute_scenario(
        registered_share, compliance_rate, peak_shift_share,
        corridor_share=corridor_share,
    )

    return {
        "compliance_rate": compliance_rate,
        "registered_share": registered_share,
        "peak_shift_share": peak_shift_share,
        "corridor_share": corridor_share,
        "network_peak_reduction_pct": agg["network_peak_reduction_pct"],
        "corridor_relief_pct": agg["corridor_relief_pct"],
        "corridor_relief_display_pct":
            scenario_engine.display_corridor_relief_pct(agg["corridor_relief_pct"]),
    }


if __name__ == "__main__":
    # A worked example: Alexanderplatz -> Kurfuerstendamm/Zoo at 08:15 (peak,
    # both endpoints on targeted corridors -> red zone).
    trip = plan_trip("Alexanderplatz", "Kurfuerstendamm/Zoo", 8 * 60 + 15)
    print(f"Trip {trip['origin']} -> {trip['dest']} at {trip['depart_hhmm']}  "
          f"route={trip['route_corridors']}")
    print(f"  in_peak={trip['in_peak']}  uses_targeted={trip['uses_targeted']}  "
          f"red_zone={trip['red_zone']}  "
          f"targeted_intensity={trip['targeted_intensity']:.3f}")

    print("\nAlternatives + rewards:")
    for alt in generate_alternatives(trip):
        rw = compute_reward(alt)
        print(f"  {alt['type']:7} crowd={alt['crowding_reduction']:.4f} "
              f"carbon={alt['carbon_reduction']:.4f}  "
              f"pulse={rw['pulse_points']:>4} green_bonus={rw['green_bonus']:>3} "
              f"total={rw['total_points']:>4} lottery={rw['lottery_entry']}")

    print("\nAggregate effect (reconciles with Scenario Explorer):")
    for name, preset in scenario_engine.PRESETS.items():
        res = aggregate_effect(
            preset["active_share"],
            registered_share=preset["registered_share"],
            peak_shift_share=preset["peak_shift_share"])
        print(f"  {name:7} compliance={preset['active_share']:.2f}  "
              f"network={res['network_peak_reduction_pct']:.4f}%  "
              f"corridor={res['corridor_relief_pct']:.4f}% "
              f"(display {res['corridor_relief_display_pct']:.4f}%)")
