"""
Pure-logic engine for the Berlin Pulse Scenario Explorer (paper Section 6).

This module makes the manuscript's congestion / peak-shift scenarios
interactive. Like engine.py it is a *pure* module: no Streamlit, no file I/O,
no global state — just arithmetic over the share inputs, so its numbers can be
locked by test_scenario.py exactly the way test_engine.py locks the optimizer.

The model (paper Section 6) is a simple chained product of adoption shares:

    network_peak_reduction = registered_share * active_share * peak_shift_share
    corridor_relief        = network_peak_reduction / corridor_share

`corridor_share` defaults to 0.25 — the targeted corridors carry ~25% of peak
trips per the paper, so the same network-wide shift concentrates ~4x on them.

All `*_share` arguments are fractions in [0, 1]; every returned figure is a
percentage (share * 100). These are illustrative what-if figures from the
paper's equations, NOT forecasts.

Public API:

    compute_scenario(registered_share, active_share, peak_shift_share,
                     corridor_share=0.25, energy_shift_share=None) -> dict
"""

# Default corridor concentration: targeted corridors carry ~25% of peak trips.
DEFAULT_CORRIDOR_SHARE = 0.25

# The paper reports corridor relief within an 8-12% band; the raw arithmetic at
# the HIGH preset (14.4%) exceeds that, so the UI caps the *displayed* figure to
# this ceiling. compute_scenario itself returns the uncapped arithmetic value.
CORRIDOR_DISPLAY_BAND = (8.0, 12.0)

# Empirical anchors quoted in the paper for the peak-shift slider.
INSINC_PEAK_SHIFT_PCT = 7.49   # INSINC field trial empirical anchor
JR_EAST_PEAK_SHIFT_PCT = 8.5   # JR East off-peak incentive programme


def compute_scenario(registered_share, active_share, peak_shift_share,
                     corridor_share=DEFAULT_CORRIDOR_SHARE,
                     energy_shift_share=None):
    """Compute the Section 6 congestion scenario from adoption shares.

    Parameters
    ----------
    registered_share : float   Fraction of travellers registered in the scheme.
    active_share : float       Fraction of the registered who actively respond.
    peak_shift_share : float   Fraction of an active user's peak trips shifted.
    corridor_share : float     Fraction of peak trips carried by the targeted
                               corridors (default 0.25). The network-wide shift
                               concentrates by 1/corridor_share on the corridor.
    energy_shift_share : float or None
                               Optional fraction of energy demand shifted; passed
                               through to energy_shift_pct (None -> None).

    Returns
    -------
    dict with:
        network_peak_reduction_pct : network-wide peak-trip reduction (%)
        corridor_relief_pct        : relief on the targeted corridor (%), uncapped
        energy_shift_pct           : energy-shift figure (%) or None
    """
    network_peak_reduction = registered_share * active_share * peak_shift_share

    if corridor_share <= 0:
        raise ValueError("corridor_share must be positive (it is a fraction of "
                         "peak trips carried by the targeted corridor).")
    corridor_relief = network_peak_reduction / corridor_share

    energy_shift_pct = None if energy_shift_share is None else energy_shift_share * 100.0

    return {
        "network_peak_reduction_pct": network_peak_reduction * 100.0,
        "corridor_relief_pct": corridor_relief * 100.0,
        "energy_shift_pct": energy_shift_pct,
    }


def display_corridor_relief_pct(corridor_relief_pct):
    """Clamp corridor relief to the paper's reported 8-12% display band.

    The raw arithmetic can exceed the band (e.g. 14.4% at the HIGH preset); the
    paper reports corridor relief capped at this ceiling, so the UI shows the
    clamped figure while compute_scenario keeps the exact value.
    """
    return min(corridor_relief_pct, CORRIDOR_DISPLAY_BAND[1])


# --- Section 6 presets --------------------------------------------------------
# Each preset is the full keyword set for compute_scenario; corridor_share is
# left at its 0.25 default.

LOW = {
    "registered_share": 0.05,
    "active_share": 0.30,
    "peak_shift_share": 0.02,
    "energy_shift_share": 0.05,
}

MEDIUM = {
    "registered_share": 0.20,
    "active_share": 0.45,
    "peak_shift_share": 0.075,
    "energy_shift_share": 0.20,
}

HIGH = {
    "registered_share": 0.40,
    "active_share": 0.60,
    "peak_shift_share": 0.15,
    "energy_shift_share": 0.40,
}

PRESETS = {"Low": LOW, "Medium": MEDIUM, "High": HIGH}


if __name__ == "__main__":
    for name, preset in PRESETS.items():
        r = compute_scenario(**preset)
        print(f"{name:7} network={r['network_peak_reduction_pct']:.4f}%  "
              f"corridor={r['corridor_relief_pct']:.4f}%  "
              f"energy={r['energy_shift_pct']:.1f}%")
