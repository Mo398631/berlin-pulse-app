"""
Pure-logic engine for the Berlin Pulse Unified Model (paper Appendix A).

This module makes Appendix A - the social-planner optimization of Berlin Pulse -
interactive *at a conceptual level*. Like engine.py and scenario_engine.py it is
a *pure* module: no Streamlit, no file I/O, no global state, and it NEVER reads
the manuscript at run time. Every form and coefficient below is hard-coded as a
constant so the figures can be locked by test_unified.py.

It is a NUMERIC implementation (no sympy). The structure mirrors Appendix A:

  Welfare decomposes additively into two non-interacting channels (Eq. A.17):

      W(rho)  =  W_bus*            +   W_pax(rho)
                 (energy channel)      (congestion channel)
                 (Section A.5,          (Section A.4,
                  water-filling)         participation rho)

  CHANNEL SEPARABILITY (Theorem A.22 / Eq. A.18). The cross-partial Hessian of
  the two blocks vanishes, the constraint set factorizes, and the constrained
  maximum decomposes. The practical payoff (Corollary A.23): the bus-side
  optimum does NOT depend on the participation rate rho. We make this *testable*
  by passing rho into the bus-block optimizer through a single `coupling`
  coefficient that is zero whenever the decoupling assumptions (DA1)-(DA4) of
  Section A.3.2 hold. With coupling = 0 the bus optimum is rho-invariant to
  machine precision; a non-zero coupling is exactly the specification that
  Section A.3.6 shows would break separability.

  PASSENGER BLOCK (Section A.4). The welfare gradient is linear in rho
  (Eq. A.24f) and welfare is the concave quadratic that integrates it
  (Eq. A.24j):

      dW_pax/drho = A - B * rho                         (A.24f)
      W_pax(rho)  = A * rho - (B / 2) * rho**2          (A.24j, W_pax(0) = 0)

  with the two parameter aggregates built from the economic primitives
  (Eq. A.24e/f, general capacities):

      A = 2*alpha/K_peak + chi + sigma_dE - beta*dt
      B = 2*alpha*(1/K_peak + 1/K_off) + chi

  where alpha = crowding aversion, beta = late schedule-delay coefficient,
  dt = shift magnitude, chi = operator crowding-cost scaler, and
  sigma_dE = sigma*(eps_peak - eps_off) = social cost of carbon times the
  peak-vs-off-peak per-trip emission differential. B > 0 whenever alpha > 0, so
  W_pax is strictly concave (diminishing returns) and the gradient slopes down.

  The factor of two on alpha in A is the Wardrop-vs-system-optimum gap
  (Section A.2.5): the planner internalizes the crowding externality the
  individual passenger ignores.

  CONCEPTUAL ILLUSTRATION, not a forecast. The headline calibration in the paper
  (Fig. A.1) places the unconstrained optimum rho* = A/B at 0.05, so its welfare
  curve turns over almost immediately. For an interactive tool whose rho axis is
  the *registration rate* over the full [0, 1] interval, the DEFAULT_PARAMS below
  instead place rho* beyond the displayed range, so welfare rises monotonically
  with diminishing marginal returns across [0, 1] - the qualitative story the tab
  is illustrating. The functional forms are exactly Appendix A's; lowering the
  benefit sliders (or raising beta*dt) pulls rho* back into [0, 1] and reproduces
  the interior turning point of Fig. A.1. Values are illustrative, in alpha-units,
  and are NOT claimed to reflect Berlin's actual elasticities.

Public API:

    welfare(rho, params=None) -> float                 W(rho) = W_bus* + W_pax(rho)
    marginal_welfare(rho, params=None) -> float        dW/drho (central difference)
    bus_block_optimum(rho, params=None) -> float       W_bus* (rho-invariant)
    passenger_welfare(rho, params=None) -> float       W_pax(rho)
    welfare_aggregates(params=None) -> (A, B)          the two A.24f aggregates
    optimum_rho(params=None) -> float                  rho* = A/B (Eq. A.24i)
    scenario_rhos() -> (0.05, 0.20, 0.40)              Low / Medium / High
"""

# --- Scenario registration rates ---------------------------------------------
# The three Section 6 scenarios sampled as points along the single participation
# parameter rho (Section A.4.1). These are the Low/Medium/High *registration
# rates* (share of adult residents on the platform) - identical to the
# registered_share of scenario_engine.LOW/MEDIUM/HIGH.
SCENARIO_RHOS = (0.05, 0.20, 0.40)
SCENARIO_LABELS = ("Low", "Medium", "High")


# --- Default calibration (illustrative; see module docstring) ----------------
# Passenger-block economic primitives (Eq. A.24e/f). Defaults are chosen so the
# unconstrained optimum rho* = A/B sits beyond [0, 1], giving a strictly rising,
# concave (diminishing-returns) welfare curve over the displayed registration
# interval. See welfare_aggregates() for how these map to A and B.
DEFAULT_PARAMS = {
    "alpha": 0.10,     # crowding aversion (the alpha-unit numeraire is small here)
    "beta": 0.20,      # late schedule-delay coefficient
    "dt": 0.50,        # shift magnitude in hours (midpoint of the 15-30 min range)
    "chi": 0.10,       # operator crowding-cost scaler (quadratic C_T, Section A.4.2)
    "sigma_dE": 0.80,  # sigma * (eps_peak - eps_off): carbon damage differential
    "K_peak": 1.0,     # peak service capacity (normalisation, Section A.4.3)
    "K_off": 1.0,      # off-peak service capacity
    # Bus-block (energy-channel) parameters - merit-order water-filling.
    "sigma": 1.0,      # social-cost-of-carbon weight on grid intensity (Eq. A.9d)
    # Passenger->bus coupling. Zero under the decoupling assumptions (DA1)-(DA4);
    # this is the single knob whose non-zero value would break separability
    # (Section A.3.6). Kept at 0 so the bus optimum is rho-invariant.
    "coupling": 0.0,
}


# --- Energy-channel (bus block) data -----------------------------------------
# Overnight depot charging window (Appendix B): seven clock-hours 22:00 - 04:00.
# Per-hour training-window means are illustrative, in the spirit of Table B.1:
# a nearly-flat carbon profile that rises gently across the night, and a price
# profile that falls steeply to the small hours then ticks up. Anchored on the
# Table B.1 endpoints (carbon 376.6 g/kWh @22:00 -> 389.1 @04:00; price
# 110.21 EUR/MWh @22:00 -> 83.84 @03:00).
WINDOW_HOURS = (22, 23, 0, 1, 2, 3, 4)
CARBON_INTENSITY = (376.6, 378.7, 380.8, 382.9, 385.0, 387.0, 389.1)  # g CO2 / kWh
DAY_AHEAD_PRICE = (110.21, 103.0, 96.0, 90.0, 86.5, 83.84, 88.0)      # EUR / MWh

# Baseline depot configuration (Section 6.5 / Appendix B): 277 buses x 240 kWh
# at 50 kW chargers fill four hours fully plus one partial hour -> five of the
# seven window slots are used. The naive baseline (Strategy A) charges the
# earliest five clock-hours; the optimum water-fills the lowest-weight hours.
N_SLOTS_USED = 5


def welfare_aggregates(params=None):
    """Return the two passenger-block aggregates (A, B) of Eq. A.24f.

        A = 2*alpha/K_peak + chi + sigma_dE - beta*dt   (gradient intercept)
        B = 2*alpha*(1/K_peak + 1/K_off) + chi          (gradient slope, > 0)

    B collects the curvature from crowding-externality internalization (the
    2*alpha*(1/K_peak + 1/K_off) term) plus the convex operator cost (chi); it
    is strictly positive whenever alpha > 0, which is what makes W_pax strictly
    concave (Section A.4.3).
    """
    p = _merged(params)
    A = (2.0 * p["alpha"] / p["K_peak"]
         + p["chi"]
         + p["sigma_dE"]
         - p["beta"] * p["dt"])
    B = 2.0 * p["alpha"] * (1.0 / p["K_peak"] + 1.0 / p["K_off"]) + p["chi"]
    return A, B


def passenger_welfare(rho, params=None):
    """Congestion-channel welfare W_pax(rho) (Eq. A.24j), with W_pax(0) = 0.

        W_pax(rho) = A*rho - (B/2)*rho**2

    The concave quadratic that integrates the linear gradient A - B*rho.
    """
    A, B = welfare_aggregates(params)
    return A * rho - 0.5 * B * rho * rho


def bus_block_optimum(rho, params=None):
    """Energy-channel optimum W_bus* by merit-order water-filling (Eq. A.9d / A.25).

    The planner charges the fixed nightly energy into the lowest-weight hours,
    where each hour's weight is the joint cost-and-carbon merit-order price

        weight_t = price_t + sigma * carbon_t   (+ coupling * rho)

    (Eq. A.9d). The returned value is the fractional saving of this water-filling
    schedule over the naive earliest-hours baseline (Strategy A) - a number in
    the spirit of the paper's percentage savings.

    `rho` enters ONLY through `coupling`, which is zero under the decoupling
    assumptions (DA1)-(DA4) of Section A.3.2. With coupling = 0 the merit order,
    and hence W_bus*, is identical for every rho - the numeric signature of
    channel separability (Corollary A.23). A non-zero coupling is precisely the
    specification Section A.3.6 shows would contaminate the separation.
    """
    p = _merged(params)
    sigma = p["sigma"]
    coupling = p["coupling"]

    # Joint merit-order weight per window hour (Eq. A.9d).
    weights = [price + sigma * carbon + coupling * rho
               for price, carbon in zip(DAY_AHEAD_PRICE, CARBON_INTENSITY)]

    # Naive baseline: charge the earliest N_SLOTS_USED clock-hours (Strategy A).
    naive_cost = sum(weights[:N_SLOTS_USED])
    # Water-filling optimum: charge the N_SLOTS_USED lowest-weight hours.
    opt_cost = sum(sorted(weights)[:N_SLOTS_USED])

    # Fractional saving vs naive (>= 0; zero only if naive already optimal).
    return (naive_cost - opt_cost) / naive_cost


def welfare(rho, params=None):
    """Total social welfare W(rho) = W_bus* + W_pax(rho) (Eq. A.17).

    The bus block is a constant offset (the energy-channel optimum, independent
    of rho); the passenger block carries all the rho-dependence. So at rho = 0
    welfare already equals the pure energy-channel optimum, and passenger
    participation adds on top of it - the two channels stacking independently.
    """
    return bus_block_optimum(rho, params) + passenger_welfare(rho, params)


def marginal_welfare(rho, params=None, h=1e-6):
    """dW/drho by central finite difference (Eq. A.24f gives A - B*rho).

    Because the bus block is rho-invariant under separability it drops out of
    the derivative exactly (envelope theorem, Section A.2.6), leaving the
    passenger gradient A - B*rho. A finite difference is used per the brief; for
    the quadratic W_pax it is exact up to floating-point error.
    """
    return (welfare(rho + h, params) - welfare(rho - h, params)) / (2.0 * h)


def optimum_rho(params=None):
    """Unconstrained welfare-maximising participation rho* = A/B (Eq. A.24i).

    Returns A/B (the unique zero of the linear gradient). May lie outside [0, 1]:
    with DEFAULT_PARAMS it sits beyond 1, so welfare rises across the whole
    displayed interval (see module docstring).
    """
    A, B = welfare_aggregates(params)
    return A / B


def scenario_rhos():
    """The three Low/Medium/High registration rates (Section A.4.1)."""
    return SCENARIO_RHOS


def _merged(params):
    """Overlay user `params` on DEFAULT_PARAMS without mutating either."""
    if not params:
        return dict(DEFAULT_PARAMS)
    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    return merged


if __name__ == "__main__":
    A, B = welfare_aggregates()
    print("Unified model (Appendix A) - default calibration")
    print(f"  A = {A:.4f}   B = {B:.4f}   rho* = A/B = {optimum_rho():.4f}")
    print(f"  W_bus* (energy channel, rho-invariant) = {bus_block_optimum(0.0):.6f}")
    print()
    print(f"  {'rho':>6} {'W(rho)':>12} {'dW/drho':>12} {'W_bus*':>12} {'W_pax':>12}")
    for rho in (0.0, *SCENARIO_RHOS, 0.6, 1.0):
        print(f"  {rho:6.2f} {welfare(rho):12.6f} {marginal_welfare(rho):12.6f} "
              f"{bus_block_optimum(rho):12.6f} {passenger_welfare(rho):12.6f}")
