"""
Qualitative-property test for unified_model (paper Appendix A).

Locks the *structural* claims Appendix A proves, not fragile decimals:

  1. W(rho) is strictly increasing in rho over [0, 1]            (Section A.4)
  2. dW/drho > 0 across [0, 1] and is non-increasing             (Eq. A.24f:
     diminishing returns - the gradient slopes down)               concavity)
  3. Channel separability (Theorem A.22 / Corollary A.23): the bus-block
     optimal value is identical at rho = 0.05, 0.20, 0.40 to within 1e-9.

These hold for the DEFAULT_PARAMS calibration, whose unconstrained optimum
rho* = A/B is placed beyond [0, 1] so the displayed registration interval lies
entirely on the rising arm of the concave welfare curve (see unified_model
docstring). The forms themselves are exactly Appendix A's.

Run:  python test_unified.py
"""

from unified_model import (
    welfare, marginal_welfare, bus_block_optimum,
    welfare_aggregates, optimum_rho, scenario_rhos,
)

# A fine grid over the unit interval for the monotonicity / concavity checks.
GRID = [i / 200.0 for i in range(201)]            # 0.00, 0.005, ..., 1.00
SEP_TOL = 1e-9                                     # separability tolerance


def test_qualitative_properties():
    A, B = welfare_aggregates()
    print("Unified model (Appendix A) - qualitative property test")
    print(f"  aggregates: A = {A:.4f}, B = {B:.4f}, rho* = A/B = {optimum_rho():.4f}")
    print(f"  (rho* lies {'beyond' if optimum_rho() > 1 else 'inside'} [0, 1] -> "
          f"unit interval is on the rising, concave arm)")
    print("-" * 64)

    results = []

    # --- Property 1: W(rho) strictly increasing over [0, 1] -------------------
    w = [welfare(r) for r in GRID]
    min_step = min(w[i + 1] - w[i] for i in range(len(w) - 1))
    inc_pass = min_step > 0.0
    results.append(inc_pass)
    print(f"  W(0) = {w[0]:.6f}   W(1) = {w[-1]:.6f}")
    print(f"  [1] W(rho) strictly increasing on [0,1]      "
          f"(min step = {min_step:.3e} > 0)   {_pf(inc_pass)}")

    # --- Property 2a: dW/drho > 0 across [0, 1] -------------------------------
    g = [marginal_welfare(r) for r in GRID]
    min_grad = min(g)
    pos_pass = min_grad > 0.0
    results.append(pos_pass)
    print(f"  [2a] dW/drho > 0 across [0,1]                "
          f"(min grad = {min_grad:.6f} > 0)   {_pf(pos_pass)}")

    # --- Property 2b: dW/drho non-increasing (diminishing returns) ------------
    # Allow a tiny finite-difference tolerance so float noise can't flip it.
    max_rise = max(g[i + 1] - g[i] for i in range(len(g) - 1))
    noninc_pass = max_rise <= 1e-9
    results.append(noninc_pass)
    print(f"  grad at rho=0: {g[0]:.6f}   grad at rho=1: {g[-1]:.6f}")
    print(f"  [2b] dW/drho non-increasing (diminishing)    "
          f"(max rise = {max_rise:.3e} <= 1e-9)   {_pf(noninc_pass)}")

    # --- Property 3: channel separability ------------------------------------
    bus_vals = [bus_block_optimum(r) for r in scenario_rhos()]
    spread = max(bus_vals) - min(bus_vals)
    sep_pass = spread <= SEP_TOL
    results.append(sep_pass)
    print(f"  W_bus* at rho = {scenario_rhos()}:")
    for r, v in zip(scenario_rhos(), bus_vals):
        print(f"        rho = {r:.2f}  ->  W_bus* = {v:.12f}")
    print(f"  [3] separability: W_bus* rho-invariant       "
          f"(spread = {spread:.2e} <= {SEP_TOL:.0e})   {_pf(sep_pass)}")

    print("-" * 64)

    # Hard assertions so the test fails loudly.
    assert inc_pass, f"W(rho) not strictly increasing (min step {min_step:.3e})"
    assert pos_pass, f"dW/drho not positive across [0,1] (min {min_grad:.6f})"
    assert noninc_pass, f"dW/drho not non-increasing (max rise {max_rise:.3e})"
    assert sep_pass, f"bus block not rho-invariant (spread {spread:.2e})"

    assert all(results)
    print("PASS: unified model reproduces the qualitative results of Appendix A.")


def _pf(ok):
    return "PASS" if ok else "FAIL"


if __name__ == "__main__":
    test_qualitative_properties()
