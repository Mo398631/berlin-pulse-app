"""
Appendix E, Session 02: Forecast Recovery tests.

Locks the reproduce-before-perturb contract and the Appendix E recovery results.
The reused Appendix B / Appendix D physics is re-asserted first (gate.main and
the golden-numbers guard), then the forecast-recovery figures are checked: the
operational share predictor must dominate, every naive forecast and the random
null must fall below it, and the within-night rank correlations must separate
the operational signal (>0.50) from the naive forecasts (<0.10).

Run:
    python -m pytest test_forecast.py -v
"""

import importlib.util
from pathlib import Path

import pytest

import gate
from baseline_simulation import complete_nights
from gate import night_sets, PARQUET

import pandas as pd

from forecast_engine import compute_forecast_recovery

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def results():
    return compute_forecast_recovery()


# ---- reproduce-before-perturb: reused physics must still hold ----------------

def test_gate_main_returns_true():
    assert bool(gate.main()) is True


def test_golden_numbers_pass():
    """The standalone golden-numbers guard must run its assertions clean."""
    path = ROOT / "scripts_check" / "golden_numbers.py"
    spec = importlib.util.spec_from_file_location("golden_numbers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run()   # raises AssertionError on any failed golden number


def test_night_counts():
    df = pd.read_parquet(PARQUET).set_index("timestamp_berlin")
    full, train, q4 = night_sets(df)
    assert len(complete_nights(df)) == 364
    assert len(full) == 364
    assert len(train) == 273
    assert len(q4) == 91


# ---- bookends carried through the engine -------------------------------------

def test_engine_night_counts(results):
    assert results["full_nights"] == 364
    assert results["training_nights"] == 273
    assert results["test_nights"] == 91


def test_deployable_carbon_q4_zero(results):
    assert abs(results["bookends"]["deployable_carbon_q4_pct"] - 0.000) <= 1e-6


def test_carbon_oracle_q4(results):
    assert abs(results["bookends"]["carbon_oracle_q4_pct"] - 3.756) <= 0.014


def test_recovery_bookends(results):
    """Blind rule == 0% floor, oracle == 100% ceiling, by construction."""
    assert abs(results["bookends"]["blind_recovery"] - 0.0) < 1e-9
    assert abs(results["bookends"]["oracle_recovery"] - 1.0) < 1e-4


# ---- Appendix E recovery results ---------------------------------------------

def test_share_predictor_recovers_q4(results):
    """The E.3 share predictor must recover most of the oracle (expect ~0.905)."""
    assert results["operational"]["share"]["q4_recovery"] > 0.80


def test_naive_below_share(results):
    """Both naive pillars recover less of Q4 than the share predictor."""
    share = results["operational"]["share"]["q4_recovery"]
    for name in ("persistence", "climatology"):
        assert results["naive"][name]["q4_recovery"] < share


def test_random_null_below_share(results):
    share = results["operational"]["share"]["q4_recovery"]
    assert results["random_null"]["q4_recovery"] < share


def test_rank_correlation_separation(results):
    """Operational share signal ranks hours; naive forecasts essentially do not."""
    assert results["operational"]["share"]["rank_corr"] > 0.50
    for name in ("persistence", "climatology"):
        assert results["naive"][name]["rank_corr"] < 0.10


# ---- gap-cell repair ---------------------------------------------------------

def test_gap_cell_interpolated(results):
    """The Oct-26 midnight gap is repaired by day-ahead interpolation, not 0.0."""
    gap = results["gap_cell"]
    assert gap["interpolated"] is True
    assert gap["timestamp"].startswith("2025-10-26 00:00:00")
    # wind_onshore was the 0.0-filled cell; interpolation lifts it well above 0.
    assert gap["before"]["forecast_wind_onshore"] == 0.0
    assert gap["after"]["forecast_wind_onshore"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
