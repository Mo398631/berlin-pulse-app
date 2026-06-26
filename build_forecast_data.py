"""
Berlin Pulse, Appendix E, Session 01: forecast data ingest.

Reads the SMARD day-ahead generation forecast CSV and writes a new parquet,
data/berlin_pulse_forecast_dataset.parquet, aligned hour-for-hour to the
validated parquet's timestamp_berlin index (the index is the authority).

Reuses _localize_berlin and _de_num from baseline_simulation.py (imported,
not rewritten) so the DST/number-format conventions stay identical.

Does NOT modify data/berlin_pulse_validated_dataset.parquet or any other
existing file.
"""

import pandas as pd

from baseline_simulation import _de_num, _localize_berlin

# ---- Paths -------------------------------------------------------------------

FORECAST_CSV = "data/Forecasted_generation_Day-Ahead_202501010000_202601010000_Hour.csv"
VALIDATED_PARQUET = "data/berlin_pulse_validated_dataset.parquet"
FORECAST_PARQUET = "data/berlin_pulse_forecast_dataset.parquet"

# SMARD raw forecast column -> output column name.
FORECAST_COLUMN_MAP = {
    "Wind onshore [MWh] Calculated resolutions": "forecast_wind_onshore",
    "Wind offshore [MWh] Calculated resolutions": "forecast_wind_offshore",
    "Photovoltaics [MWh] Calculated resolutions": "forecast_solar",
}
# Optional total-generation forecast column (present in the day-ahead CSV).
TOTAL_FORECAST_COLUMN = "Total [MWh] Calculated resolutions"


def build_forecast():
    # ---- Load the validated parquet (index authority) ------------------------
    validated = pd.read_parquet(VALIDATED_PARQUET)
    index_ts = validated["timestamp_berlin"]

    # ---- Read and parse the forecast CSV -------------------------------------
    fc = pd.read_csv(FORECAST_CSV, sep=";", dtype=str, encoding="utf-8-sig")

    # Use .to_numpy() so the DataFrame constructor positions values by row order
    # against fc_idx rather than aligning on the Series' integer index.
    parsed = {}
    for raw, clean in FORECAST_COLUMN_MAP.items():
        parsed[clean] = _de_num(fc[raw]).to_numpy()

    has_total = TOTAL_FORECAST_COLUMN in fc.columns
    if has_total:
        parsed["forecast_total_generation_mwh"] = _de_num(fc[TOTAL_FORECAST_COLUMN]).to_numpy()

    fc_idx = _localize_berlin(fc["Start date"])
    fc_frame = pd.DataFrame(parsed, index=fc_idx).sort_index()
    # Guard against any duplicated (fall-back DST) timestamps before reindex.
    fc_frame = fc_frame[~fc_frame.index.duplicated(keep="first")]

    # ---- Align hour-for-hour to the validated index --------------------------
    # The validated timestamp_berlin is the authority; reindex onto it so the
    # output is row-for-row comparable and never re-derived from the CSV.
    # Match the datetime resolution to the validated index (us) so the reindex
    # aligns on equal timestamps rather than mismatching units.
    target = pd.DatetimeIndex(index_ts)
    fc_frame.index = fc_frame.index.as_unit(target.unit)
    aligned = fc_frame.reindex(target)

    out = pd.DataFrame({"timestamp_berlin": index_ts.reset_index(drop=True)})
    for col in FORECAST_COLUMN_MAP.values():
        out[col] = aligned[col].values
    if has_total:
        out["forecast_total_generation_mwh"] = aligned["forecast_total_generation_mwh"].values
    else:
        print("NOTE: total generation forecast column not found in CSV; "
              "forecast_total_generation_mwh omitted.")

    # ---- Report source gaps, then fill to the validated convention -----------
    # The SMARD day-ahead forecast has a small number of '-' (missing) cells.
    # The validated pipeline (baseline_simulation.load_smard) fills missing
    # generation with 0.0; mirror that convention exactly here so the forecast
    # parquet aligns with the validated one. Filled cells are reported below so
    # the source gaps are visible, not silently masked.
    value_cols = [c for c in out.columns if c != "timestamp_berlin"]
    filled_report = []
    for col in value_cols:
        gap_mask = out[col].isna()
        n_gap = int(gap_mask.sum())
        if n_gap:
            ts_list = out.loc[gap_mask, "timestamp_berlin"].tolist()
            filled_report.append((col, n_gap, ts_list))
    out[value_cols] = out[value_cols].fillna(0.0)

    if filled_report:
        print("SOURCE GAPS FILLED WITH 0.0 (validated-parquet convention):")
        for col, n_gap, ts_list in filled_report:
            shown = ", ".join(str(t) for t in ts_list[:5])
            more = "" if len(ts_list) <= 5 else f" (+{len(ts_list) - 5} more)"
            print(f"  {col}: {n_gap} cell(s) -> {shown}{more}")
    else:
        print("No source gaps: all forecast cells present before fill.")

    # ---- Derived: renewable sum (three filled forecast components) -----------
    out["forecast_renewable_mwh"] = (
        out["forecast_wind_onshore"]
        + out["forecast_wind_offshore"]
        + out["forecast_solar"]
    )

    out.to_parquet(FORECAST_PARQUET, index=False)
    return out, validated, has_total


def verify(out, validated, has_total):
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    fc = pd.read_parquet(FORECAST_PARQUET)
    val = pd.read_parquet(VALIDATED_PARQUET)

    print(f"forecast parquet rows : {len(fc)}")
    print(f"validated parquet rows: {len(val)}")
    same_rows = len(fc) == len(val)
    print(f"row counts match (expect 8760): {same_rows} "
          f"({'OK' if same_rows and len(fc) == 8760 else 'CHECK'})")

    # Index alignment check.
    idx_match = bool((fc["timestamp_berlin"].values == val["timestamp_berlin"].values).all())
    print(f"timestamp_berlin aligns to validated index: {idx_match}")

    print(f"forecast columns: {list(fc.columns)}")
    print(f"total generation forecast included: {has_total}")

    # ---- NaN check on in-window rows -----------------------------------------
    in_window = val["in_window"].values
    fc_cols = [c for c in fc.columns if c != "timestamp_berlin"]
    in_win_rows = fc.loc[in_window, fc_cols]
    nan_in_window = int(in_win_rows.isna().sum().sum())
    print(f"in-window rows: {int(in_window.sum())}")
    print(f"NaNs in in-window forecast cells: {nan_in_window} "
          f"({'OK' if nan_in_window == 0 else 'FAIL'})")
    if nan_in_window:
        print("  per-column in-window NaNs:")
        print(in_win_rows.isna().sum().to_string())

    # ---- Gap report ----------------------------------------------------------
    print("-" * 70)
    print("GAP REPORT (all rows)")
    print("-" * 70)
    total_nans = fc[fc_cols].isna().sum()
    print(total_nans.to_string())
    print(f"  total NaN cells (all rows) : {int(fc[fc_cols].isna().sum().sum())}")

    # Continuous-grid gap check on the Berlin timestamps.
    ts = pd.DatetimeIndex(fc["timestamp_berlin"]).tz_convert("UTC")
    full = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
    print(f"  expected hourly slots      : {len(full)}")
    print(f"  missing hours vs grid      : {len(full.difference(ts))}")
    print(f"  duplicate UTC hours        : {int(ts.duplicated(keep=False).sum())}")

    print("=" * 70)
    ok = same_rows and len(fc) == 8760 and idx_match and nan_in_window == 0
    print("FORECAST INGEST: " + ("PASS" if ok else "REVIEW NEEDED"))
    print("=" * 70)


if __name__ == "__main__":
    out, validated, has_total = build_forecast()
    verify(out, validated, has_total)
