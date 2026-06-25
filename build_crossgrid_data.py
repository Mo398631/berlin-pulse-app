"""DEV-ONLY: build cross-grid comparison parquet files for the Tab 8 prototype.

Reads hand-downloaded ENTSO-E CSV exports from crossgrid_raw/ and normalizes four
European grids (FR, PL, NO2, ES) into small committed parquet files under
crossgrid_data/. Pure pandas/numpy, no network calls.

This script touches NO existing tab, engine, or test. It only produces data.

Run:  python build_crossgrid_data.py
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd

import baseline_simulation

RAW_DIR = "crossgrid_raw"
OUT_DIR = "crossgrid_data"

# (filename token, output code, IANA timezone)
GRIDS = [
    ("FRANCE", "FR", "Europe/Paris"),
    ("POLAND", "PL", "Europe/Warsaw"),
    ("NORWAY_2", "NO2", "Europe/Oslo"),
    ("SPAIN", "ES", "Europe/Madrid"),
]

# The 12 canonical generation columns, in the exact order they appear in
# data/berlin_pulse_validated_dataset.parquet.
CANONICAL_COLUMNS = [
    "biomass",
    "hydropower",
    "wind_offshore",
    "wind_onshore",
    "photovoltaics",
    "other_renewable",
    "nuclear",
    "lignite",
    "hard_coal",
    "fossil_gas",
    "pumped_storage",
    "other_conventional",
]

# ENTSO-E "Production Type" -> canonical column.
PRODUCTION_TYPE_MAP = {
    "Biomass": "biomass",
    "Hydro Run-of-river and pondage": "hydropower",
    "Hydro Water Reservoir": "hydropower",
    "Wind Offshore": "wind_offshore",
    "Wind Onshore": "wind_onshore",
    "Solar": "photovoltaics",
    "Geothermal": "other_renewable",
    "Other renewable": "other_renewable",
    "Marine": "other_renewable",
    "Nuclear": "nuclear",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Hard coal": "hard_coal",
    "Fossil Gas": "fossil_gas",
    "Hydro Pumped Storage": "pumped_storage",
    "Energy storage": "pumped_storage",
    "Fossil Oil": "other_conventional",
    "Fossil Oil shale": "other_conventional",
    "Fossil Coal-derived gas": "other_conventional",
    "Fossil Peat": "other_conventional",
    "Waste": "other_conventional",
    "Other": "other_conventional",
}

# First "DD/MM/YYYY HH:MM:SS" token inside an MTU cell.
_DT_RE = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")


def _find_file(token, name_part):
    """Locate the single raw CSV whose name contains both name_part and token."""
    matches = [
        f
        for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if name_part in os.path.basename(f) and token in os.path.basename(f)
    ]
    if len(matches) != 1:
        print(
            f"ERROR: expected exactly one file with '{name_part}' and '{token}', "
            f"found {len(matches)}: {matches}"
        )
        sys.exit(1)
    return matches[0]


def _extract_start(mtu_series):
    """Extract the interval START datetime from each MTU cell (naive)."""
    starts = mtu_series.str.split(" - ", n=1).str[0]
    dt_token = starts.str.extract(_DT_RE, expand=False)
    return pd.to_datetime(dt_token, format="%d/%m/%Y %H:%M:%S")


def _to_numeric(series):
    """Coerce to numeric; blanks / '-' / 'n/e' -> 0."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "n/e": np.nan, "N/A": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def load_generation(path, tz):
    """Load a generation CSV -> hourly-mean wide frame (UTC index, 12 canon cols, MW)."""
    df = pd.read_csv(path, dtype=str)

    # 1. interval START (naive local clock)
    start_naive = _extract_start(df["MTU (CET/CEST)"])

    # 2. DST-ambiguity flag from the suffix attached to the START portion.
    start_part = df["MTU (CET/CEST)"].str.split(" - ", n=1).str[0]
    ambiguous = np.where(
        start_part.str.contains("CEST"),
        True,
        np.where(start_part.str.contains("CET"), False, True),
    ).astype(bool)

    # 3. localize naive local -> UTC
    idx_local = pd.DatetimeIndex(start_naive).tz_localize(
        tz, ambiguous=ambiguous, nonexistent="shift_forward"
    )
    utc = idx_local.tz_convert("UTC")

    # 4. numeric generation
    gen = _to_numeric(df["Generation (MW)"])

    # 5. map production type -> canonical; halt on anything unmapped
    ptype = df["Production Type"]
    unknown = sorted(set(ptype.dropna().unique()) - set(PRODUCTION_TYPE_MAP))
    if unknown:
        for name in unknown:
            print(f"ERROR: unmapped Production Type: {name!r}")
        sys.exit(1)
    canon = ptype.map(PRODUCTION_TYPE_MAP)

    work = pd.DataFrame({"utc": utc, "canon": canon, "gen": gen.values})

    # 6. pivot wide on UTC, summing fuels that share a canonical column
    wide = work.pivot_table(
        index="utc", columns="canon", values="gen", aggfunc="sum"
    )
    for col in CANONICAL_COLUMNS:
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[CANONICAL_COLUMNS].fillna(0.0)

    # 7. resample to hourly MEAN (15-min MW averaged over the hour)
    hourly = wide.resample("h").mean()
    return hourly


def load_prices(path):
    """Load a price CSV -> hourly-mean dayahead_price (UTC index)."""
    df = pd.read_csv(path, dtype=str)

    # 1. keep only "Without Sequence"
    df = df[df["Sequence"].str.strip() == "Without Sequence"].copy()

    # 2. interval START, already UTC -> localize as UTC
    start_naive = _extract_start(df["MTU (UTC)"])
    utc = pd.DatetimeIndex(start_naive).tz_localize("UTC")

    # 3. numeric price, hourly mean
    price = _to_numeric(df["Day-ahead Price (EUR/MWh)"])
    out = pd.DataFrame({"dayahead_price": price.values}, index=utc)
    hourly = out.resample("h").mean()
    return hourly


def build_grid(token, code, tz):
    gen_path = _find_file(token, "AGGREGATED_GENERATION_PER_TYPE")
    price_path = _find_file(token, "GUI_ENERGY_PRICES")

    gen = load_generation(gen_path, tz)
    price = load_prices(price_path)

    # join on the hourly UTC index
    merged = gen.join(price, how="inner")

    # to local tz, trim to civil year 2025, sort
    merged.index = merged.index.tz_convert(tz)
    merged = merged.sort_index()
    year_start = pd.Timestamp("2025-01-01", tz=tz)
    year_end = pd.Timestamp("2026-01-01", tz=tz)
    merged = merged[(merged.index >= year_start) & (merged.index < year_end)]

    # assemble output: timestamp_local, 12 canon cols, dayahead_price
    out = merged.reset_index().rename(columns={"index": "timestamp_local"})
    out = out.rename(columns={out.columns[0]: "timestamp_local"})
    out = out[["timestamp_local"] + CANONICAL_COLUMNS + ["dayahead_price"]]

    # match reference dtypes: float64 gen cols + us-precision tz-aware timestamp
    for col in CANONICAL_COLUMNS + ["dayahead_price"]:
        out[col] = out[col].astype("float64")
    out["timestamp_local"] = out["timestamp_local"].astype(f"datetime64[us, {tz}]")

    return out


def carbon_intensity(df):
    """Annual mean intensity = sum(gen*factor) / sum(gen) over the 12 canon cols."""
    total_emissions = 0.0
    total_gen = 0.0
    for col in CANONICAL_COLUMNS:
        total_emissions += (df[col] * baseline_simulation.EMISSION_FACTORS[col]).sum()
        total_gen += df[col].sum()
    return total_emissions / total_gen if total_gen else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = []
    for token, code, tz in GRIDS:
        df = build_grid(token, code, tz)

        # validation
        n = len(df)
        if not (8758 <= n <= 8761):
            print(f"ERROR: {code} has {n} rows, expected ~8760 (8758-8761).")
            sys.exit(1)
        neg_cells = int((df[CANONICAL_COLUMNS] < 0).sum().sum())
        assert neg_cells == 0, f"{code}: {neg_cells} negative generation cells"
        nan_price = int(df["dayahead_price"].isna().sum())

        out_path = os.path.join(OUT_DIR, f"{code}.parquet")
        df.to_parquet(out_path, index=False)

        summary.append(
            {
                "code": code,
                "rows": n,
                "nan_price": nan_price,
                "neg_gen_cells": neg_cells,
                "mean_intensity": carbon_intensity(df),
            }
        )

    # summary table
    print()
    print("=" * 64)
    print("CROSS-GRID DATA SUMMARY")
    print("=" * 64)
    print(f"{'code':<6}{'rows':>8}{'nan_price':>12}{'neg_gen':>10}{'mean_gCO2/kWh':>16}")
    print("-" * 64)
    for s in summary:
        print(
            f"{s['code']:<6}{s['rows']:>8}{s['nan_price']:>12}"
            f"{s['neg_gen_cells']:>10}{s['mean_intensity']:>16.1f}"
        )
    print("=" * 64)
    print(f"Wrote {len(summary)} parquet files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
