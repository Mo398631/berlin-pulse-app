"""
build_prototype_data.py  --  DEV-ONLY, run once on your machine.

Builds the small illustrative GeoJSON inputs for the (future) sixth Streamlit
tab: a map-based simulation of the congestion-redirection idea.

It does NOT touch any existing tab, engine, or test. It only WRITES data:

    prototype_data/corridors.geojson   (mandatory output)
    prototype_data/bus_lines.geojson   (real shapes, or a deferred-note file)

Heavy / dev-only intermediates are written under gitignored folders:

    prototype_raw/   raw OSMnx street graph (*.graphml)
    gtfs_raw/        downloaded VBB GTFS feed

Requirements (DEV-ONLY -- intentionally NOT in requirements.txt):
    osmnx >= 2.0, geopandas, shapely, pandas, requests

Run it with the project venv:
    .venv/Scripts/python.exe build_prototype_data.py

OSMnx 2.x API note: graph_from_bbox / features_from_bbox take a SINGLE bbox
tuple ordered (left, bottom, right, top) == (west, south, east, north).
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent

RAW_DIR = HERE / "prototype_raw"      # gitignored, large, dev-only
GTFS_DIR = HERE / "gtfs_raw"          # gitignored, large, dev-only
OUT_DIR = HERE / "prototype_data"     # small, committed

# Raw street-graph bbox (the area the prototype map will draw), as requested.
# OSMnx 2.x order: (left, bottom, right, top) = (west, south, east, north).
GRAPH_BBOX = (13.30, 52.47, 13.50, 52.56)

# Slightly wider bbox used only to harvest full corridor geometry from OSM,
# so arterials that run past the map edge are not truncated mid-extraction.
FEATURES_BBOX = (13.25, 52.44, 13.55, 52.58)

# Illustrative demand model -- DOCUMENTED FIXED SEED so the synthetic baseline
# peak-hour trip counts are reproducible. The numbers below are NOT measured
# traffic; they exist only to drive a visual redirection demo.
DEMAND_SEED = 42
PEAK_RANGE_TARGETED = (2600, 4200)    # busier, "targeted" arterials
PEAK_RANGE_NORMAL = (1100, 2400)      # the rest

# GTFS is OPTIONAL this session. Corridors are the mandatory output; if the
# GTFS download/parse is troublesome, we ship a deferred-note bus file.
#
# The permanent mirror serves the feed as individual CSV files (not a zip),
# transparently zstd-compressed, and uses EXTENDED GTFS route types (city
# buses are 700-799, not the classic 3).
ATTEMPT_GTFS = True
GTFS_BASE = "https://vbb-gtfs.jannisr.de/latest/"
GTFS_DOWNLOAD_DEADLINE_S = 300        # abort a single file download past this
GTFS_PARSE_DEADLINE_S = 300           # abort the shapes.csv scan past this

# A few real BVG bus lines that run along the corridors. We extract one
# representative shape per line. (route_short_name -> corridor display name)
BUS_LINE_TO_CORRIDOR = {
    "300": "Karl-Marx-Allee",
    "M29": "Kurfuerstendamm",
    "M41": "Sonnenallee",
    "M48": "Potsdamer Strasse",
}

# The 6-8 arterial corridors. 'targeted' marks the more congested ones that
# the redirection policy would act on. ASCII names are matched against OSM
# names via umlaut/ess-zett normalization (see _norm).
CORRIDORS = [
    {"name": "Karl-Marx-Allee",   "targeted": True},
    {"name": "Frankfurter Allee", "targeted": True},
    {"name": "Kurfuerstendamm",   "targeted": True},
    {"name": "Leipziger Strasse", "targeted": False},
    {"name": "Sonnenallee",       "targeted": True},
    {"name": "Schoenhauser Allee", "targeted": False},
    {"name": "Tempelhofer Damm",  "targeted": False},
    {"name": "Potsdamer Strasse", "targeted": True},
]

MIN_CORRIDORS_REQUIRED = 5

SYNTHETIC_NOTE = (
    "ILLUSTRATIVE: baseline_peak is synthetic demand generated with a fixed "
    f"random seed ({DEMAND_SEED}); it is NOT measured traffic data."
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Normalize a street name for matching: lowercase, strip, fold German
    umlauts and ess-zett to their ASCII digraphs."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def _human_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Step 1 -- raw drivable street network
# --------------------------------------------------------------------------- #

def download_street_graph() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    out = RAW_DIR / "berlin_central_drive.graphml"
    if out.exists():
        log(f"[1/5] Street graph already present: {out.name} "
            f"({_human_size(out)}) -- skipping download.")
        return
    log(f"[1/5] Downloading drivable street network for bbox {GRAPH_BBOX} ...")
    G = ox.graph_from_bbox(GRAPH_BBOX, network_type="drive")
    ox.save_graphml(G, out)
    log(f"      Saved {out.name} ({_human_size(out)}); "
        f"{len(G.nodes)} nodes, {len(G.edges)} edges.")


# --------------------------------------------------------------------------- #
# Step 2 -- corridor geometries from OSM
# --------------------------------------------------------------------------- #

def extract_corridors() -> gpd.GeoDataFrame:
    log("[2/5] Harvesting named arterials from OSM ...")
    tags = {"highway": ["motorway", "trunk", "primary", "secondary",
                        "tertiary", "residential", "living_street",
                        "unclassified"]}
    feats = ox.features_from_bbox(FEATURES_BBOX, tags)
    # keep only line geometries that carry a name
    feats = feats[feats.geometry.type.isin(["LineString", "MultiLineString"])]
    if "name" not in feats.columns:
        raise RuntimeError("OSM features returned no 'name' column.")
    feats = feats[feats["name"].notna()].copy()
    feats["_norm"] = feats["name"].map(_norm)

    rng = np.random.default_rng(DEMAND_SEED)
    rows = []
    for c in CORRIDORS:
        target = _norm(c["name"])
        sel = feats[feats["_norm"] == target]
        if sel.empty:
            warn(f"corridor '{c['name']}' did not resolve in OSM -- SKIPPING.")
            continue
        merged = linemerge(unary_union(list(sel.geometry)))
        if merged.is_empty:
            warn(f"corridor '{c['name']}' produced empty geometry -- SKIPPING.")
            continue
        lo, hi = PEAK_RANGE_TARGETED if c["targeted"] else PEAK_RANGE_NORMAL
        baseline_peak = int(rng.integers(lo, hi))
        rows.append({
            "name": c["name"],
            "osm_name": str(sel.iloc[0]["name"]),
            "targeted": bool(c["targeted"]),
            "baseline_peak": baseline_peak,
            "synthetic": True,
            "demand_note": SYNTHETIC_NOTE,
            "geometry": merged,
        })
        log(f"      + {c['name']:<20} segments={len(sel):>3}  "
            f"baseline_peak={baseline_peak:>5}  "
            f"targeted={c['targeted']}")

    if len(rows) < MIN_CORRIDORS_REQUIRED:
        raise RuntimeError(
            f"Only {len(rows)} corridor(s) resolved, need at least "
            f"{MIN_CORRIDORS_REQUIRED}. Stopping. Check OSM name matching / "
            f"network connectivity.")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # approximate length in km (metric CRS for Berlin: ETRS89 / UTM 33N)
    gdf["length_km"] = (gdf.to_crs(25833).length / 1000.0).round(2)
    return gdf


# --------------------------------------------------------------------------- #
# Step 3 -- VBB GTFS bus shapes (optional / deferrable)
# --------------------------------------------------------------------------- #

def _fetch_gtfs_file(name: str) -> Path | None:
    """Stream one GTFS CSV from the mirror into gtfs_raw/, honoring the
    download deadline. requests transparently decodes the zstd encoding."""
    GTFS_DIR.mkdir(exist_ok=True)
    out = GTFS_DIR / name
    if out.exists() and out.stat().st_size > 0:
        log(f"      {name} already present ({_human_size(out)}).")
        return out
    url = GTFS_BASE + name
    start = time.time()
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    if time.time() - start > GTFS_DOWNLOAD_DEADLINE_S:
                        raise TimeoutError(
                            f"{name} download exceeded "
                            f"{GTFS_DOWNLOAD_DEADLINE_S}s")
            tmp.replace(out)
        log(f"      Fetched {name} ({_human_size(out)}) in "
            f"{time.time() - start:.0f}s.")
        return out
    except Exception as exc:  # noqa: BLE001 -- deferral is an accepted outcome
        warn(f"download of {name} failed: {exc}")
        return None


def _is_bus(route_type) -> bool:
    """True for buses under classic (3) or extended (700-799) GTFS types."""
    try:
        v = int(route_type)
    except (TypeError, ValueError):
        return False
    return v == 3 or 700 <= v <= 799


def extract_bus_lines(corridors: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """Return a GeoDataFrame of bus shapes, or None to signal deferral."""
    if not ATTEMPT_GTFS:
        warn("ATTEMPT_GTFS is False -- buses deferred by configuration.")
        return None

    routes_p = _fetch_gtfs_file("routes.csv")
    trips_p = _fetch_gtfs_file("trips.csv")
    if routes_p is None or trips_p is None:
        return None

    try:
        routes = pd.read_csv(routes_p, dtype=str)
        wanted = set(BUS_LINE_TO_CORRIDOR.keys())
        rmask = routes["route_type"].map(_is_bus) & \
            routes["route_short_name"].isin(wanted)
        routes = routes[rmask]
        if routes.empty:
            warn("none of the curated bus lines found in feed -- deferring.")
            return None
        rid_to_short = dict(zip(routes["route_id"], routes["route_short_name"]))

        trips = pd.read_csv(trips_p, dtype=str)
        trips = trips[trips["route_id"].isin(rid_to_short)]
        if "shape_id" not in trips.columns or trips.empty:
            warn("no usable trips/shape_id for curated lines -- deferring.")
            return None
        # one representative (most common) shape per bus line
        shape_for_short: dict[str, str] = {}
        for rid, grp in trips.groupby("route_id"):
            short = rid_to_short[rid]
            if short in shape_for_short:
                continue
            shape_for_short[short] = grp["shape_id"].value_counts().index[0]
        wanted_shapes = set(shape_for_short.values())
    except Exception as exc:  # noqa: BLE001
        warn(f"routes/trips parsing failed: {exc} -- deferring buses.")
        return None

    # shapes.csv is the large file -- fetch then scan in chunks, keeping only
    # the handful of shapes we need.
    shapes_p = _fetch_gtfs_file("shapes.csv")
    if shapes_p is None:
        return None
    try:
        log(f"      Scanning shapes.csv for {len(wanted_shapes)} shapes ...")
        start = time.time()
        collected: dict[str, list[tuple]] = {s: [] for s in wanted_shapes}
        reader = pd.read_csv(shapes_p, dtype=str, chunksize=500_000)
        for chunk in reader:
            hit = chunk[chunk["shape_id"].isin(wanted_shapes)]
            for _, row in hit.iterrows():
                collected[row["shape_id"]].append((
                    int(row["shape_pt_sequence"]),
                    float(row["shape_pt_lon"]),
                    float(row["shape_pt_lat"]),
                ))
            if time.time() - start > GTFS_PARSE_DEADLINE_S:
                warn(f"shapes.csv scan exceeded {GTFS_PARSE_DEADLINE_S}s "
                     f"-- deferring buses.")
                return None

        rows = []
        for short, shape_id in shape_for_short.items():
            pts = sorted(collected.get(shape_id, []))
            if len(pts) < 2:
                warn(f"bus line {short}: shape {shape_id} had too few points.")
                continue
            line = LineString([(lon, lat) for _, lon, lat in pts])
            rows.append({
                "bus_line": short,
                "corridor": BUS_LINE_TO_CORRIDOR[short],
                "operator": "BVG",
                "shape_id": shape_id,
                "source": "VBB GTFS (vbb-gtfs.jannisr.de)",
                "geometry": line,
            })
        if not rows:
            warn("no bus shapes assembled -- deferring buses.")
            return None
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        log(f"      Extracted {len(gdf)} bus line shape(s).")
        return gdf
    except Exception as exc:  # noqa: BLE001 -- deferral is an accepted outcome
        warn(f"shapes parsing failed: {exc} -- deferring buses.")
        return None


def write_deferred_bus_file(path: Path) -> None:
    """Write a valid (empty) GeoJSON FeatureCollection with a deferral note."""
    fc = {
        "type": "FeatureCollection",
        "note": ("Bus lines deferred this session. Corridors are the mandatory "
                 "output; re-run build_prototype_data.py with ATTEMPT_GTFS=True "
                 "and a reachable VBB GTFS feed to populate bus shapes."),
        "features": [],
    }
    path.write_text(json.dumps(fc, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(exist_ok=True)

    # Step 1
    download_street_graph()

    # Step 2 -- mandatory
    corridors = extract_corridors()
    corridors_path = OUT_DIR / "corridors.geojson"
    # geopandas wants a fresh file; remove any stale one first
    if corridors_path.exists():
        corridors_path.unlink()
    corridors.to_file(corridors_path, driver="GeoJSON")

    # Step 3/4 -- optional buses
    log("[3/5] Bus lines from VBB GTFS (optional this session) ...")
    buses = extract_bus_lines(corridors)
    bus_path = OUT_DIR / "bus_lines.geojson"
    if bus_path.exists():
        bus_path.unlink()
    buses_deferred = buses is None
    if buses_deferred:
        write_deferred_bus_file(bus_path)
        log("      Buses DEFERRED -- wrote empty bus_lines.geojson with a note.")
    else:
        buses.to_file(bus_path, driver="GeoJSON")

    # Step 5 -- summary
    n_corr = len(corridors)
    n_bus = 0 if buses_deferred else len(buses)
    log("")
    log("=" * 64)
    log("SUMMARY")
    log("=" * 64)
    log(f"Corridors written : {n_corr}")
    for _, r in corridors.iterrows():
        flag = "TARGETED" if r["targeted"] else "        "
        log(f"   - {r['name']:<20} {flag}  "
            f"peak={r['baseline_peak']:>5}  len={r['length_km']:>5} km")
    log(f"Bus lines written : {n_bus}"
        + ("  (DEFERRED -- corridors-only this session)" if buses_deferred
           else ""))
    if not buses_deferred:
        for _, r in buses.iterrows():
            log(f"   - line {r['bus_line']:<4} along {r['corridor']}")
    log("")
    log(f"Demand            : SYNTHETIC, fixed seed = {DEMAND_SEED}")
    log("Files:")
    log(f"   {corridors_path}  ({_human_size(corridors_path)})")
    log(f"   {bus_path}  ({_human_size(bus_path)})")
    log("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
