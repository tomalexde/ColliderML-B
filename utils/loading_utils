"""
Load track data from ColliderML parquet files (event-table format).

Uses Polars for fast vectorised exploding of list columns — much faster than
iterating row-by-row in pandas, especially for large tables like calo hits.
"""

import glob
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl


def _parse_events_range(filename: str) -> Optional[Tuple[int, int]]:
    """Parse eventsSTART-END from parquet filename. Returns (start, end) or None."""
    match = re.search(r"events(\d+)-(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _get_parquet_shards(path: str) -> List[Tuple[Path, int, int]]:
    """
    Resolve path to a list of (file_path, start_event, end_event).
    Path can be a directory (glob *events*.parquet) or a single .parquet file.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet" and p.exists():
        r = _parse_events_range(p.name)
        if r:
            return [(p.resolve(), r[0], r[1])]
        # No range in name: read and infer from event_id column
        df = pl.read_parquet(p, columns=["event_id"])
        if df.is_empty():
            return []
        start = int(df["event_id"].min())
        end = int(df["event_id"].max())
        return [(p.resolve(), start, end)]
    if p.is_dir():
        pattern = str(p / "*events*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            files = sorted(glob.glob(str(p / "*.parquet")))
        shards = []
        for f in files:
            r = _parse_events_range(f)
            if r:
                shards.append((Path(f).resolve(), r[0], r[1]))
        return shards
    return []


def _file_for_event(shards: List[Tuple[Path, int, int]], event_id: int) -> Optional[Path]:
    """Return path to the parquet file that contains event_id, or None."""
    for path, start, end in shards:
        if start <= event_id <= end:
            return path
    return None


def _load_events(base_dir: str, event_ids: List[int]) -> pl.DataFrame:
    """
    Read the minimal set of parquet shards needed for event_ids and return
    a Polars DataFrame filtered to those events (event-table format).
    """
    shards = _get_parquet_shards(base_dir)
    if not shards:
        raise FileNotFoundError(f"No parquet shards found at {base_dir}")
    event_ids = sorted(set(event_ids))
    file_to_events: dict = {}
    for eid in event_ids:
        filepath = _file_for_event(shards, eid)
        if filepath is not None:
            file_to_events.setdefault(filepath, []).append(eid)
    if not file_to_events:
        return pl.DataFrame()
    parts = []
    for filepath, eids in file_to_events.items():
        df = pl.read_parquet(filepath)
        parts.append(df.filter(pl.col("event_id").is_in(eids)))
    return pl.concat(parts)


def _explode_list_cols(df: pl.DataFrame, list_cols: List[str]) -> pd.DataFrame:
    """
    Explode all list_cols in lock-step (Polars vectorised) and return pandas.
    Columns not present in df are silently skipped.
    """
    cols = [c for c in list_cols if c in df.columns]
    return df.select(["event_id"] + cols).explode(cols).to_pandas()


def _add_pT(df: pd.DataFrame) -> pd.DataFrame:
    """Add pT = sin(theta) / abs(qop), NaN where qop == 0."""
    theta = df["theta"].values
    qop = df["qop"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        pT = np.sin(theta) / np.abs(qop)
    df = df.copy()
    df["pT"] = np.where(np.isfinite(pT), pT, np.nan)
    return df


# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

TRACK_LIST_COLS = ["d0", "z0", "phi", "theta", "qop", "majority_particle_id", "hit_ids", "track_id"]
HIT_LIST_COLS = [
    "x", "y", "z", "true_x", "true_y", "true_z",
    "time", "particle_id", "detector", "volume_id", "layer_id", "surface_id",
]
PARTICLE_LIST_COLS = [
    "particle_id", "pdg_id", "mass", "energy", "charge",
    "vx", "vy", "vz", "time",
    "px", "py", "pz",
    "perigee_d0", "perigee_z0",
    "vertex_primary", "parent_id", "primary",
]
CALO_CELL_COLS = ["detector", "total_energy", "x", "y", "z"]
CALO_CONTRIB_COLS = ["contrib_particle_ids", "contrib_energies", "contrib_times"]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def read_event_tracks(path: str, event_id: int) -> pd.DataFrame:
    """
    Read tracks for a single event from parquet (dir or single file).

    Returns:
        DataFrame with one row per track: event_id, d0, z0, phi, theta, qop,
        majority_particle_id, hit_ids, track_id, pT.
    """
    return read_events_tracks(path, [event_id])


def read_chunk_tracks(path: str) -> pd.DataFrame:
    """
    Read all events from one parquet file (or first shard if path is a directory).

    Returns:
        DataFrame with one row per track, all events in the file/shard, plus pT.
    """
    shards = _get_parquet_shards(path)
    if not shards:
        raise FileNotFoundError(f"No parquet shards found at {path}")
    df = pl.read_parquet(shards[0][0])
    out = _explode_list_cols(df, TRACK_LIST_COLS)
    return _add_pT(out)


def read_events_tracks(
    base_dir: str,
    event_ids: List[int],
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read tracks for specified events from parquet shards under base_dir.

    Returns:
        DataFrame with one row per track for requested events, plus pT.
    """
    df = _load_events(base_dir, event_ids)
    if df.is_empty():
        return pd.DataFrame()
    out = _explode_list_cols(df, TRACK_LIST_COLS)
    return _add_pT(out)


def read_events_hits(
    base_dir: str,
    event_ids: List[int],
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read tracker hits for specified events from parquet shards under base_dir.

    Returns:
        DataFrame with one row per hit: event_id, x, y, z, true_x, true_y, true_z,
        time, particle_id, detector, volume_id, layer_id, surface_id.
    """
    df = _load_events(base_dir, event_ids)
    if df.is_empty():
        return pd.DataFrame()
    return _explode_list_cols(df, HIT_LIST_COLS)


def read_events_particles(
    base_dir: str,
    event_ids: List[int],
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read particles for specified events from parquet shards under base_dir.

    Returns:
        DataFrame with one row per particle: event_id, particle_id, pdg_id,
        mass, energy, charge, vx, vy, vz, time, px, py, pz,
        perigee_d0, perigee_z0, vertex_primary, parent_id, primary.
    """
    df = _load_events(base_dir, event_ids)
    if df.is_empty():
        return pd.DataFrame()
    return _explode_list_cols(df, PARTICLE_LIST_COLS)


def read_events_calo_hits(
    base_dir: str,
    event_ids: List[int],
    dataset_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read calorimeter data for specified events from parquet shards under base_dir.

    Args:
        base_dir: Directory containing *events*.parquet shards.
        event_ids: Event IDs to read.
        dataset_name: Ignored (kept for API compatibility).

    Returns:
        (calo_cells, calo_contribs)
        - calo_cells: one row per calo cell: event_id, cell_index,
          detector, total_energy, x, y, z.
        - calo_contribs: one row per contributing particle: event_id,
          cell_index, particle_id, energy, time.
    """
    df = _load_events(base_dir, event_ids)
    if df.is_empty():
        return pd.DataFrame(), pd.DataFrame()

    cell_cols = [c for c in CALO_CELL_COLS if c in df.columns]
    contrib_cols = [c for c in CALO_CONTRIB_COLS if c in df.columns]

    # First explode: one row per calo cell (contrib_* still lists per cell)
    cells_lf = (
        df.select(["event_id"] + cell_cols + contrib_cols)
        .explode(cell_cols + contrib_cols)
        .with_columns(
            (pl.col("event_id").cum_count().over("event_id") - 1).alias("cell_index")
        )
    )

    cells = cells_lf.select(["event_id", "cell_index"] + cell_cols).to_pandas()

    # Second explode: one row per contribution (inner lists)
    contribs = (
        cells_lf.select(["event_id", "cell_index"] + contrib_cols)
        .explode(contrib_cols)
        .rename({
            "contrib_particle_ids": "particle_id",
            "contrib_energies": "energy",
            "contrib_times": "time",
        })
        .to_pandas()
    )

    return cells, contribs
