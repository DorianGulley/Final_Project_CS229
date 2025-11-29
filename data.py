# data.py
# --------------------------------------------------------------------------------------
# Purpose
#   Centralize dataset access for the Clash Royale S18 project.
#   - Downloads (and caches) the Kaggle dataset via kagglehub
#   - Resolves expected file paths
#   - Loads the raw CSVs into pandas DataFrames (ALL battle files by default)
#   - Provides small utilities for introspection/debugging
#
# Why this exists
#   We want training/evaluation code to be model-pluggable. Moving I/O concerns here keeps
#   other modules (features, models, train) focused on their responsibilities.
#
# Usage (quick start)
#   from data import get_raw_data
#   daydf, cards, paths = get_raw_data()           # loads ALL battle files (~21GB)
#   daydf, cards, paths = get_raw_data(quick=True) # loads only Dec 27 (~1GB, faster)
#
#   # If you already have the dataset locally (no download):
#   daydf, cards, paths = get_raw_data(use_local="/path/to/dataset/root")
#
#   # If you want to force a fresh pull of the cached dataset:
#   daydf, cards, paths = get_raw_data(force_download=True)
#
# Notes
#   - kagglehub caches under ~/.cache/kagglehub by default (configurable via KAGGLEHUB_CACHE).
#   - This module does *not* perform feature engineering or splitting. See features.py / datasets.py.
# --------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

# Optional import: provide a helpful error message if missing
try:
    import kagglehub  # type: ignore
except Exception as e:  # pragma: no cover
    kagglehub = None  # defer error until first use

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Pinned snapshot for reproducibility. Changing this will trigger a new download in cache.
DEFAULT_HANDLE = "bwandowando/clash-royale-season-18-dec-0320-dataset/versions/20"

# Quick-mode: single day file
QUICK_BATTLES_SUBDIR = "BattlesStaging_12272020_WL_tagged"
QUICK_BATTLES_FILE = "battlesStaging_12272020_WL_tagged.csv"

# Card master list
DEFAULT_CARDS_FILE = "CardMasterListSeason18_12082020.csv"

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPaths:
    """Container for dataset file locations.

    Attributes
    ----------
    base_dir : Path
        Root directory returned by kagglehub (or provided via use_local).
    battles_csvs : List[Path]
        List of battle CSV paths (one or more files).
    cards_csv : Path
        Path to the card master list CSV.
    """

    base_dir: Path
    battles_csvs: List[Path] = field(default_factory=list)
    cards_csv: Path = field(default=Path())

    def exists(self) -> bool:
        """Return True if cards CSV and at least one battles CSV exist on disk."""
        return self.cards_csv.is_file() and len(self.battles_csvs) > 0 and all(p.is_file() for p in self.battles_csvs)

# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------

def _ensure_kagglehub_available() -> None:
    """Raise a clear error if kagglehub is not importable.

    This defers the import error to runtime so unit tests or environments that
    don't need downloading can still import the module.
    """
    if kagglehub is None:  # pragma: no cover
        raise ImportError(
            "kagglehub is not installed. Install with `pip install kagglehub` or use "
            "get_raw_data(use_local=...) to point at an existing dataset directory."
        )


def dataset_download(
    handle: str = DEFAULT_HANDLE,
    *,
    force_download: bool = False,
    cache_dir: Optional[str] = None,
) -> Path:
    """Download (or retrieve from cache) the Kaggle dataset via kagglehub.

    Parameters
    ----------
    handle : str
        KaggleHub dataset handle. Keep pinned for reproducibility.
    force_download : bool
        If True, forces a fresh download even if cached.
    cache_dir : Optional[str]
        Optional override for the kagglehub cache directory (e.g., a larger drive).
        Set this before other modules so kagglehub respects it.

    Returns
    -------
    Path
        Local directory path containing the dataset files (cached or newly downloaded).
    """
    _ensure_kagglehub_available()

    # Allow users to steer the cache location at runtime if desired
    if cache_dir:
        # kagglehub uses the KAGGLEHUB_CACHE environment variable to choose cache root.
        import os
        os.environ.setdefault("KAGGLEHUB_CACHE", cache_dir)

    data_dir_str = kagglehub.dataset_download(handle, force_download=force_download)
    base = Path(data_dir_str)
    print(f"Local cache directory: {base}")
    return base


def discover_battle_csvs(base_dir: Path) -> List[Path]:
    """Find all battle CSV files in the dataset directory.

    Looks for files matching the pattern *WL_tagged*.csv in subdirectories
    that start with 'BattlesStaging' or 'battlesStaging'.

    Returns files sorted by filename (chronological order by date in filename).
    """
    battle_csvs = []
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and subdir.name.lower().startswith("battlesstaging"):
            for csv_file in subdir.glob("*.csv"):
                if "wl_tagged" in csv_file.name.lower():
                    battle_csvs.append(csv_file)
    
    # Sort by filename to get chronological order (dates are in filenames)
    return sorted(battle_csvs, key=lambda p: p.name.lower())


def resolve_paths(
    base_dir: Path,
    *,
    quick: bool = False,
    cards_file: str = DEFAULT_CARDS_FILE,
) -> DatasetPaths:
    """Resolve CSV file locations under a dataset root.

    Parameters
    ----------
    base_dir : Path
        Dataset root directory.
    quick : bool
        If True, only include the Dec 27, 2020 file (faster for iteration).
        If False (default), include ALL battle CSV files.
    cards_file : str
        Filename for the card master list CSV.

    Returns
    -------
    DatasetPaths
        Immutable container with resolved file paths.
    """
    cards_csv = base_dir / cards_file
    
    if quick:
        # Single-day mode: just Dec 27, 2020
        battles_csvs = [base_dir / QUICK_BATTLES_SUBDIR / QUICK_BATTLES_FILE]
    else:
        # Full mode: all battle files
        battles_csvs = discover_battle_csvs(base_dir)
    
    return DatasetPaths(base_dir=base_dir, battles_csvs=battles_csvs, cards_csv=cards_csv)


def list_csvs(base_dir: Path, limit: Optional[int] = 25) -> List[Path]:
    """List CSV files under the dataset root (for debugging/exploration).

    Parameters
    ----------
    base_dir : Path
        Dataset root directory.
    limit : Optional[int]
        If provided, truncate the returned list to the first `limit` files.

    Returns
    -------
    List[Path]
        Sorted list of CSV paths found under `base_dir`.
    """
    csvs = sorted(base_dir.rglob("*.csv"))
    return csvs[:limit] if limit is not None else csvs


def load_raw(
    paths: DatasetPaths,
    *,
    low_memory: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw CSVs into DataFrames.

    If multiple battle files are present, they are concatenated into a single
    DataFrame sorted by date (based on filename order).

    Parameters
    ----------
    paths : DatasetPaths
        Resolved paths for the dataset files.
    low_memory : bool
        Passed through to pandas.read_csv for the battles CSV. Set False if you
        see dtype warnings and prefer a second pass.

    Returns
    -------
    (battles_df, cards) : Tuple[pd.DataFrame, pd.DataFrame]
        battles_df — per-battle records (chronologically ordered)
        cards      — card master list with card IDs and metadata
    """
    if not paths.exists():
        raise FileNotFoundError(
            f"Expected CSVs not found.\n  battles: {paths.battles_csvs}\n  cards: {paths.cards_csv}"
        )

    # Load and concatenate all battle files
    print(f"\nLoading {len(paths.battles_csvs)} battle file(s)...")
    dfs = []
    total_rows = 0
    for csv_path in paths.battles_csvs:
        print(f"  Loading: {csv_path.name}", end="")
        df = pd.read_csv(csv_path, low_memory=low_memory)
        print(f" — {len(df):,} rows")
        total_rows += len(df)
        dfs.append(df)
    
    if len(dfs) == 1:
        battles_df = dfs[0]
    else:
        battles_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total battles: {len(battles_df):,} rows")

    print(f"\nLoading: {paths.cards_csv.name}")
    cards = pd.read_csv(paths.cards_csv)
    print(f"Shape: {cards.shape}")

    return battles_df, cards

# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------

def get_raw_data(
    *,
    handle: str = DEFAULT_HANDLE,
    use_local: Optional[str | Path] = None,
    force_download: bool = False,
    cache_dir: Optional[str] = None,
    low_memory: bool = True,
    quick: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetPaths]:
    """Obtain raw DataFrames and resolved paths, downloading if needed.

    This is the single function other modules should call.

    Parameters
    ----------
    handle : str
        KaggleHub dataset handle. Keep the default for reproducibility.
    use_local : Optional[str | Path]
        If provided, **skip download** and use this directory as the dataset root.
        Useful when you've already unpacked the dataset manually.
    force_download : bool
        If True, forces a fresh download (ignored if use_local is set).
    cache_dir : Optional[str]
        Optional override for kagglehub cache root (see dataset_download).
    low_memory : bool
        Passed to pandas.read_csv for the battles CSV.
    quick : bool
        If True, only load Dec 27, 2020 data (~1GB) for faster iteration.
        If False (default), load ALL battle files (~21GB).

    Returns
    -------
    (battles_df, cards, paths)
        battles_df — per-battle DataFrame (all days concatenated, or single day if quick=True)
        cards      — card master list DataFrame
        paths      — DatasetPaths with resolved file locations
    """
    if quick:
        print("Quick mode: loading only Dec 27, 2020 data (~1GB)")
    else:
        print("Full mode: loading ALL battle files (~21GB)")
    
    if use_local is not None:
        base = Path(use_local).expanduser().resolve()
        print(f"Using local dataset directory: {base}")
    else:
        base = dataset_download(handle, force_download=force_download, cache_dir=cache_dir)

    paths = resolve_paths(base, quick=quick)
    
    # Friendly hint if files are missing / the structure changed
    if not paths.exists():
        print("\n[Hint] CSVs not found at expected locations. Here's a quick directory peek:")
        for p in list_csvs(base, limit=50):
            print("  -", p.relative_to(base))
        raise FileNotFoundError(
            "Could not locate expected CSVs. If the dataset layout changed, call "
            "resolve_paths(...) with the new filenames, or pass use_local to the "
            "appropriate directory."
        )

    battles_df, cards = load_raw(paths, low_memory=low_memory)

    # Optional: echo columns for quick inspection (comment out if too verbose)
    print("\nColumns (cards):")
    try:
        print(cards.columns.tolist())
    except Exception:
        pass

    return battles_df, cards, paths


# --------------------------------------------------------------------------------------
# CLI test hook
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("--quick", action="store_true", help="Load only Dec 27, 2020 data")
    args = parser.parse_args()
    
    df_battles, df_cards, ds_paths = get_raw_data(quick=args.quick)
    print("\nDone. Dataset root:", ds_paths.base_dir)
    print(f"Loaded {len(ds_paths.battles_csvs)} battle file(s), {len(df_battles):,} total rows")
