# data.py
# --------------------------------------------------------------------------------------
# Purpose
#   Centralize dataset access for the Clash Royale S18 project.
#   - Downloads (and caches) the Kaggle dataset via kagglehub
#   - Resolves expected file paths
#   - Loads the raw CSVs into pandas DataFrames
#   - Provides small utilities for introspection/debugging
#
# Why this exists
#   We want training/evaluation code to be model-pluggable. Moving I/O concerns here keeps
#   other modules (features, models, train) focused on their responsibilities.
#
# Usage (quick start)
#   from data import get_raw_data
#   daydf, cards, paths = get_raw_data()
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

from dataclasses import dataclass
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

# Default filenames we expect under the dataset root directory
DEFAULT_BATTLES_SUBDIR = "BattlesStaging_12272020_WL_tagged"
DEFAULT_BATTLES_FILE = "battlesStaging_12272020_WL_tagged.csv"
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
    battles_csv : Path
        Path to the per-battle CSV.
    cards_csv : Path
        Path to the card master list CSV.
    """

    base_dir: Path
    battles_csv: Path
    cards_csv: Path

    def exists(self) -> bool:
        """Return True if both key CSVs exist on disk."""
        return self.battles_csv.is_file() and self.cards_csv.is_file()

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


def resolve_paths(
    base_dir: Path,
    *,
    battles_subdir: str = DEFAULT_BATTLES_SUBDIR,
    battles_file: str = DEFAULT_BATTLES_FILE,
    cards_file: str = DEFAULT_CARDS_FILE,
) -> DatasetPaths:
    """Resolve canonical CSV file locations under a dataset root.

    If filenames ever change, pass explicit names via parameters, or implement a
    glob fallback here.

    Parameters
    ----------
    base_dir : Path
        Dataset root directory.
    battles_subdir : str
        Subdirectory containing the battles CSV.
    battles_file : str
        Filename for the battles CSV.
    cards_file : str
        Filename for the card master list CSV.

    Returns
    -------
    DatasetPaths
        Immutable container with resolved file paths.
    """
    battles_csv = base_dir / battles_subdir / battles_file
    cards_csv = base_dir / cards_file
    return DatasetPaths(base_dir=base_dir, battles_csv=battles_csv, cards_csv=cards_csv)


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

    Parameters
    ----------
    paths : DatasetPaths
        Resolved paths for the dataset files.
    low_memory : bool
        Passed through to pandas.read_csv for the battles CSV. Set False if you
        see dtype warnings and prefer a second pass.

    Returns
    -------
    (daydf, cards) : Tuple[pd.DataFrame, pd.DataFrame]
        daydf  — per-battle records (chronologically ordered in the source)
        cards  — card master list with card IDs and metadata
    """
    if not paths.exists():
        raise FileNotFoundError(
            f"Expected CSVs not found.\n  battles: {paths.battles_csv}\n  cards:   {paths.cards_csv}"
        )

    print(f"\nLoading: {paths.battles_csv.name}")
    daydf = pd.read_csv(paths.battles_csv, low_memory=low_memory)
    print(f"Shape: {daydf.shape}")

    print(f"\nLoading: {paths.cards_csv.name}")
    cards = pd.read_csv(paths.cards_csv)
    print(f"Shape: {cards.shape}")

    return daydf, cards

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

    Returns
    -------
    (daydf, cards, paths)
        daydf  — per-battle DataFrame
        cards  — card master list DataFrame
        paths  — DatasetPaths with resolved file locations
    """
    if use_local is not None:
        base = Path(use_local).expanduser().resolve()
        print(f"Using local dataset directory: {base}")
    else:
        base = dataset_download(handle, force_download=force_download, cache_dir=cache_dir)

    paths = resolve_paths(base)
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

    daydf, cards = load_raw(paths, low_memory=low_memory)

    # Optional: echo columns for quick inspection (comment out if too verbose)
    print("\nColumns (cards):")
    try:
        print(cards.columns.tolist())
    except Exception:
        pass

    return daydf, cards, paths


# --------------------------------------------------------------------------------------
# CLI test hook
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test when running this file directly.
    # - Change `use_local` to a local path if you want to avoid downloading.
    df_battles, df_cards, ds_paths = get_raw_data()
    print("\nDone. Dataset root:", ds_paths.base_dir)
