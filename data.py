# data.py
# Purpose: Dataset I/O and helpers (download, resolve paths, and load raw CSVs).

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from urllib.parse import urlparse

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

# Columns we actually need (extended to include levels)
REQUIRED_COLUMNS = [
    "winner.card1.id", "winner.card2.id", "winner.card3.id", "winner.card4.id",
    "winner.card5.id", "winner.card6.id", "winner.card7.id", "winner.card8.id",
    "loser.card1.id", "loser.card2.id", "loser.card3.id", "loser.card4.id",
    "loser.card5.id", "loser.card6.id", "loser.card7.id", "loser.card8.id",
    "winner.startingTrophies", "loser.startingTrophies",
    # Card levels (total sum of all 8 card levels per player)
    "winner.totalcard.level", "loser.totalcard.level",
]

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


def _parse_gs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a `gs://` URI into (bucket, prefix).

    Returns the bucket name and the prefix (without a leading slash). The prefix
    may be empty when pointing at the bucket root.
    """
    p = urlparse(gcs_uri)
    if p.scheme != "gs":
        raise ValueError("Expect gs:// URI")
    bucket = p.netloc
    prefix = p.path.lstrip("/")
    return bucket, prefix


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
        # Local/KaggleHub layout: file lives in a subdirectory
        battles_csvs = [base_dir / QUICK_BATTLES_SUBDIR / QUICK_BATTLES_FILE]
    else:
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
    sample: Optional[float] = None,
    random_state: int = 42,
    only_required_cols: bool = True,
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
    sample : Optional[float]
        If provided, sample this fraction of each CSV during loading.
        This reduces peak memory usage compared to sampling after loading.
    random_state : int
        Random seed for reproducible sampling.
    only_required_cols : bool
        If True (default), only load the 18 columns needed for feature building.
        This significantly speeds up loading and reduces memory usage.

    Returns
    -------
    (battles_df, cards) : Tuple[pd.DataFrame, pd.DataFrame]
        battles_df — per-battle records (chronologically ordered)
        cards      — card master list with card IDs and metadata
    """
    # If paths.cards_csv is a GCS URI (string), skip local exists checks —
    # we'll rely on pandas/gcsfs or the cloud client to surface missing files.
    cards_is_gs = isinstance(paths.cards_csv, str) and str(paths.cards_csv).startswith("gs://")
    if not cards_is_gs and not paths.exists():
        raise FileNotFoundError(
            f"Expected CSVs not found.\n  battles: {paths.battles_csvs}\n  cards: {paths.cards_csv}"
        )

    # Optimize loading by only reading needed columns
    usecols = REQUIRED_COLUMNS if only_required_cols else None
    
    # Specify dtypes for faster parsing (card IDs are ints, trophies are ints)
    dtype = {col: "int32" for col in REQUIRED_COLUMNS} if only_required_cols else None

    # Load and concatenate all battle files
    sample_msg = f" (sampling {sample:.0%} of each)" if sample else ""
    cols_msg = f", {len(REQUIRED_COLUMNS)} cols" if only_required_cols else ""
    print(f"\nLoading {len(paths.battles_csvs)} battle file(s){sample_msg}{cols_msg}...")
    dfs = []
    total_rows_original = 0
    total_rows_sampled = 0
    for csv_path in paths.battles_csvs:
        name = getattr(csv_path, "name", None) or str(csv_path).split("/")[-1]
        print(f"  Loading: {name}", end="", flush=True)
        df = pd.read_csv(csv_path, low_memory=low_memory, usecols=usecols, dtype=dtype)
        original_len = len(df)
        total_rows_original += original_len
        
        # Sample immediately to reduce memory
        if sample is not None and 0 < sample < 1:
            df = df.sample(frac=sample, random_state=random_state)
            print(f" — {original_len:,} → {len(df):,}")
        else:
            print(f" — {len(df):,}")
        
        total_rows_sampled += len(df)
        dfs.append(df)
    
    if len(dfs) == 1:
        battles_df = dfs[0]
    else:
        battles_df = pd.concat(dfs, ignore_index=True)
    
    if sample:
        print(f"Total: {total_rows_original:,} → {len(battles_df):,} rows ({sample:.0%} sampled)")
    else:
        print(f"Total battles: {len(battles_df):,} rows")

    cards_name = getattr(paths.cards_csv, "name", None) or str(paths.cards_csv).split("/")[-1]
    print(f"\nLoading: {cards_name}")
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
    sample: Optional[float] = None,
    random_state: int = 42,
    gcs_direct: bool = False,
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
    sample : Optional[float]
        If provided (e.g., 0.5 for 50%), sample each CSV during loading.
        This reduces peak memory usage significantly.
    random_state : int
        Random seed for reproducible sampling.

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
        use_local_str = str(use_local)
        # Option A: Direct read from GCS using gcsfs/pandas (no local download)
        if use_local_str.startswith("gs://") and gcs_direct:
            print(f"Using GCS dataset root (direct read): {use_local_str}")
            from google.cloud import storage

            bucket_name, prefix = _parse_gs_uri(use_local_str)
            client = storage.Client()

            # Ensure prefix ends with slash for consistent relative names
            if prefix and not prefix.endswith("/"):
                prefix = prefix + "/"

            # Collect battle CSVs and card file as gs:// URIs
            battles = []
            cards_uri = None
            bucket = client.bucket(bucket_name)

            # If quick mode is requested, try to directly reference the single-day CSV
            if quick:
                # Just the single CSV under the prefix root
                quick_rel = (prefix + QUICK_BATTLES_FILE) if prefix else QUICK_BATTLES_FILE
                quick_blob = bucket.blob(quick_rel)
                if quick_blob.exists(client=client):
                    battles = [f"gs://{bucket_name}/{quick_rel}"]
                else:
                    battles = []

            if not battles:
                # List blobs and collect matching CSVs and card file
                for blob in client.list_blobs(bucket_name, prefix=prefix):
                    name = blob.name[len(prefix):] if prefix else blob.name
                    if not name:
                        continue
                    lower = name.lower()
                    if lower.endswith(".csv") and "wl_tagged" in lower:
                        battles.append(f"gs://{bucket_name}/{blob.name}")
                    if name.endswith(DEFAULT_CARDS_FILE):
                        cards_uri = f"gs://{bucket_name}/{blob.name}"

            # If we didn't already locate cards_uri above, try a direct path under prefix
            if cards_uri is None:
                candidate = (prefix + DEFAULT_CARDS_FILE) if prefix else DEFAULT_CARDS_FILE
                if bucket.blob(candidate).exists(client=client):
                    cards_uri = f"gs://{bucket_name}/{candidate}"

            if not battles:
                raise FileNotFoundError(f"No battle CSVs found at {use_local_str}")
            if cards_uri is None:
                raise FileNotFoundError(f"Card master list {DEFAULT_CARDS_FILE} not found under {use_local_str}")

            paths = DatasetPaths(base_dir=use_local_str, battles_csvs=battles, cards_csv=cards_uri)

        # Option B: Download GCS prefix to a local tempdir (preserve existing behavior)
        elif use_local_str.startswith("gs://"):
            print(f"Using GCS dataset root: {use_local_str} — downloading to temporary directory...")
            from google.cloud import storage
            import tempfile

            bucket_name, prefix = _parse_gs_uri(use_local_str)
            client = storage.Client()

            # Create temp dir where objects will be downloaded
            tmpdir = Path(tempfile.mkdtemp(prefix="cr_dataset_"))
            print(f"  Temporary dataset dir: {tmpdir}")

            # Ensure prefix ends with slash for accurate relative paths
            if prefix and not prefix.endswith("/"):
                prefix = prefix + "/"

            blobs = client.list_blobs(bucket_name, prefix=prefix)
            found = False
            for blob in blobs:
                # Compute relative path under the prefix
                rel_path = blob.name[len(prefix):] if prefix else blob.name
                if rel_path == "":
                    # skip placeholder
                    continue
                found = True
                local_path = tmpdir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Downloading: {blob.name} -> {local_path}")
                blob.download_to_filename(str(local_path))

            if not found:
                raise FileNotFoundError(f"No objects found at {use_local_str}")

            # If we downloaded a GCS prefix, the blobs may have been written
            # under a subdirectory matching that prefix (e.g., tmpdir/20/...).
            # Set `base` to that subdirectory when available so resolve_paths()
            # finds the expected files (cards CSV and battle files).
            prefix_clean = prefix.rstrip("/") if prefix else ""
            if prefix_clean:
                candidate = tmpdir / prefix_clean
                if candidate.exists():
                    base = candidate
                else:
                    base = tmpdir
            else:
                base = tmpdir

            paths = resolve_paths(base, quick=quick)

        else:
            base = Path(use_local).expanduser().resolve()
            print(f"Using local dataset directory: {base}")
            paths = resolve_paths(base, quick=quick)
    else:
        base = dataset_download(handle, force_download=force_download, cache_dir=cache_dir)
        paths = resolve_paths(base, quick=quick)

    # Friendly hint if files are missing / the structure changed (skip when direct GCS)
    try:
        gs_direct_check = isinstance(paths.cards_csv, str) and str(paths.cards_csv).startswith("gs://")
    except Exception:
        gs_direct_check = False

    if not gs_direct_check and not paths.exists():
        print("\n[Hint] CSVs not found at expected locations. Here's a quick directory peek:")
        for p in list_csvs(base, limit=50):
            print("  -", p.relative_to(base))
        raise FileNotFoundError(
            "Could not locate expected CSVs. If the dataset layout changed, call "
            "resolve_paths(...) with the new filenames, or pass use_local to the "
            "appropriate directory."
        )

    battles_df, cards = load_raw(paths, low_memory=low_memory, sample=sample, random_state=random_state)

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
