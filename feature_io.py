# feature_io.py
# Purpose: Save and load precomputed feature artifacts (local or gs://) for fast reuse.

import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from features import AssembledDataset, SplitBlocks, FeatureMaps, DeltaStats, LevelStats


def _is_gcs_uri(path: str) -> bool:
    """Check if a path is a gs:// URI."""
    return str(path).startswith("gs://")


def _download_gcs_blob_to_local(gcs_uri: str) -> str:
    """Download a blob from GCS to a local temp file and return the local path."""
    from google.cloud import storage
    from urllib.parse import urlparse
    
    p = urlparse(gcs_uri)
    bucket_name = p.netloc
    blob_path = p.path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Create a temp file and download
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    
    blob.download_to_filename(tmp_path)
    return tmp_path


def _upload_local_file_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to a GCS blob."""
    from google.cloud import storage
    from urllib.parse import urlparse
    
    p = urlparse(gcs_uri)
    bucket_name = p.netloc
    blob_path = p.path.lstrip("/")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def _gcs_list_blobs(gcs_dir: str) -> list:
    """List all blobs under a GCS directory (prefix)."""
    from google.cloud import storage
    from urllib.parse import urlparse
    
    p = urlparse(gcs_dir)
    bucket_name = p.netloc
    prefix = p.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return blobs


def _ensure_local_path(path: str) -> Path:
    """Ensure a path is local and return as Path. If GCS, raise error for setup."""
    if _is_gcs_uri(path):
        raise ValueError(f"Path {path} is a GCS URI. Convert or download it first for local setup.")
    return Path(path)


def save_features(ds: AssembledDataset, base_path: str) -> None:
    """Save AssembledDataset to a directory (local or GCS).

    Saves:
      - X_train.npz, X_val.npz, X_test.npz (sparse CSR matrices)
      - y_train.npy, y_val.npy, y_test.npy (label arrays)
      - metadata.json (feature maps, stats, config)

    Parameters
    ----------
    ds : AssembledDataset
        The assembled dataset to save.
    base_path : str
        Local path or gs:// URI where to save artifacts.
    """
    base_path_str = str(base_path)
    is_gcs = _is_gcs_uri(base_path_str)

    if is_gcs:
        # Save to local temp directory, then upload each file to GCS
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _save_features_local(ds, tmp_path)

            # Upload all files to GCS
            print(f"Uploading feature artifacts to {base_path_str}...")
            for fpath in tmp_path.glob("*"):
                gcs_uri = f"{base_path_str.rstrip('/')}/{fpath.name}"
                print(f"  uploading {fpath.name} → {gcs_uri}")
                _upload_local_file_to_gcs(str(fpath), gcs_uri)
    else:
        # Save directly to local directory
        base_path_obj = Path(base_path_str)
        base_path_obj.mkdir(parents=True, exist_ok=True)
        _save_features_local(ds, base_path_obj)

    print(f"Feature artifacts saved to {base_path_str}")


def _save_features_local(ds: AssembledDataset, base_path: Path) -> None:
    """Save AssembledDataset to a local directory."""
    base_path.mkdir(parents=True, exist_ok=True)

    # Save sparse matrices
    sp.save_npz(str(base_path / "X_train.npz"), ds.X_train)
    sp.save_npz(str(base_path / "X_val.npz"), ds.X_val)
    sp.save_npz(str(base_path / "X_test.npz"), ds.X_test)

    # Save label arrays
    np.save(str(base_path / "y_train.npy"), ds.y_train)
    np.save(str(base_path / "y_val.npy"), ds.y_val)
    np.save(str(base_path / "y_test.npy"), ds.y_test)

    # Save metadata (feature maps, stats)
    metadata = {
        "card_to_col": {int(k): int(v) for k, v in ds.maps.card_to_col.items()},
        "pair_to_col": {str(k): int(v) for k, v in ds.maps.pair_to_col.items()},
        "card_ids": ds.maps.card_ids.tolist(),
        "D": int(ds.maps.D),
        "P": int(ds.maps.P),
        "delta_stats": {
            "mu": float(ds.stats.mu),
            "sd": float(ds.stats.sd),
        },
        "level_stats": (
            {
                "mu": float(ds.level_stats.mu),
                "sd": float(ds.level_stats.sd),
            }
            if ds.level_stats is not None
            else None
        ),
        "shapes": {
            "X_train": ds.X_train.shape,
            "X_val": ds.X_val.shape,
            "X_test": ds.X_test.shape,
            "y_train": ds.y_train.shape,
            "y_val": ds.y_val.shape,
            "y_test": ds.y_test.shape,
        },
    }

    with open(str(base_path / "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  saved X_train.npz, X_val.npz, X_test.npz")
    print(f"  saved y_train.npy, y_val.npy, y_test.npy")
    print(f"  saved metadata.json")


def load_features(base_path: str) -> AssembledDataset:
    """Load AssembledDataset from a directory (local or GCS).

    Loads:
      - X_train.npz, X_val.npz, X_test.npz (sparse CSR matrices)
      - y_train.npy, y_val.npy, y_test.npy (label arrays)
      - metadata.json (feature maps, stats, config)

    Parameters
    ----------
    base_path : str
        Local path or gs:// URI where artifacts are stored.

    Returns
    -------
    AssembledDataset
        Reconstructed dataset with all matrices, labels, and metadata.
    """
    base_path_str = str(base_path)
    is_gcs = _is_gcs_uri(base_path_str)

    if is_gcs:
        # Download all files from GCS to local temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            print(f"Downloading feature artifacts from {base_path_str}...")
            _download_gcs_artifacts(base_path_str, tmp_path)
            return _load_features_local(tmp_path)
    else:
        # Load directly from local directory
        return _load_features_local(Path(base_path_str))


def _download_gcs_artifacts(gcs_dir: str, local_path: Path) -> None:
    """Download all artifacts from a GCS directory to a local directory."""
    blobs = _gcs_list_blobs(gcs_dir)
    
    for blob in blobs:
        if blob.name.endswith("/"):
            # Skip directory markers
            continue
        
        file_name = blob.name.split("/")[-1]
        local_file = local_path / file_name
        
        print(f"  downloading {file_name}")
        blob.download_to_filename(str(local_file))


def _load_features_local(base_path: Path) -> AssembledDataset:
    """Load AssembledDataset from a local directory."""
    base_path = Path(base_path)

    # Load sparse matrices
    X_train = sp.load_npz(str(base_path / "X_train.npz"))
    X_val = sp.load_npz(str(base_path / "X_val.npz"))
    X_test = sp.load_npz(str(base_path / "X_test.npz"))

    # Load label arrays
    y_train = np.load(str(base_path / "y_train.npy"))
    y_val = np.load(str(base_path / "y_val.npy"))
    y_test = np.load(str(base_path / "y_test.npy"))

    # Load metadata
    with open(str(base_path / "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Reconstruct FeatureMaps
    card_to_col = {int(k): int(v) for k, v in metadata["card_to_col"].items()}
    pair_to_col = {}
    for k, v in metadata["pair_to_col"].items():
        # Key is stringified tuple; parse it back
        if k.startswith("(") and k.endswith(")"):
            # Parse tuple string like "(0, 1)"
            inner = k[1:-1].split(",")
            i, j = int(inner[0].strip()), int(inner[1].strip())
            pair_to_col[(i, j)] = int(v)
        else:
            # Fallback for other formats
            pair_to_col[eval(k)] = int(v)

    card_ids = np.array(metadata["card_ids"], dtype=np.int32)
    D = int(metadata["D"])
    P = int(metadata["P"])

    maps = FeatureMaps(
        card_to_col=card_to_col,
        pair_to_col=pair_to_col,
        card_ids=card_ids,
        D=D,
        P=P,
    )

    # Reconstruct DeltaStats and LevelStats
    delta_stats = DeltaStats(
        mu=float(metadata["delta_stats"]["mu"]),
        sd=float(metadata["delta_stats"]["sd"]),
    )

    level_stats = None
    if metadata["level_stats"] is not None:
        level_stats = LevelStats(
            mu=float(metadata["level_stats"]["mu"]),
            sd=float(metadata["level_stats"]["sd"]),
        )

    # Reconstruct SplitBlocks (without the raw arrays which are not saved)
    # Note: we don't save the raw delta/levels arrays from SplitBlocks, only the metadata
    blocks_train = SplitBlocks(X_deck=None, X_ab=None, delta=None, levels=None, y=y_train)
    blocks_val = SplitBlocks(X_deck=None, X_ab=None, delta=None, levels=None, y=y_val)
    blocks_test = SplitBlocks(X_deck=None, X_ab=None, delta=None, levels=None, y=y_test)

    # Reconstruct AssembledDataset
    ds = AssembledDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        blocks_train=blocks_train,
        blocks_val=blocks_val,
        blocks_test=blocks_test,
        stats=delta_stats,
        level_stats=level_stats,
        maps=maps,
    )

    print(f"Feature artifacts loaded from {base_path}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return ds


if __name__ == "__main__":
    print("feature_io.py ready. Use save_features() and load_features().")
