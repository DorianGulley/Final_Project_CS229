# features.py
# --------------------------------------------------------------------------------------
# Purpose
#   Transform raw Clash Royale S18 tables into sparse, model-ready features.
#   - Build stable feature maps (card → column, unordered pair → column)
#   - Construct anti-symmetric "A vs B" examples with deck and pairwise blocks
#   - Produce train/val/test splits (time-respecting by index) and standardize Δtrophies
#   - Assemble final CSR matrices with selectable feature blocks
#
# Public API
#   - build_feature_maps(cards_df)
#   - make_splits(daydf, train_frac=0.70, val_frac=0.15)
#   - build_split(split_df, maps, win_cols, los_cols)
#   - standardize_delta(train_delta, val_delta, test_delta)
#   - assemble_blocks(blocks, use=("deck","ab","delta"))
#   - build_all_features(daydf, cards_df, ...)
# --------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMaps:
    """Mappings that define the column layout of the feature space.

    Attributes
    ----------
    card_to_col : Dict[int, int]
        Maps card ID → column index in the deck block [0..D-1].
    pair_to_col : Dict[Tuple[int, int], int]
        Maps unordered pair of *deck-column indices* (i<j) → column index in the
        pairwise block [0..P-1].
    card_ids : np.ndarray
        Array of the card IDs in the canonical order (len D).
    D : int
        Number of distinct cards (size of deck block).
    P : int
        Number of unordered pairs D*(D-1)/2 (size of pairwise block).
    """

    card_to_col: Dict[int, int]
    pair_to_col: Dict[Tuple[int, int], int]
    card_ids: np.ndarray
    D: int
    P: int


@dataclass(frozen=True)
class SplitBlocks:
    """Sparse feature blocks + labels for a single data split.

    Attributes
    ----------
    X_deck : sp.csr_matrix
        +1/-1 deck presence features (A minus B) per example; shape (n_rows, D).
    X_ab : sp.csr_matrix
        Anti-symmetric pairwise counter features; shape (n_rows, P).
    delta : np.ndarray
        Raw Δtrophies (A - B), shape (n_rows, 1), before standardization.
    y : np.ndarray
        Binary labels (1 for A wins, 0 otherwise); shape (n_rows,).
    """

    X_deck: sp.csr_matrix
    X_ab: sp.csr_matrix
    delta: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class DeltaStats:
    """Mean/std used to standardize Δtrophies based on TRAIN only."""
    mu: float
    sd: float


@dataclass(frozen=True)
class AssembledDataset:
    """Final assembled matrices and labels for all splits + bookkeeping."""
    X_train: sp.csr_matrix
    X_val: sp.csr_matrix
    X_test: sp.csr_matrix
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    blocks_train: SplitBlocks
    blocks_val: SplitBlocks
    blocks_test: SplitBlocks
    stats: DeltaStats
    maps: FeatureMaps


# --------------------------------------------------------------------------------------
# Constants & column helpers
# --------------------------------------------------------------------------------------

NUM_DECK_CARDS = 8


def make_win_los_columns(n: int = NUM_DECK_CARDS) -> Tuple[List[str], List[str]]:
    """Return column names for winner/loser card IDs.

    Parameters
    ----------
    n : int
        Number of card slots per deck (default 8).

    Returns
    -------
    (win_cols, los_cols)
        Lists like ["winner.card1.id", ..., "winner.card8.id"].
    """
    win_cols = [f"winner.card{i}.id" for i in range(1, n + 1)]
    los_cols = [f"loser.card{i}.id" for i in range(1, n + 1)]
    return win_cols, los_cols


# --------------------------------------------------------------------------------------
# Feature map construction
# --------------------------------------------------------------------------------------

def build_feature_maps(cards_df: pd.DataFrame, card_id_col: str = "team.card1.id") -> FeatureMaps:
    """Create stable feature maps from the card master list.

    The card block order follows the order of `card_id_col` in `cards_df` to keep
    reproducibility and alignment with prior experiments.
    """
    card_ids = cards_df[card_id_col].astype(int).to_numpy()
    card_to_col = {int(cid): i for i, cid in enumerate(card_ids)}

    D = len(card_ids)
    pair_to_col: Dict[Tuple[int, int], int] = {}
    col = 0
    for i in range(D):
        for j in range(i + 1, D):
            pair_to_col[(i, j)] = col
            col += 1
    P = col

    return FeatureMaps(card_to_col=card_to_col, pair_to_col=pair_to_col, card_ids=card_ids, D=D, P=P)


# --------------------------------------------------------------------------------------
# Split construction (time-respecting by index)
# --------------------------------------------------------------------------------------

def make_splits(daydf: pd.DataFrame, *, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by index into train/val/test without shuffling.

    Assumes `daydf` is time-sorted upstream (dataset source is ordered). This keeps
    evaluation forward-looking and prevents leakage.
    """
    n = len(daydf)
    i1 = int(train_frac * n)
    i2 = int((train_frac + val_frac) * n)
    train_df = daydf.iloc[:i1]
    val_df = daydf.iloc[i1:i2]
    test_df = daydf.iloc[i2:]
    return train_df, val_df, test_df


# --------------------------------------------------------------------------------------
# Core feature construction per split
# --------------------------------------------------------------------------------------

# Default chunk size for memory-efficient processing
DEFAULT_CHUNK_SIZE = 500_000  # Process 500K battles at a time


def _build_chunk(
    W: np.ndarray,
    L: np.ndarray,
    tW: np.ndarray,
    tL: np.ndarray,
    maps: FeatureMaps,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build sparse matrices for a single chunk of data.
    
    Returns (X_deck, X_ab, delta, y) for this chunk.
    """
    D, P = maps.D, maps.P
    n_battles = len(W)
    n_rows = n_battles * 2  # Anti-symmetric: 2 rows per battle

    # Sparse scaffolding
    deck_data: List[int] = []
    deck_rows: List[int] = []
    deck_cols: List[int] = []

    xab_data: List[int] = []
    xab_rows: List[int] = []
    xab_cols: List[int] = []

    deltas: List[float] = []
    labels: List[int] = []

    r = 0

    def add_example(r_idx: int, A_ids: Iterable[int], B_ids: Iterable[int], delta: float, y: int) -> None:
        """Emit one anti-symmetric example row (A vs B)."""
        A = sorted({maps.card_to_col[int(cid)] for cid in A_ids if int(cid) in maps.card_to_col})
        B = sorted({maps.card_to_col[int(cid)] for cid in B_ids if int(cid) in maps.card_to_col})

        for c in A:
            deck_rows.append(r_idx); deck_cols.append(c); deck_data.append(+1)
        for c in B:
            deck_rows.append(r_idx); deck_cols.append(c); deck_data.append(-1)

        for i in A:
            for j in B:
                if i == j:
                    continue
                if i < j:
                    col = maps.pair_to_col[(i, j)]
                    xab_rows.append(r_idx); xab_cols.append(col); xab_data.append(+1)
                else:
                    col = maps.pair_to_col[(j, i)]
                    xab_rows.append(r_idx); xab_cols.append(col); xab_data.append(-1)

        deltas.append(float(delta))
        labels.append(int(y))

    # Emit two rows per battle
    for j in range(n_battles):
        add_example(r, W[j], L[j], tW[j] - tL[j], 1); r += 1
        add_example(r, L[j], W[j], tL[j] - tW[j], 0); r += 1

    # Build CSR matrices for this chunk
    X_deck = sp.csr_matrix(
        (np.asarray(deck_data, dtype=np.int8),
         (np.asarray(deck_rows, dtype=np.int64), np.asarray(deck_cols, dtype=np.int64))),
        shape=(r, D),
    )
    X_ab = sp.csr_matrix(
        (np.asarray(xab_data, dtype=np.int8),
         (np.asarray(xab_rows, dtype=np.int64), np.asarray(xab_cols, dtype=np.int64))),
        shape=(r, P),
    )
    delta = np.asarray(deltas, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int8)

    return X_deck, X_ab, delta, y


def _build_split_from_arrays(
    W_all: np.ndarray,
    L_all: np.ndarray,
    tW_all: np.ndarray,
    tL_all: np.ndarray,
    maps: FeatureMaps,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SplitBlocks:
    """Build sparse blocks from pre-extracted numpy arrays.
    
    Uses chunked processing to reduce peak memory usage.
    """
    n_total = len(W_all)

    # Process in chunks to limit memory usage
    deck_chunks: List[sp.csr_matrix] = []
    ab_chunks: List[sp.csr_matrix] = []
    delta_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    
    for i in range(0, n_total, chunk_size):
        end = min(i + chunk_size, n_total)
        chunk_num = i // chunk_size + 1
        
        if n_chunks > 1:
            print(f"    chunk {chunk_num}/{n_chunks} ({i:,}-{end:,})", flush=True)
        
        X_deck, X_ab, delta, y = _build_chunk(
            W_all[i:end], L_all[i:end], tW_all[i:end], tL_all[i:end], maps
        )
        
        deck_chunks.append(X_deck)
        ab_chunks.append(X_ab)
        delta_chunks.append(delta)
        y_chunks.append(y)

    # Stack all chunks vertically
    if len(deck_chunks) == 1:
        X_deck_final = deck_chunks[0]
        X_ab_final = ab_chunks[0]
        delta_final = delta_chunks[0]
        y_final = y_chunks[0]
    else:
        X_deck_final = sp.vstack(deck_chunks, format="csr")
        X_ab_final = sp.vstack(ab_chunks, format="csr")
        delta_final = np.vstack(delta_chunks)
        y_final = np.concatenate(y_chunks)

    return SplitBlocks(X_deck=X_deck_final, X_ab=X_ab_final, delta=delta_final, y=y_final)


def build_split(
    split_df: pd.DataFrame,
    maps: FeatureMaps,
    win_cols: Sequence[str],
    los_cols: Sequence[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SplitBlocks:
    """Convert a slice of the battle log into sparse blocks and labels.

    For each original match, emits two rows:
      - Row A: winner deck vs loser deck → y=1, Δ = tW - tL
      - Row B: loser  deck vs winner deck → y=0, Δ = tL - tW

    Features are anti-symmetric so flipping A/B flips the sign, enabling models
    without an intercept to learn relative effects.
    
    Uses chunked processing to reduce peak memory usage.
    """
    # Pull arrays once
    W_all = split_df[list(win_cols)].to_numpy(dtype=np.int32)
    L_all = split_df[list(los_cols)].to_numpy(dtype=np.int32)
    tW_all = split_df["winner.startingTrophies"].to_numpy(dtype=np.float32)
    tL_all = split_df["loser.startingTrophies"].to_numpy(dtype=np.float32)

    return _build_split_from_arrays(W_all, L_all, tW_all, tL_all, maps, chunk_size)


# --------------------------------------------------------------------------------------
# Δtrophies standardization and block assembly
# --------------------------------------------------------------------------------------

def standardize_delta(train_delta: np.ndarray, val_delta: np.ndarray, test_delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DeltaStats]:
    """Standardize Δ using TRAIN mean/std; return standardized arrays + stats."""
    mu = float(train_delta.mean())
    sd = float(train_delta.std()) or 1.0
    dtr = (train_delta - mu) / sd
    dva = (val_delta - mu) / sd
    dte = (test_delta - mu) / sd
    return dtr, dva, dte, DeltaStats(mu=mu, sd=sd)


def assemble_blocks(blocks: SplitBlocks, d_std: Optional[np.ndarray] = None, use: Sequence[str] = ("deck", "ab", "delta")) -> sp.csr_matrix:
    """Horizontally stack selected blocks into a final CSR matrix."""
    mats: List[sp.csr_matrix] = []
    for key in use:
        if key == "deck":
            mats.append(blocks.X_deck)
        elif key == "ab":
            mats.append(blocks.X_ab)
        elif key == "delta":
            if d_std is None:
                raise ValueError("assemble_blocks: standardized Δ (d_std) must be provided when using 'delta'.")
            mats.append(sp.csr_matrix(d_std))
        else:
            raise ValueError(f"Unknown block '{key}'. Expected 'deck', 'ab', 'delta'.")
    return sp.hstack(mats, format="csr") if len(mats) > 1 else mats[0]


# --------------------------------------------------------------------------------------
# High-level pipeline
# --------------------------------------------------------------------------------------

def build_all_features(
    daydf: pd.DataFrame,
    cards_df: pd.DataFrame,
    *,
    card_id_col: str = "team.card1.id",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    use_blocks: Sequence[str] = ("deck", "ab", "delta"),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AssembledDataset:
    """End-to-end feature build: maps → splits → blocks → standardize → assemble."""
    import gc
    
    maps = build_feature_maps(cards_df, card_id_col=card_id_col)
    win_cols, los_cols = make_win_los_columns()

    # Extract numpy arrays for each split, then delete DataFrame to free memory
    print("\nExtracting data arrays...")
    train_df, val_df, test_df = make_splits(daydf, train_frac=train_frac, val_frac=val_frac)
    
    # Pre-extract all arrays before deleting DataFrame
    train_data = (
        train_df[list(win_cols)].to_numpy(dtype=np.int32),
        train_df[list(los_cols)].to_numpy(dtype=np.int32),
        train_df["winner.startingTrophies"].to_numpy(dtype=np.float32),
        train_df["loser.startingTrophies"].to_numpy(dtype=np.float32),
    )
    val_data = (
        val_df[list(win_cols)].to_numpy(dtype=np.int32),
        val_df[list(los_cols)].to_numpy(dtype=np.int32),
        val_df["winner.startingTrophies"].to_numpy(dtype=np.float32),
        val_df["loser.startingTrophies"].to_numpy(dtype=np.float32),
    )
    test_data = (
        test_df[list(win_cols)].to_numpy(dtype=np.int32),
        test_df[list(los_cols)].to_numpy(dtype=np.int32),
        test_df["winner.startingTrophies"].to_numpy(dtype=np.float32),
        test_df["loser.startingTrophies"].to_numpy(dtype=np.float32),
    )
    
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    
    # Free the DataFrame memory
    del daydf, train_df, val_df, test_df
    gc.collect()
    print("  DataFrame freed, building sparse features...")

    print(f"\nBuilding features (chunk_size={chunk_size:,})...")
    print(f"  Train split ({n_train:,} battles)...")
    blocks_train = _build_split_from_arrays(*train_data, maps, chunk_size=chunk_size)
    del train_data; gc.collect()
    
    print(f"  Val split ({n_val:,} battles)...")
    blocks_val = _build_split_from_arrays(*val_data, maps, chunk_size=chunk_size)
    del val_data; gc.collect()
    
    print(f"  Test split ({n_test:,} battles)...")
    blocks_test = _build_split_from_arrays(*test_data, maps, chunk_size=chunk_size)
    del test_data; gc.collect()

    dtr_std, dva_std, dte_std, stats = standardize_delta(blocks_train.delta, blocks_val.delta, blocks_test.delta)

    X_train = assemble_blocks(blocks_train, d_std=dtr_std, use=use_blocks)
    X_val = assemble_blocks(blocks_val, d_std=dva_std, use=use_blocks)
    X_test = assemble_blocks(blocks_test, d_std=dte_std, use=use_blocks)

    y_train = blocks_train.y.astype(np.int32)
    y_val = blocks_val.y.astype(np.int32)
    y_test = blocks_test.y.astype(np.int32)

    return AssembledDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        blocks_train=blocks_train,
        blocks_val=blocks_val,
        blocks_test=blocks_test,
        stats=stats,
        maps=maps,
    )


# --------------------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    print("features.py ready. Import and call build_all_features(daydf, cards_df).")
