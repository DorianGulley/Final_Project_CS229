# greedy_search.py
# Greedy local search to generate adversarial counter-decks
#
# Usage:
#   python greedy_search.py --quick --target-rank 4    # X-Bow is rank 4
#   python greedy_search.py --quick --lambda 0.6       # Different lambda

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from data import get_raw_data
from features import build_feature_maps, build_all_features, FeatureMaps
from models import get_model


# --------------------------------------------------------------------------------------
# Feature building for matchups (reused from oracle_eval)
# --------------------------------------------------------------------------------------

def build_matchup_features(
    deck_a: Tuple[int, ...],
    deck_b: Tuple[int, ...],
    maps: FeatureMaps,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> sp.csr_matrix:
    """Build the feature vector for deck_a vs deck_b matchup."""
    D, P = maps.D, maps.P
    
    A_cols = sorted({maps.card_to_col[int(cid)] for cid in deck_a if int(cid) in maps.card_to_col})
    B_cols = sorted({maps.card_to_col[int(cid)] for cid in deck_b if int(cid) in maps.card_to_col})
    
    deck_data, deck_cols = [], []
    xab_data, xab_cols = [], []
    
    for c in A_cols:
        deck_cols.append(c)
        deck_data.append(+1)
    for c in B_cols:
        deck_cols.append(c)
        deck_data.append(-1)
    
    for i in A_cols:
        for j in B_cols:
            if i == j:
                continue
            if i < j:
                col = maps.pair_to_col[(i, j)]
                xab_cols.append(col)
                xab_data.append(+1)
            else:
                col = maps.pair_to_col[(j, i)]
                xab_cols.append(col)
                xab_data.append(-1)
    
    parts = []
    for block in use_blocks:
        if block == "deck":
            X_deck = sp.csr_matrix(
                (np.array(deck_data, dtype=np.int8), 
                 (np.zeros(len(deck_data), dtype=np.int64), np.array(deck_cols, dtype=np.int64))),
                shape=(1, D)
            )
            parts.append(X_deck)
        elif block == "ab":
            X_ab = sp.csr_matrix(
                (np.array(xab_data, dtype=np.int8),
                 (np.zeros(len(xab_data), dtype=np.int64), np.array(xab_cols, dtype=np.int64))),
                shape=(1, P)
            )
            parts.append(X_ab)
        elif block == "delta":
            X_delta = sp.csr_matrix((1, 1), dtype=np.float32)
            parts.append(X_delta)
    
    return sp.hstack(parts, format="csr")


def predict_matchup(
    deck_a: Tuple[int, ...],
    deck_b: Tuple[int, ...],
    model,
    maps: FeatureMaps,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> float:
    """Predict probability that deck_a beats deck_b."""
    X = build_matchup_features(deck_a, deck_b, maps, use_blocks)
    return float(model.predict_proba(X)[0, 1])


def batch_predict_vs_target(
    decks: List[Tuple[int, ...]],
    target: Tuple[int, ...],
    model,
    maps: FeatureMaps,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> np.ndarray:
    """Batch predict win probabilities for multiple decks vs a single target."""
    X_list = [build_matchup_features(d, target, maps, use_blocks) for d in decks]
    X = sp.vstack(X_list, format="csr")
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------------------
# Scoring function: J(D) = λ·p(D,T) + (1-λ)·R(D)
# --------------------------------------------------------------------------------------

def compute_robustness(
    deck: Tuple[int, ...],
    meta_decks: List[Tuple[int, ...]],
    model,
    maps: FeatureMaps,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> float:
    """R(D) = average win probability against meta decks."""
    if not meta_decks:
        return 0.5
    probs = [predict_matchup(deck, m, model, maps, use_blocks) for m in meta_decks]
    return float(np.mean(probs))


def score_deck(
    deck: Tuple[int, ...],
    target: Tuple[int, ...],
    meta_decks: List[Tuple[int, ...]],
    model,
    maps: FeatureMaps,
    lambda_: float,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> Tuple[float, float, float]:
    """Compute J(D) = λ·p(D,T) + (1-λ)·R(D).
    
    Returns (total_score, p_vs_target, robustness).
    """
    p_vs_target = predict_matchup(deck, target, model, maps, use_blocks)
    robustness = compute_robustness(deck, meta_decks, model, maps, use_blocks)
    total = lambda_ * p_vs_target + (1 - lambda_) * robustness
    return total, p_vs_target, robustness


# --------------------------------------------------------------------------------------
# Neighbor generation
# --------------------------------------------------------------------------------------

def generate_neighbors(
    deck: Tuple[int, ...],
    all_card_ids: List[int],
) -> List[Tuple[int, ...]]:
    """Generate all single-card swap neighbors of a deck."""
    neighbors = []
    deck_set = set(deck)
    
    for i in range(len(deck)):  # 8 positions
        for new_card in all_card_ids:
            if new_card not in deck_set:  # No duplicates
                new_deck = list(deck)
                new_deck[i] = new_card
                neighbors.append(tuple(sorted(new_deck)))
    
    # Remove duplicates (different swaps might produce same deck)
    return list(set(neighbors))


# --------------------------------------------------------------------------------------
# Hill climbing
# --------------------------------------------------------------------------------------

@dataclass
class HillClimbResult:
    """Result of a hill climbing run."""
    start_deck: Tuple[int, ...]
    final_deck: Tuple[int, ...]
    start_score: float
    final_score: float
    final_p_vs_target: float
    final_robustness: float
    iterations: int
    path: List[Tuple[int, ...]]  # Sequence of decks


def hill_climb(
    start_deck: Tuple[int, ...],
    target: Tuple[int, ...],
    meta_decks: List[Tuple[int, ...]],
    model,
    maps: FeatureMaps,
    all_card_ids: List[int],
    lambda_: float,
    max_iter: int = 50,
    verbose: bool = True,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> HillClimbResult:
    """Run hill climbing to find a local optimum counter-deck."""
    
    current = start_deck
    current_score, current_p, current_r = score_deck(
        current, target, meta_decks, model, maps, lambda_, use_blocks
    )
    start_score = current_score
    
    path = [current]
    
    for iteration in range(max_iter):
        neighbors = generate_neighbors(current, all_card_ids)
        
        if verbose:
            print(f"  Iter {iteration + 1}: score={current_score:.4f} (p={current_p:.3f}, R={current_r:.3f}), {len(neighbors)} neighbors")
        
        # Find best neighbor
        best_neighbor = None
        best_score = current_score
        best_p, best_r = current_p, current_r
        
        for neighbor in neighbors:
            s, p, r = score_deck(neighbor, target, meta_decks, model, maps, lambda_, use_blocks)
            if s > best_score:
                best_score = s
                best_neighbor = neighbor
                best_p, best_r = p, r
        
        if best_neighbor is None:
            if verbose:
                print(f"  Converged at iteration {iteration + 1} (no improvement found)")
            break
        
        current = best_neighbor
        current_score = best_score
        current_p, current_r = best_p, best_r
        path.append(current)
    
    return HillClimbResult(
        start_deck=start_deck,
        final_deck=current,
        start_score=start_score,
        final_score=current_score,
        final_p_vs_target=current_p,
        final_robustness=current_r,
        iterations=len(path) - 1,
        path=path,
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def deck_to_tuple(row, cols) -> Tuple[int, ...]:
    return tuple(sorted(int(row[c]) for c in cols))


def deck_to_names(deck: Tuple[int, ...], id_to_name: Dict[int, str]) -> str:
    return ", ".join(id_to_name.get(cid, f"?{cid}") for cid in deck)


def main():
    parser = argparse.ArgumentParser(description="Greedy local search for counter-decks")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (single day)")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction")
    parser.add_argument("--target-rank", type=int, default=4, help="Rank of target deck (4=X-Bow)")
    parser.add_argument("--num-meta", type=int, default=10, help="Number of meta decks for R(D)")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0, help="Lambda for J(D)")
    parser.add_argument("--lambda2", type=float, default=0.6, help="Second lambda to compare")
    parser.add_argument("--max-iter", type=int, default=50, help="Max hill climbing iterations")
    parser.add_argument("--start-rank", type=int, default=1, help="Rank of starting deck (1=most popular)")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- 1) Load data ---
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    daydf, cards, paths = get_raw_data(quick=args.quick, sample=args.sample)
    
    win_cols = [f"winner.card{i}.id" for i in range(1, 9)]
    los_cols = [f"loser.card{i}.id" for i in range(1, 9)]
    
    id_to_name = dict(zip(cards["team.card1.id"].astype(int), cards["team.card1.name"]))
    all_card_ids = list(cards["team.card1.id"].astype(int))
    
    print("\nExtracting decks...")
    winner_decks = [deck_to_tuple(row, win_cols) for _, row in daydf.iterrows()]
    loser_decks = [deck_to_tuple(row, los_cols) for _, row in daydf.iterrows()]
    all_decks = winner_decks + loser_decks
    deck_counts = Counter(all_decks)
    top_decks = deck_counts.most_common(50)
    
    print(f"Total battles: {len(daydf):,}")
    print(f"Unique decks: {len(deck_counts):,}")
    
    # --- 2) Define target and meta decks ---
    target_deck = top_decks[args.target_rank - 1][0]
    target_name = deck_to_names(target_deck, id_to_name)
    print(f"\n{'=' * 70}")
    print(f"TARGET DECK (rank {args.target_rank}): {target_name}")
    print("=" * 70)
    
    # Meta decks = top N excluding target
    meta_decks = [d for d, _ in top_decks[:args.num_meta + 1] if d != target_deck][:args.num_meta]
    print(f"\nMeta decks for R(D) ({len(meta_decks)} decks):")
    for i, m in enumerate(meta_decks, 1):
        print(f"  {i}. {deck_to_names(m, id_to_name)[:60]}...")
    
    # Starting deck
    start_deck = top_decks[args.start_rank - 1][0]
    if start_deck == target_deck:
        start_deck = top_decks[args.start_rank][0]  # Skip if same as target
    start_name = deck_to_names(start_deck, id_to_name)
    print(f"\nStarting deck (rank {args.start_rank}): {start_name}")
    
    # --- 3) Build feature maps and train model ---
    print(f"\n{'=' * 70}")
    print("TRAINING MODEL")
    print("=" * 70)
    
    maps = build_feature_maps(cards)
    use_blocks = ["deck", "ab", "delta"]
    ds = build_all_features(daydf, cards, use_blocks=use_blocks)
    
    model = get_model("logreg", C=args.C, max_iter=200, verbose=1, fit_intercept=False, penalty="l2")
    model.fit(ds.X_train, ds.y_train)
    
    from metrics import evaluate
    p_val = model.predict_proba(ds.X_val)[:, 1]
    val_metrics = evaluate(ds.y_val, p_val)
    print(f"\nValidation AUC: {val_metrics['auc']:.4f}")
    
    # --- 4) Run hill climbing with both lambda values ---
    lambdas = [args.lambda_, args.lambda2]
    results = {}
    
    for lam in lambdas:
        print(f"\n{'=' * 70}")
        print(f"HILL CLIMBING (λ = {lam})")
        print("=" * 70)
        
        result = hill_climb(
            start_deck=start_deck,
            target=target_deck,
            meta_decks=meta_decks,
            model=model,
            maps=maps,
            all_card_ids=all_card_ids,
            lambda_=lam,
            max_iter=args.max_iter,
            verbose=True,
            use_blocks=use_blocks,
        )
        results[lam] = result
        
        print(f"\n--- RESULT (λ = {lam}) ---")
        print(f"Iterations: {result.iterations}")
        print(f"Score: {result.start_score:.4f} → {result.final_score:.4f}")
        print(f"P(beats target): {result.final_p_vs_target:.1%}")
        print(f"Robustness R(D): {result.final_robustness:.1%}")
        print(f"\nFinal deck:")
        print(f"  {deck_to_names(result.final_deck, id_to_name)}")
    
    # --- 5) Compare results ---
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\n{'λ':>6} {'Iters':>6} {'Score':>8} {'P(beat)':>8} {'R(D)':>8}")
    print("-" * 50)
    for lam in lambdas:
        r = results[lam]
        print(f"{lam:>6.1f} {r.iterations:>6} {r.final_score:>8.4f} {r.final_p_vs_target:>7.1%} {r.final_robustness:>7.1%}")
    
    # Show what cards changed
    print(f"\n{'=' * 70}")
    print("DECK EVOLUTION")
    print("=" * 70)
    
    for lam in lambdas:
        r = results[lam]
        start_set = set(r.start_deck)
        final_set = set(r.final_deck)
        
        removed = start_set - final_set
        added = final_set - start_set
        
        print(f"\nλ = {lam}:")
        print(f"  Removed: {', '.join(id_to_name.get(c, str(c)) for c in removed)}")
        print(f"  Added:   {', '.join(id_to_name.get(c, str(c)) for c in added)}")
    
    # --- 6) Check if final deck exists in data ---
    print(f"\n{'=' * 70}")
    print("VALIDATION: Do generated decks exist in data?")
    print("=" * 70)
    
    for lam in lambdas:
        r = results[lam]
        count = deck_counts.get(r.final_deck, 0)
        print(f"\nλ = {lam}: Final deck appears {count} times in data")
        
        if count == 0:
            # Find nearest neighbor
            best_match = None
            best_overlap = 0
            for d, c in top_decks[:100]:
                overlap = len(set(d) & set(r.final_deck))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = d
            
            if best_match:
                print(f"  Nearest match ({best_overlap}/8 cards): {deck_to_names(best_match, id_to_name)[:50]}...")
                match_count = deck_counts[best_match]
                print(f"  That deck appears {match_count} times")


if __name__ == "__main__":
    main()

