# oracle_eval.py
# Evaluate the trained model as a matchup oracle
# Compares predicted win probabilities vs empirical win rates for deck matchups
#
# Usage:
#   python oracle_eval.py --quick                    # Quick mode (single day)
#   python oracle_eval.py --quick --min-games 20    # Require at least 20 head-to-head games
#   python oracle_eval.py --quick --top-k 10        # Analyze top 10 most common decks

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr

from data import get_raw_data
from features import build_feature_maps, FeatureMaps
from models import get_model


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class DeckMatchup:
    """Container for a single deck matchup's predicted and empirical stats."""
    opponent_deck: Tuple[int, ...]
    opponent_names: str
    games: int
    wins: int
    losses: int
    empirical_wr: float
    predicted_wr: float


# --------------------------------------------------------------------------------------
# Feature building for a single matchup
# --------------------------------------------------------------------------------------

def build_matchup_features(
    deck_a: Tuple[int, ...],
    deck_b: Tuple[int, ...],
    maps: FeatureMaps,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> sp.csr_matrix:
    """Build the feature vector for deck_a vs deck_b matchup.
    
    Returns a sparse row vector suitable for model.predict_proba().
    """
    D, P = maps.D, maps.P
    
    # Map card IDs to column indices
    A_cols = sorted({maps.card_to_col[int(cid)] for cid in deck_a if int(cid) in maps.card_to_col})
    B_cols = sorted({maps.card_to_col[int(cid)] for cid in deck_b if int(cid) in maps.card_to_col})
    
    # Build sparse data
    deck_data, deck_cols = [], []
    xab_data, xab_cols = [], []
    
    # Deck block: +1 for A, -1 for B
    for c in A_cols:
        deck_cols.append(c)
        deck_data.append(+1)
    for c in B_cols:
        deck_cols.append(c)
        deck_data.append(-1)
    
    # Pairwise block
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
    
    # Assemble blocks
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
            # Delta = 0 (assume equal trophies for oracle comparison)
            X_delta = sp.csr_matrix((1, 1), dtype=np.float32)
            parts.append(X_delta)
    
    return sp.hstack(parts, format="csr")


def batch_predict_matchups(
    deck_a: Tuple[int, ...],
    opponents: List[Tuple[int, ...]],
    maps: FeatureMaps,
    model,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> np.ndarray:
    """Predict win probabilities for deck_a vs multiple opponents in batch."""
    # Build all feature vectors
    X_list = [build_matchup_features(deck_a, opp, maps, use_blocks) for opp in opponents]
    X = sp.vstack(X_list, format="csr")
    
    # Predict
    probs = model.predict_proba(X)[:, 1]
    return probs


# --------------------------------------------------------------------------------------
# Main oracle evaluation
# --------------------------------------------------------------------------------------

def deck_to_tuple(row, cols) -> Tuple[int, ...]:
    """Convert a row's card columns to a sorted tuple."""
    return tuple(sorted(int(row[c]) for c in cols))


def evaluate_oracle_for_target(
    target_deck: Tuple[int, ...],
    winner_decks: List[Tuple[int, ...]],
    loser_decks: List[Tuple[int, ...]],
    maps: FeatureMaps,
    model,
    id_to_name: Dict[int, str],
    min_games: int = 20,
    use_blocks: List[str] = ["deck", "ab", "delta"],
) -> List[DeckMatchup]:
    """Evaluate oracle predictions vs empirical win rates for a target deck."""
    
    # Find all battles involving target
    target_as_winner = []  # opponent decks when target won
    target_as_loser = []   # opponent decks when target lost
    
    for w_deck, l_deck in zip(winner_decks, loser_decks):
        if w_deck == target_deck:
            target_as_loser.append(l_deck)  # Target beat this deck (opponent lost)
        if l_deck == target_deck:
            target_as_winner.append(w_deck)  # Target lost to this deck (opponent won)
    
    # Count games and wins for each opponent
    # Note: we're computing win rate of OPPONENT vs TARGET
    # So wins = times opponent beat target, losses = times target beat opponent
    opponent_wins = Counter(target_as_winner)    # Opponent beat target
    opponent_losses = Counter(target_as_loser)   # Target beat opponent (opponent lost)
    all_opponents = set(opponent_wins.keys()) | set(opponent_losses.keys())
    
    # Filter to opponents with enough games
    frequent_opponents = []
    for opp in all_opponents:
        wins = opponent_wins.get(opp, 0)
        losses = opponent_losses.get(opp, 0)
        total = wins + losses
        if total >= min_games:
            frequent_opponents.append(opp)
    
    if not frequent_opponents:
        return []
    
    # Get model predictions (probability that opponent beats target)
    predicted_wrs = batch_predict_matchups(
        frequent_opponents[0], [target_deck], maps, model, use_blocks
    )
    
    # Actually we need to predict for each opponent vs target
    # p(opponent beats target) = model.predict_proba(opponent vs target)
    all_predictions = []
    for opp in frequent_opponents:
        X = build_matchup_features(opp, target_deck, maps, use_blocks)
        pred = model.predict_proba(X)[0, 1]
        all_predictions.append(pred)
    
    # Build results
    results = []
    for opp, pred in zip(frequent_opponents, all_predictions):
        wins = opponent_wins.get(opp, 0)
        losses = opponent_losses.get(opp, 0)
        total = wins + losses
        empirical_wr = wins / total
        
        opp_names = ", ".join(id_to_name.get(cid, f"?{cid}") for cid in opp)
        
        results.append(DeckMatchup(
            opponent_deck=opp,
            opponent_names=opp_names,
            games=total,
            wins=wins,
            losses=losses,
            empirical_wr=empirical_wr,
            predicted_wr=pred,
        ))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model as matchup oracle")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (single day)")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction")
    parser.add_argument("--min-games", type=int, default=20, help="Min head-to-head games required")
    parser.add_argument("--top-k", type=int, default=5, help="Analyze top-k most common decks")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg regularization")
    parser.add_argument("--max-iter", type=int, default=200, help="LogReg max iterations")
    args = parser.parse_args()

    # --- 1) Load data ---
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    daydf, cards, paths = get_raw_data(quick=args.quick, sample=args.sample)
    
    win_cols = [f"winner.card{i}.id" for i in range(1, 9)]
    los_cols = [f"loser.card{i}.id" for i in range(1, 9)]
    
    # Build card name mapping
    id_to_name = dict(zip(cards["team.card1.id"].astype(int), cards["team.card1.name"]))
    
    # Extract decks
    print("\nExtracting decks...")
    winner_decks = [deck_to_tuple(row, win_cols) for _, row in daydf.iterrows()]
    loser_decks = [deck_to_tuple(row, los_cols) for _, row in daydf.iterrows()]
    
    # Count deck frequencies
    all_decks = winner_decks + loser_decks
    deck_counts = Counter(all_decks)
    top_decks = deck_counts.most_common(args.top_k)
    
    print(f"Total battles: {len(daydf):,}")
    print(f"Unique decks: {len(deck_counts):,}")
    
    # --- 2) Build feature maps ---
    print("\n" + "=" * 70)
    print("BUILDING FEATURE MAPS")
    print("=" * 70)
    maps = build_feature_maps(cards)
    print(f"Cards (D): {maps.D}")
    print(f"Pairs (P): {maps.P}")
    
    # --- 3) Train model on full data ---
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # We need to train the model - import and use the feature building from features.py
    from features import build_all_features
    
    use_blocks = ["deck", "ab", "delta"]
    ds = build_all_features(daydf, cards, use_blocks=use_blocks)
    
    model = get_model("logreg", 
                      C=args.C, 
                      max_iter=args.max_iter, 
                      verbose=1,
                      fit_intercept=False,
                      penalty="l2")
    model.fit(ds.X_train, ds.y_train)
    
    # Quick validation check
    from metrics import evaluate
    p_val = model.predict_proba(ds.X_val)[:, 1]
    val_metrics = evaluate(ds.y_val, p_val)
    print(f"\nValidation AUC: {val_metrics['auc']:.4f}")
    
    # --- 4) Evaluate oracle for top decks ---
    print("\n" + "=" * 70)
    print(f"ORACLE EVALUATION (top {args.top_k} decks, min {args.min_games} games)")
    print("=" * 70)
    
    all_correlations = []
    
    for rank, (target_deck, count) in enumerate(top_decks, 1):
        target_names = ", ".join(id_to_name.get(cid, f"?{cid}") for cid in target_deck)
        print(f"\n{'─' * 70}")
        print(f"TARGET DECK #{rank}: {target_names[:60]}...")
        print(f"Appearances: {count:,}")
        print(f"{'─' * 70}")
        
        results = evaluate_oracle_for_target(
            target_deck, winner_decks, loser_decks,
            maps, model, id_to_name,
            min_games=args.min_games,
            use_blocks=use_blocks,
        )
        
        if len(results) < 3:
            print(f"  ⚠ Only {len(results)} opponents with ≥{args.min_games} games, skipping...")
            continue
        
        # Sort by predicted win rate (best counters first)
        results.sort(key=lambda x: -x.predicted_wr)
        
        # Compute correlation between predicted and empirical
        predicted = [r.predicted_wr for r in results]
        empirical = [r.empirical_wr for r in results]
        
        spearman_corr, spearman_p = spearmanr(predicted, empirical)
        pearson_corr, pearson_p = pearsonr(predicted, empirical)
        all_correlations.append((rank, target_names[:30], len(results), spearman_corr, pearson_corr))
        
        print(f"\nOpponents with ≥{args.min_games} games: {len(results)}")
        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
        print(f"Pearson correlation:  {pearson_corr:.3f} (p={pearson_p:.4f})")
        
        # Show top 5 predicted counters
        print(f"\nTOP 5 PREDICTED COUNTERS (by model):")
        print(f"{'Rank':<5} {'Pred':>6} {'Emp':>6} {'Games':>6}  {'Opponent'}")
        print("-" * 70)
        for i, r in enumerate(results[:5], 1):
            short_name = r.opponent_names[:45] + "..." if len(r.opponent_names) > 45 else r.opponent_names
            print(f"{i:<5} {r.predicted_wr:>5.1%} {r.empirical_wr:>5.1%} {r.games:>6}  {short_name}")
        
        # Show top 5 empirical counters
        results_by_emp = sorted(results, key=lambda x: -x.empirical_wr)
        print(f"\nTOP 5 EMPIRICAL COUNTERS (by actual win rate):")
        print(f"{'Rank':<5} {'Pred':>6} {'Emp':>6} {'Games':>6}  {'Opponent'}")
        print("-" * 70)
        for i, r in enumerate(results_by_emp[:5], 1):
            short_name = r.opponent_names[:45] + "..." if len(r.opponent_names) > 45 else r.opponent_names
            print(f"{i:<5} {r.predicted_wr:>5.1%} {r.empirical_wr:>5.1%} {r.games:>6}  {short_name}")
    
    # --- 5) Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Correlation Between Predicted and Empirical Win Rates")
    print("=" * 70)
    print(f"{'Deck':<35} {'N':>5} {'Spearman':>10} {'Pearson':>10}")
    print("-" * 70)
    for rank, name, n, spear, pear in all_correlations:
        print(f"{name:<35} {n:>5} {spear:>+10.3f} {pear:>+10.3f}")
    
    if all_correlations:
        avg_spearman = np.mean([c[3] for c in all_correlations])
        avg_pearson = np.mean([c[4] for c in all_correlations])
        print("-" * 70)
        print(f"{'AVERAGE':<35} {'':>5} {avg_spearman:>+10.3f} {avg_pearson:>+10.3f}")
    
    print("\n✓ Positive correlation = model predictions align with empirical win rates!")
    print("  (Higher correlation = better oracle)")


if __name__ == "__main__":
    main()

