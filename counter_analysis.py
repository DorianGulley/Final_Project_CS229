# counter_analysis.py
# Analyze counter-cards using model coefficients, then find real decks
#
# Usage:
#   python counter_analysis.py --quick --target-rank 4   # X-Bow is rank 4
#   python counter_analysis.py --quick --top-cards 15    # Show top 15 counter-cards

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data import get_raw_data
from features import build_feature_maps, build_all_features, FeatureMaps
from models import get_model


def deck_to_tuple(row, cols) -> Tuple[int, ...]:
    return tuple(sorted(int(row[c]) for c in cols))


def deck_to_names(deck: Tuple[int, ...], id_to_name: Dict[int, str]) -> str:
    return ", ".join(id_to_name.get(cid, f"?{cid}") for cid in deck)


def compute_counter_scores(
    target_deck: Tuple[int, ...],
    maps: FeatureMaps,
    coefs: np.ndarray,
    id_to_name: Dict[int, str],
) -> List[Tuple[str, int, float]]:
    """Compute counter-score for each card against the target deck.
    
    For each potential counter-card C, the score is the sum of:
    - Deck coefficient for C (how good is C in general)
    - Pairwise coefficients M[C, T_i] for each card T_i in target deck
    
    Returns list of (card_name, card_id, score) sorted by score descending.
    """
    D = maps.D
    P = maps.P
    
    # Extract coefficient blocks
    deck_coefs = coefs[:D]  # Individual card effects
    pair_coefs = coefs[D:D+P]  # Pairwise effects
    
    # Get target deck column indices
    target_cols = [maps.card_to_col[int(cid)] for cid in target_deck if int(cid) in maps.card_to_col]
    
    # Reverse maps
    col_to_card_id = {v: k for k, v in maps.card_to_col.items()}
    col_to_pair = {v: k for k, v in maps.pair_to_col.items()}
    
    # For each card, compute counter-score
    scores = []
    for card_col in range(D):
        card_id = col_to_card_id[card_col]
        card_name = id_to_name.get(card_id, f"Card_{card_id}")
        
        # Skip if card is in target deck
        if card_col in target_cols:
            continue
        
        # Start with the deck coefficient (positive = good for player A)
        score = float(deck_coefs[card_col])
        
        # Add pairwise contributions against each target card
        for target_col in target_cols:
            # Get the pair coefficient
            if card_col < target_col:
                pair_key = (card_col, target_col)
                if pair_key in maps.pair_to_col:
                    pair_idx = maps.pair_to_col[pair_key]
                    # Positive coefficient means card_col counters target_col
                    score += float(pair_coefs[pair_idx])
            else:
                pair_key = (target_col, card_col)
                if pair_key in maps.pair_to_col:
                    pair_idx = maps.pair_to_col[pair_key]
                    # Negative because the stored coef is for (smaller, larger)
                    # If target < card, then M[target, card] being positive means target counters card
                    # So we want the negative: -M[target, card] = how much card counters target
                    score -= float(pair_coefs[pair_idx])
        
        scores.append((card_name, card_id, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: -x[2])
    return scores


def find_decks_with_cards(
    required_cards: List[int],
    deck_counts: Counter,
    min_matches: int = 2,
    min_appearances: int = 100,
) -> List[Tuple[Tuple[int, ...], int, int]]:
    """Find real decks containing at least min_matches of the required cards.
    
    Returns list of (deck, num_matches, appearances) sorted by num_matches desc.
    """
    results = []
    required_set = set(required_cards)
    
    for deck, count in deck_counts.items():
        if count < min_appearances:
            continue
        
        deck_set = set(deck)
        matches = len(deck_set & required_set)
        
        if matches >= min_matches:
            results.append((deck, matches, count))
    
    # Sort by matches (desc), then by appearances (desc)
    results.sort(key=lambda x: (-x[1], -x[2]))
    return results


def compute_empirical_winrate(
    deck: Tuple[int, ...],
    target_deck: Tuple[int, ...],
    winner_decks: List[Tuple[int, ...]],
    loser_decks: List[Tuple[int, ...]],
) -> Tuple[int, int, float]:
    """Compute empirical win rate of deck vs target_deck.
    
    Returns (wins, total_games, win_rate).
    """
    wins = 0
    losses = 0
    
    for w, l in zip(winner_decks, loser_decks):
        if w == deck and l == target_deck:
            wins += 1
        elif w == target_deck and l == deck:
            losses += 1
    
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    return wins, total, win_rate


def main():
    parser = argparse.ArgumentParser(description="Counter-card analysis using coefficients")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (single day)")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction")
    parser.add_argument("--target-rank", type=int, default=4, help="Rank of target deck (4=X-Bow)")
    parser.add_argument("--top-cards", type=int, default=20, help="Number of top counter-cards to show")
    parser.add_argument("--min-matches", type=int, default=3, help="Min counter-cards in deck to consider")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg regularization")
    args = parser.parse_args()

    # --- 1) Load data ---
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    daydf, cards, paths = get_raw_data(quick=args.quick, sample=args.sample)
    
    win_cols = [f"winner.card{i}.id" for i in range(1, 9)]
    los_cols = [f"loser.card{i}.id" for i in range(1, 9)]
    
    id_to_name = dict(zip(cards["team.card1.id"].astype(int), cards["team.card1.name"]))
    name_to_id = {v: k for k, v in id_to_name.items()}
    
    print("\nExtracting decks...")
    winner_decks = [deck_to_tuple(row, win_cols) for _, row in daydf.iterrows()]
    loser_decks = [deck_to_tuple(row, los_cols) for _, row in daydf.iterrows()]
    all_decks = winner_decks + loser_decks
    deck_counts = Counter(all_decks)
    top_decks = deck_counts.most_common(50)
    
    print(f"Total battles: {len(daydf):,}")
    print(f"Unique decks: {len(deck_counts):,}")
    
    # --- 2) Define target deck ---
    target_deck = top_decks[args.target_rank - 1][0]
    target_name = deck_to_names(target_deck, id_to_name)
    target_cards = [id_to_name.get(cid, str(cid)) for cid in target_deck]
    
    print(f"\n{'=' * 70}")
    print(f"TARGET DECK (rank {args.target_rank})")
    print("=" * 70)
    print(f"Cards: {target_name}")
    print(f"\nIndividual cards:")
    for card in target_cards:
        print(f"  - {card}")
    
    # --- 3) Train model ---
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
    
    # --- 4) Compute counter-scores from coefficients ---
    print(f"\n{'=' * 70}")
    print(f"TOP {args.top_cards} COUNTER-CARDS (from model coefficients)")
    print("=" * 70)
    
    coefs = model.coef_.ravel()
    counter_scores = compute_counter_scores(target_deck, maps, coefs, id_to_name)
    
    print(f"\n{'Rank':<5} {'Score':>8}  {'Card'}")
    print("-" * 50)
    for i, (card_name, card_id, score) in enumerate(counter_scores[:args.top_cards], 1):
        print(f"{i:<5} {score:>+8.4f}  {card_name}")
    
    # Get top counter card IDs for deck search
    top_counter_ids = [card_id for _, card_id, _ in counter_scores[:args.top_cards]]
    
    # --- 5) Find real decks containing top counter-cards ---
    print(f"\n{'=' * 70}")
    print(f"REAL DECKS CONTAINING {args.min_matches}+ TOP COUNTER-CARDS")
    print("=" * 70)
    
    matching_decks = find_decks_with_cards(
        top_counter_ids, deck_counts, 
        min_matches=args.min_matches, 
        min_appearances=100
    )
    
    print(f"\nFound {len(matching_decks)} decks with {args.min_matches}+ counter-cards (≥100 appearances)")
    
    # Show top 10 and compute empirical win rates
    print(f"\n{'Rank':<5} {'Match':>5} {'Appear':>8} {'vs Target':>12}  {'Deck'}")
    print("-" * 90)
    
    for i, (deck, num_matches, appearances) in enumerate(matching_decks[:15], 1):
        # Compute empirical win rate vs target
        wins, total, win_rate = compute_empirical_winrate(deck, target_deck, winner_decks, loser_decks)
        
        # Highlight which cards are counter-cards
        deck_names = []
        for cid in deck:
            name = id_to_name.get(cid, str(cid))
            if cid in top_counter_ids:
                name = f"**{name}**"
            deck_names.append(name)
        
        wr_str = f"{win_rate:.0%} ({total}g)" if total > 0 else "N/A"
        deck_str = ", ".join(deck_names)[:50]
        print(f"{i:<5} {num_matches:>5} {appearances:>8} {wr_str:>12}  {deck_str}...")
    
    # --- 6) Detailed analysis of best matching deck ---
    if matching_decks:
        print(f"\n{'=' * 70}")
        print("BEST MATCHING DECK (detailed)")
        print("=" * 70)
        
        best_deck, num_matches, appearances = matching_decks[0]
        wins, total, win_rate = compute_empirical_winrate(best_deck, target_deck, winner_decks, loser_decks)
        
        print(f"\nDeck: {deck_to_names(best_deck, id_to_name)}")
        print(f"Appearances: {appearances:,}")
        print(f"Counter-cards matched: {num_matches}/{args.top_cards}")
        print(f"Empirical win rate vs target: {win_rate:.1%} ({wins}/{total} games)")
        
        print(f"\nCards breakdown:")
        for cid in best_deck:
            name = id_to_name.get(cid, str(cid))
            # Find this card's counter score
            card_score = None
            for cn, ci, cs in counter_scores:
                if ci == cid:
                    card_score = cs
                    break
            
            if cid in top_counter_ids:
                rank = top_counter_ids.index(cid) + 1
                print(f"  ★ {name} (counter rank #{rank}, score: {card_score:+.4f})")
            else:
                print(f"    {name}")
    
    # --- 7) Hidden Counter Search ---
    print(f"\n{'=' * 70}")
    print("HIDDEN COUNTERS (underrated decks the model thinks should work)")
    print("=" * 70)
    print(f"\nCriteria: 3+ counter-cards, ≥200 appearances, <10 games vs target")
    print("These are real decks that should counter X-Bow but are rarely used against it!\n")
    
    hidden_counters = []
    for deck, num_matches, appearances in matching_decks:
        if appearances < 200:
            continue
        wins, total, win_rate = compute_empirical_winrate(deck, target_deck, winner_decks, loser_decks)
        if total < 10:  # Underused against target
            # Compute total counter-score for this deck
            deck_counter_score = sum(
                score for _, card_id, score in counter_scores 
                if card_id in deck
            )
            hidden_counters.append((deck, num_matches, appearances, total, deck_counter_score))
    
    # Sort by counter-score (model's confidence)
    hidden_counters.sort(key=lambda x: -x[4])
    
    print(f"Found {len(hidden_counters)} hidden counter candidates!\n")
    
    print(f"{'Rank':<5} {'Cards':>5} {'Appear':>7} {'vsXbow':>7} {'Score':>8}  {'Deck'}")
    print("-" * 95)
    
    for i, (deck, num_matches, appearances, games_vs_target, counter_score) in enumerate(hidden_counters[:10], 1):
        # Highlight counter-cards
        deck_names = []
        for cid in deck:
            name = id_to_name.get(cid, str(cid))
            if cid in top_counter_ids:
                name = f"**{name}**"
            deck_names.append(name)
        
        deck_str = ", ".join(deck_names)
        if len(deck_str) > 55:
            deck_str = deck_str[:52] + "..."
        
        games_str = f"{games_vs_target}g" if games_vs_target > 0 else "0g"
        print(f"{i:<5} {num_matches:>5} {appearances:>7} {games_str:>7} {counter_score:>+8.3f}  {deck_str}")
    
    # Detailed breakdown of top hidden counter
    if hidden_counters:
        print(f"\n{'=' * 70}")
        print("TOP HIDDEN COUNTER (detailed)")
        print("=" * 70)
        
        best_hidden, num_matches, appearances, games_vs, counter_score = hidden_counters[0]
        
        print(f"\nDeck: {deck_to_names(best_hidden, id_to_name)}")
        print(f"Total appearances: {appearances:,}")
        print(f"Games vs X-Bow: {games_vs} (underused!)")
        print(f"Counter-cards matched: {num_matches}")
        print(f"Model counter-score: {counter_score:+.3f}")
        
        print(f"\nCards breakdown:")
        for cid in best_hidden:
            name = id_to_name.get(cid, str(cid))
            card_score = None
            for cn, ci, cs in counter_scores:
                if ci == cid:
                    card_score = cs
                    break
            
            if cid in top_counter_ids:
                rank = top_counter_ids.index(cid) + 1
                print(f"  ★ {name} (counter rank #{rank}, score: {card_score:+.4f})")
            elif card_score is not None:
                print(f"    {name} (score: {card_score:+.4f})")
            else:
                print(f"    {name}")
        
        print(f"\n💡 MODEL RECOMMENDATION: This deck contains {num_matches} counter-cards")
        print(f"   and appears {appearances:,} times, but has only been used {games_vs} times")
        print(f"   against X-Bow. The model predicts it should be a strong counter!")

    # --- 8) Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The model identified these as the top counter-cards against {target_cards[5] if len(target_cards) > 5 else 'target'}:

Top 5 counter-cards:""")
    for i, (card_name, _, score) in enumerate(counter_scores[:5], 1):
        print(f"  {i}. {card_name} (score: {score:+.4f})")
    
    if hidden_counters:
        best_hidden, num_matches, appearances, games_vs, counter_score = hidden_counters[0]
        print(f"""
Top hidden counter (underrated):
  {deck_to_names(best_hidden, id_to_name)}
  
  - Contains {num_matches} counter-cards (score: {counter_score:+.3f})
  - Appears {appearances:,} times in data
  - Only {games_vs} games vs X-Bow (underused!)
  - Model says: "This should work, try it!"
""")


if __name__ == "__main__":
    main()

