# explore_decks.py
# Quick exploration of deck frequencies and head-to-head matchups
# Run with: python explore_decks.py --quick  (for faster iteration)

from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from data import get_raw_data


def deck_to_tuple(row, cols):
    """Convert a row's card columns to a sorted tuple (canonical deck representation)."""
    cards = tuple(sorted(int(row[c]) for c in cols))
    return cards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use quick mode (single day)")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    daydf, cards, paths = get_raw_data(quick=args.quick, sample=args.sample)
    
    win_cols = [f"winner.card{i}.id" for i in range(1, 9)]
    los_cols = [f"loser.card{i}.id" for i in range(1, 9)]
    
    print(f"\nTotal battles: {len(daydf):,}")
    
    # --- 1) Extract all decks (winner and loser) ---
    print("\nExtracting decks...")
    winner_decks = [deck_to_tuple(row, win_cols) for _, row in daydf.iterrows()]
    loser_decks = [deck_to_tuple(row, los_cols) for _, row in daydf.iterrows()]
    
    # Count all deck appearances
    all_decks = winner_decks + loser_decks
    deck_counts = Counter(all_decks)
    
    print(f"\n{'='*60}")
    print("DECK FREQUENCY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total deck appearances: {len(all_decks):,}")
    print(f"Unique decks: {len(deck_counts):,}")
    
    # --- 2) Distribution of deck frequencies ---
    counts = np.array(list(deck_counts.values()))
    print(f"\nDeck frequency distribution:")
    print(f"  Min appearances:    {counts.min()}")
    print(f"  Max appearances:    {counts.max():,}")
    print(f"  Mean appearances:   {counts.mean():.1f}")
    print(f"  Median appearances: {np.median(counts):.0f}")
    
    # How many decks appear at least N times?
    thresholds = [1, 2, 5, 10, 50, 100, 500, 1000, 5000]
    print(f"\nDecks with at least N appearances:")
    for thresh in thresholds:
        n_decks = sum(1 for c in counts if c >= thresh)
        print(f"  N ≥ {thresh:>5}: {n_decks:>6,} decks")
    
    # --- 3) Top 20 most common decks ---
    print(f"\n{'='*60}")
    print("TOP 20 MOST COMMON DECKS")
    print(f"{'='*60}")
    
    # Build card ID -> name mapping
    id_to_name = dict(zip(cards["team.card1.id"].astype(int), cards["team.card1.name"]))
    
    top_20 = deck_counts.most_common(20)
    for rank, (deck, count) in enumerate(top_20, 1):
        card_names = [id_to_name.get(cid, f"?{cid}") for cid in deck]
        print(f"\n{rank}. Appears {count:,} times")
        print(f"   {', '.join(card_names)}")
    
    # --- 4) Head-to-head analysis for the most popular deck ---
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD ANALYSIS (Most Popular Deck)")
    print(f"{'='*60}")
    
    target_deck = top_20[0][0]
    target_name = ", ".join(id_to_name.get(cid, f"?{cid}") for cid in target_deck)
    print(f"\nTarget deck: {target_name}")
    print(f"Total appearances: {top_20[0][1]:,}")
    
    # Find all battles where target_deck appeared
    target_as_winner = []
    target_as_loser = []
    
    for i, (w_deck, l_deck) in enumerate(zip(winner_decks, loser_decks)):
        if w_deck == target_deck:
            target_as_winner.append(l_deck)
        if l_deck == target_deck:
            target_as_loser.append(w_deck)
    
    print(f"\nBattles where target was winner: {len(target_as_winner):,}")
    print(f"Battles where target was loser:  {len(target_as_loser):,}")
    print(f"Total battles involving target:  {len(target_as_winner) + len(target_as_loser):,}")
    
    # Count opponent decks
    opponent_when_won = Counter(target_as_winner)
    opponent_when_lost = Counter(target_as_loser)
    all_opponents = Counter(target_as_winner + target_as_loser)
    
    print(f"\nUnique opponent decks faced: {len(all_opponents):,}")
    
    # How many opponents did target face at least N times?
    opp_counts = np.array(list(all_opponents.values()))
    print(f"\nOpponents faced at least N times:")
    for thresh in [1, 2, 5, 10, 20, 50, 100]:
        n_opps = sum(1 for c in opp_counts if c >= thresh)
        print(f"  N ≥ {thresh:>3}: {n_opps:>5,} opponent decks")
    
    # --- 5) Empirical win rates against frequent opponents ---
    print(f"\n{'='*60}")
    print("EMPIRICAL WIN RATES vs FREQUENT OPPONENTS")
    print(f"{'='*60}")
    print("(Opponents that target deck faced ≥ 20 times)\n")
    
    frequent_opponents = [(deck, all_opponents[deck]) for deck in all_opponents if all_opponents[deck] >= 20]
    frequent_opponents.sort(key=lambda x: -x[1])  # Sort by frequency
    
    results = []
    for opp_deck, total_games in frequent_opponents[:30]:  # Top 30 frequent opponents
        wins = opponent_when_won.get(opp_deck, 0)  # Times target beat this opponent
        losses = opponent_when_lost.get(opp_deck, 0)  # Times target lost to this opponent
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        opp_names = [id_to_name.get(cid, f"?{cid}") for cid in opp_deck]
        results.append({
            "opponent": ", ".join(opp_names[:3]) + "...",  # Truncate for display
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        })
    
    print(f"{'Opponent (first 3 cards)':<35} {'Games':>6} {'W':>5} {'L':>5} {'WinRate':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['opponent']:<35} {r['games']:>6} {r['wins']:>5} {r['losses']:>5} {r['win_rate']:>7.1%}")
    
    print(f"\n✓ Found {len(frequent_opponents)} opponent decks with ≥20 games against target")
    print("  This is your candidate pool for empirical validation!")


if __name__ == "__main__":
    main()

