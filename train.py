# train.py
# --------------------------------------------------------------------------------------
# Purpose
#   End-to-end training script that wires together:
#     - data.get_raw_data()        → raw DataFrames
#     - features.build_all_features → sparse model-ready matrices
#     - models.get_model("logreg") → logistic regression adapter (default)
#     - metrics.evaluate()          → common metrics on VAL (and TEST optional)
#
# Usage (examples)
#   python train.py                          # default: model=logreg, blocks=[deck,ab,delta]
#   python train.py --model logreg --use-local /path/to/dataset/root
#   python train.py --blocks deck ab         # ablation: deck-only
#   python train.py --save reports/run1.json # persist metrics
#   python train.py --show-counters 20       # show top 20 card counter matchups
#   python train.py --save-counters counters.csv  # save all counter matchups to CSV
#
# Notes
#   - Keeps the exact modeling defaults as the legacy script.
#   - Splits are time-respecting (70/15/15) and Δ is standardized on TRAIN only.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data import get_raw_data, DEFAULT_HANDLE
from features import build_all_features, FeatureMaps
from models import get_model
from metrics import evaluate, pretty_print


# --------------------------------------------------------------------------------------
# Card Counter Analysis (for logreg interpretability)
# --------------------------------------------------------------------------------------

def build_counter_table(
    coefs: np.ndarray,
    maps: FeatureMaps,
    cards_df: pd.DataFrame,
    use_blocks: Sequence[str],
    card_id_col: str = "team.card1.id",
    card_name_col: str = "team.card1.name",
) -> pd.DataFrame:
    """Extract the A×B counter coefficients and build a human-readable table.
    
    Returns a DataFrame with columns:
        M_ij, i_idx, i_id, i_name, j_idx, j_id, j_name, interpretation
    """
    # If "ab" block isn't used, return empty
    if "ab" not in use_blocks:
        return pd.DataFrame()
    
    # Build card_id -> name mapping
    id_to_name = dict(zip(cards_df[card_id_col].astype(int), cards_df[card_name_col]))
    
    # Reverse map: col_idx -> card_id
    col_to_card_id = {v: k for k, v in maps.card_to_col.items()}
    card_names = [id_to_name.get(col_to_card_id[i], f"Card_{col_to_card_id[i]}") for i in range(maps.D)]
    
    # Calculate offset for the "ab" block in the coefficient vector
    offset = 0
    for block in use_blocks:
        if block == "ab":
            break
        elif block == "deck":
            offset += maps.D
        elif block == "delta":
            offset += 1
    
    # Extract A×B counter weights
    M = coefs[offset:offset + maps.P]
    
    # Reverse map: column -> (i, j) pair with i < j
    col_to_pair = {v: k for k, v in maps.pair_to_col.items()}
    
    # Build the table
    rows = []
    for c in range(maps.P):
        i, j = col_to_pair[c]
        w = float(M[c])
        card_i_id = col_to_card_id[i]
        card_j_id = col_to_card_id[j]
        
        # Interpretation: positive => "i counters j", negative => "j counters i"
        if w >= 0:
            interp = f"{card_names[i]} → counters → {card_names[j]}"
        else:
            interp = f"{card_names[j]} → counters → {card_names[i]}"
        
        rows.append({
            "M_ij": w,
            "i_idx": i,
            "i_id": int(card_i_id),
            "i_name": card_names[i],
            "j_idx": j,
            "j_id": int(card_j_id),
            "j_name": card_names[j],
            "interpretation": interp,
        })
    
    return pd.DataFrame(rows)


def print_top_counters(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print the top positive and negative counter matchups."""
    # Sort by M_ij
    df_sorted = df.sort_values("M_ij", ascending=False)
    
    top_pos = df_sorted.head(top_n)
    top_neg = df_sorted.tail(top_n).iloc[::-1]  # Most negative first
    
    print("\n" + "=" * 80)
    print(f"TOP {top_n} COUNTER MATCHUPS (positive M_ij: i counters j)")
    print("=" * 80)
    print(f"{'Rank':<5} {'M_ij':>10}  {'Interpretation'}")
    print("-" * 80)
    for rank, (_, row) in enumerate(top_pos.iterrows(), 1):
        print(f"{rank:<5} {row['M_ij']:>+10.6f}  {row['interpretation']}")
    
    print("\n" + "=" * 80)
    print(f"TOP {top_n} REVERSE MATCHUPS (negative M_ij: j counters i)")
    print("=" * 80)
    print(f"{'Rank':<5} {'M_ij':>10}  {'Interpretation'}")
    print("-" * 80)
    for rank, (_, row) in enumerate(top_neg.iterrows(), 1):
        print(f"{rank:<5} {row['M_ij']:>+10.6f}  {row['interpretation']}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    p = argparse.ArgumentParser(description="Clash Royale S18 — Training Runner")

    # Data options
    p.add_argument("--handle", default=DEFAULT_HANDLE, help="KaggleHub dataset handle.")
    p.add_argument("--use-local", default=None, help="Use existing dataset directory (skip download).")
    p.add_argument("--force-download", action="store_true", help="Force re-download (ignored with --use-local).")
    p.add_argument("--cache-dir", default=None, help="Optional kagglehub cache root.")
    p.add_argument("--quick", action="store_true", help="Load only Dec 27, 2020 data (~1GB) for faster iteration.")
    p.add_argument("--sample", type=float, default=None, help="Randomly sample this fraction of data (e.g., 0.1 for 10%%).")

    # Feature options
    p.add_argument(
        "--blocks",
        nargs="*",
        default=["deck", "ab", "delta"],
        help="Feature blocks to include (subset/order of: deck ab delta).",
    )
    p.add_argument("--train-frac", type=float, default=0.70, help="Train fraction (default 0.70).")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction (default 0.15).")

    # Model options
    p.add_argument("--model", default="logreg", help="Model name (default 'logreg').")
    p.add_argument("--C", type=float, default=100.0, help="LogReg inverse regularization strength.")
    p.add_argument("--no-regularization", action="store_true", help="Disable regularization (sets penalty='none').")
    p.add_argument("--max-iter", type=int, default=400, help="LogReg max iterations.")
    p.add_argument("--verbose", type=int, default=1, help="LogReg verbosity level.")

    # Run options
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--save", default=None, help="Optional path to save metrics as JSON.")
    p.add_argument("--eval-test", action="store_true", help="Evaluate on test set (only use for FINAL evaluation).")

    # Counter analysis options (logreg only)
    p.add_argument("--show-counters", type=int, default=0, 
                   help="Show top N card counter matchups (logreg only, requires 'ab' block).")
    p.add_argument("--save-counters", default=None, 
                   help="Save all counter matchups to CSV (logreg only).")

    return p.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Seed
    np.random.seed(args.seed)

    import gc
    
    # 1) Load raw data (sampling happens during loading to reduce peak memory)
    daydf, cards, paths = get_raw_data(
        handle=args.handle,
        use_local=args.use_local,
        force_download=args.force_download,
        cache_dir=args.cache_dir,
        quick=args.quick,
        sample=args.sample,
        random_state=args.seed,
    )

    # 2) Build features (maps → splits → blocks → standardize → assemble)
    use_blocks: Sequence[str] = list(args.blocks)
    
    # Keep cards if we need counter analysis later
    need_counter_analysis = (
        args.model.lower() == "logreg" 
        and (args.show_counters > 0 or args.save_counters)
        and "ab" in use_blocks
    )
    cards_for_analysis = cards.copy() if need_counter_analysis else None
    
    ds = build_all_features(
        daydf,
        cards,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_blocks=use_blocks,
    )
    
    # Free the raw DataFrame - we only need the sparse matrices now
    del daydf, cards
    gc.collect()

    print("\nShapes:")
    print("  Train:", ds.X_train.shape, ds.y_train.shape)
    print("  Val  :", ds.X_val.shape, ds.y_val.shape)
    print("  Test :", ds.X_test.shape, ds.y_test.shape)

    # 3) Pick model by name
    if args.model.lower() == "logreg":
        logreg_params = {
            "max_iter": int(args.max_iter),
            "verbose": int(args.verbose),
            "fit_intercept": False,
        }
        if args.no_regularization:
            logreg_params["penalty"] = None
        else:
            logreg_params["C"] = float(args.C)
            logreg_params["penalty"] = "l2"
        model = get_model("logreg", **logreg_params)
    elif args.model.lower() == "adaboost":
        model = get_model(
            "adaboost",
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
        )
    elif args.model.lower() in {"xgb", "xgboost"}:
        model = get_model(
            "xgb",
            max_depth=6,
            n_estimators=500,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",  # or "gpu_hist" if you have a GPU
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 4) Fit on TRAIN
    model.fit(ds.X_train, ds.y_train)

    # 5) Evaluate on TRAIN and VAL
    print("\n=== Train Metrics ===")
    p_train = model.predict_proba(ds.X_train)[:, 1]
    metrics_train = evaluate(ds.y_train, p_train)
    pretty_print(metrics_train)

    print("\n=== Validation Metrics ===")
    p_val = model.predict_proba(ds.X_val)[:, 1]
    metrics_val = evaluate(ds.y_val, p_val)
    pretty_print(metrics_val)

    # Only evaluate on TEST if explicitly requested (should only be done at the very end)
    metrics_test = None
    if args.eval_test:
        p_test = model.predict_proba(ds.X_test)[:, 1]
        metrics_test = evaluate(ds.y_test, p_test)
        print("\n=== Test (FINAL EVALUATION) ===")
        pretty_print(metrics_test)

    # 6) Card counter analysis (logreg only, requires 'ab' block)
    if need_counter_analysis:
        print("\n--- Analyzing Card Counter Matchups ---")
        coefs = model.coef_.ravel()
        counter_df = build_counter_table(coefs, ds.maps, cards_for_analysis, use_blocks)
        
        if args.show_counters > 0:
            print_top_counters(counter_df, top_n=args.show_counters)
        
        if args.save_counters:
            counter_path = Path(args.save_counters)
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            # Sort by absolute value (strongest matchups first)
            counter_df_sorted = counter_df.reindex(
                counter_df["M_ij"].abs().sort_values(ascending=False).index
            )
            counter_df_sorted.to_csv(counter_path, index=False)
            print(f"\nSaved {len(counter_df)} counter matchups → {counter_path}")

    # 7) Save metrics (optional)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        params_dict = {
            "max_iter": int(args.max_iter),
            "verbose": int(args.verbose),
            "fit_intercept": False,
        }
        if args.no_regularization:
            params_dict["penalty"] = None
        else:
            params_dict["C"] = float(args.C)
            params_dict["penalty"] = "l2"
        
        payload = {
            "model": args.model,
            "params": params_dict,
            "features": list(use_blocks),
            "split": {"train_frac": args.train_frac, "val_frac": args.val_frac},
            "dataset_root": str(paths.base_dir),
            "metrics": {"train": metrics_train, "val": metrics_val},
            "seed": int(args.seed),
        }
        if metrics_test is not None:
            payload["metrics"]["test"] = metrics_test
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics → {out}")


if __name__ == "__main__":
    main(parse_args())
