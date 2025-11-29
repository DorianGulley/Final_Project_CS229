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

from data import get_raw_data, DEFAULT_HANDLE
from features import build_all_features
from models import get_model
from metrics import evaluate, pretty_print


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
    p.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength.")
    p.add_argument("--max-iter", type=int, default=400, help="LogReg max iterations.")
    p.add_argument("--verbose", type=int, default=1, help="LogReg verbosity level.")

    # Run options
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--save", default=None, help="Optional path to save metrics as JSON.")

    return p.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Seed
    np.random.seed(args.seed)

    # 1) Load raw data
    daydf, cards, paths = get_raw_data(
        handle=args.handle,
        use_local=args.use_local,
        force_download=args.force_download,
        cache_dir=args.cache_dir,
        quick=args.quick,
    )

    # 2) Build features (maps → splits → blocks → standardize → assemble)
    use_blocks: Sequence[str] = list(args.blocks)
    ds = build_all_features(
        daydf,
        cards,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_blocks=use_blocks,
    )

    print("\nShapes:")
    print("  Train:", ds.X_train.shape, ds.y_train.shape)
    print("  Val  :", ds.X_val.shape, ds.y_val.shape)
    print("  Test :", ds.X_test.shape, ds.y_test.shape)

    # 3) Pick model by name
    if args.model.lower() == "logreg":
        model = get_model(
            "logreg",
            C=float(args.C),
            max_iter=int(args.max_iter),
            verbose=int(args.verbose),
            fit_intercept=False,
        )
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

    # 5) Evaluate on VAL
    p_val = model.predict_proba(ds.X_val)[:, 1]
    metrics_val = evaluate(ds.y_val, p_val)
    from metrics import pretty_print as _pp
    _pp(metrics_val)

    # Optional: also report TEST (keep VAL as the model-selection target)
    p_test = model.predict_proba(ds.X_test)[:, 1]
    metrics_test = evaluate(ds.y_test, p_test)
    print("\n=== Test (for reference only) ===")
    pretty_print(metrics_test)

    # 6) Save metrics (optional)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "params": {
                "C": float(args.C),
                "max_iter": int(args.max_iter),
                "verbose": int(args.verbose),
                "fit_intercept": False,
            },
            "features": list(use_blocks),
            "split": {"train_frac": args.train_frac, "val_frac": args.val_frac},
            "dataset_root": str(paths.base_dir),
            "metrics": {"val": metrics_val, "test": metrics_test},
            "seed": int(args.seed),
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics → {out}")


if __name__ == "__main__":
    main(parse_args())
