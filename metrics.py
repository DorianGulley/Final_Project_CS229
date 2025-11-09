# metrics.py
# --------------------------------------------------------------------------------------
# Purpose
#   Centralized evaluation utilities for binary classifiers in the Clash Royale S18 project.
#   - Computes common metrics on probabilities (ROC-AUC, LogLoss, Brier, Accuracy@τ)
#   - Lightweight pretty printer and optional confusion-matrix helper
#   - Returns a serializable dict for saving experiment reports
#
# Usage
#   from metrics import evaluate, pretty_print, confusion
#   m = evaluate(y_val, p_val, threshold=0.5)
#   pretty_print(m)
#   tn, fp, fn, tp = confusion(y_val, p_val, threshold=0.5)
# --------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    confusion_matrix,
)

# --------------------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard metrics given true labels and predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels with values {0,1}.
    p : np.ndarray
        Predicted positive-class probabilities in [0,1]. Shape (n,) or (n,1) or (n,2) -> will be coerced.
    threshold : float
        Probability threshold for Accuracy@threshold (default 0.5).

    Returns
    -------
    Dict[str, float]
        A dictionary with keys: 'auc', 'logloss', 'brier', 'acc', 'threshold'.
    """
    # Coerce shapes
    if p.ndim == 2 and p.shape[1] == 2:
        p = p[:, 1]
    p = p.reshape(-1)

    # Guardrails for numerical stability
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)

    auc = float(roc_auc_score(y_true, p_clip))
    ll = float(log_loss(y_true, p_clip, labels=[0, 1]))
    brier = float(brier_score_loss(y_true, p_clip))
    yhat = (p_clip >= float(threshold)).astype(np.int32)
    acc = float(accuracy_score(y_true, yhat))

    return {
        "auc": auc,
        "logloss": ll,
        "brier": brier,
        "acc": acc,
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------------------

def pretty_print(metrics: Dict[str, float]) -> None:
    """Print a compact, aligned summary of metric values."""
    print("\n=== Metrics ===")
    print(f"ROC-AUC     : {metrics['auc']:.6f}")
    print(f"Log loss    : {metrics['logloss']:.6f}")
    print(f"Brier score : {metrics['brier']:.6f}")
    print(f"Accuracy@{metrics['threshold']:.2f}: {metrics['acc']:.6f}")


def confusion(y_true: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) at a given probability threshold."""
    if p.ndim == 2 and p.shape[1] == 2:
        p = p[:, 1]
    yhat = (p.reshape(-1) >= float(threshold)).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


# --------------------------------------------------------------------------------------
# Optional: calibration summary (logit mean, prob mean) for quick sanity checks
# --------------------------------------------------------------------------------------

def quick_calibration_summary(p: np.ndarray) -> Dict[str, float]:
    """Return basic distribution stats of predicted probabilities.

    Useful for spotting degenerate models (e.g., extremely peaked or flat outputs).
    """
    if p.ndim == 2 and p.shape[1] == 2:
        p = p[:, 1]
    p = p.reshape(-1)
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logit = np.log(p_clip) - np.log(1 - p_clip)
    return {
        "p_mean": float(p_clip.mean()),
        "p_std": float(p_clip.std()),
        "logit_mean": float(logit.mean()),
        "logit_std": float(logit.std()),
    }


if __name__ == "__main__":  # pragma: no cover
    # Tiny smoke test
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    p = rng.random(1000)
    m = evaluate(y, p)
    pretty_print(m)
    print("Confusion:", confusion(y, p))
    print("Calib:", quick_calibration_summary(p))
