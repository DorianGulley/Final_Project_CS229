# models/xgb.py
# --------------------------------------------------------------------------------------
# Purpose
#   Wrap XGBoost's XGBClassifier behind the same adapter API as other models so it can be
#   selected by name. XGBoost natively supports SciPy CSR matrices (sparse-friendly),
#   which makes it a good fit for large, sparse feature sets like [deck | ab | delta].
#
# Defaults
#   - objective="binary:logistic", booster="gbtree", tree_method="hist"
#   - max_depth=6, n_estimators=500, learning_rate=0.1
#   - subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=0.0
#   - n_jobs=-1, random_state=42
#   - early stopping optional via fit(..., X_val=..., y_val=..., early_stopping_rounds=50)
#
# Example
#   from models.xgb import XGBAdapter
#   model = XGBAdapter(max_depth=6, n_estimators=500, learning_rate=0.1)
#   model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=50)
#   p_val = model.predict_proba(X_val)[:, 1]
# --------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

import xgboost as xgb

@dataclass
class XGBAdapter:
    """XGBoost gradient-boosted trees adapter (sparse-friendly).

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (default 6).
    n_estimators : int
        Number of boosting rounds (default 500). If using early stopping, the
        effective best_iteration_ may be smaller.
    learning_rate : float
        Shrinkage factor (eta) (default 0.1).
    subsample : float
        Row subsampling per tree (default 0.8).
    colsample_bytree : float
        Column subsampling per tree (default 0.5).
    reg_lambda : float
        L2 regularization term on weights (default 1.0).
    reg_alpha : float
        L1 regularization term on weights (default 0.0).
    min_child_weight : float
        Minimum sum of instance weight needed in a child (default 1.0).
    gamma : float
        Minimum loss reduction to make a further partition (default 0.0).
    n_jobs : int
        Threads to use (default -1 for all cores).
    random_state : Optional[int]
        RNG seed for reproducibility (default 42).
    tree_method : str
        Tree construction algorithm (default 'hist'). Consider 'gpu_hist' if GPU is available.
    importance_type : str
        Importance type for feature_importances_ (default 'gain').
    """

    max_depth: int = 6
    n_estimators: int = 500
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    n_jobs: int = -1
    random_state: Optional[int] = 42
    tree_method: str = "hist"
    importance_type: str = "gain"

    # Adapter metadata
    name: str = field(default="xgb", init=False)
    requires_dense: bool = field(default=False, init=False)  # CSR OK
    supports_predict_proba: bool = field(default=True, init=False)

    # Internal estimator
    _est: Optional[xgb.XGBClassifier] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _ensure_xgb(self) -> None:
        if xgb is None:  # pragma: no cover
            raise ImportError(
                "xgboost is not installed. Install with `pip install xgboost`."
            )

    def get_params(self) -> Dict[str, Any]:
        """Return effective XGBClassifier params."""
        return dict(
            objective="binary:logistic",
            booster="gbtree",
            tree_method=self.tree_method,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            importance_type=self.importance_type,
            # Disable label encoder behavior from very old versions
            enable_categorical=False,
            eval_metric="logloss",  # default eval metric
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBAdapter":
        """Fit XGBoost on (X, y). Accepts optional validation set for early stopping."""
        self._ensure_xgb()
        params = self.get_params()
        est = xgb.XGBClassifier(**params)

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and early_stopping_rounds:
            fit_kwargs.update(
                dict(
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=int(early_stopping_rounds),
                    verbose=bool(verbose),
                )
            )
        else:
            # Keep verbosity consistent
            fit_kwargs["verbose"] = bool(verbose)

        est.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self._est = est
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        # XGBClassifier predict_proba returns (n, 2)
        return self._est.predict_proba(X)

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= float(threshold)).astype(np.int32)

    # Convenience ----------------------------------------------------------
    @property
    def feature_importances_(self) -> np.ndarray:  # type: ignore[override]
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._est.feature_importances_

    @property
    def best_iteration_(self) -> Optional[int]:
        if self._est is None:
            return None
        # Attribute exists when early stopping is used
        return getattr(self._est, "best_iteration", None)
