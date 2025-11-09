# models/logreg.py
# --------------------------------------------------------------------------------------
# Purpose
#   Wrap scikit-learn LogisticRegression behind a tiny, consistent adapter API so it can
#   be selected by name and swapped with other models (AdaBoost, XGBoost, etc.) without
#   touching feature code. The defaults are identical to your original script.
#
# Design
#   - No feature logic here. This file only trains a linear classifier on whatever
#     feature matrix it is given (e.g., [deck | pairwise | Δ]).
#   - Defaults mirror the original: solver="saga", penalty="l2", C=1.0,
#     fit_intercept=False, max_iter=400, n_jobs=-1, verbose=1.
#   - Methods: fit(), predict_proba(), predict().
#
# Example
#   from models.logreg import LogRegAdapter
#   model = LogRegAdapter()
#   model.fit(X_train, y_train)
#   p_val = model.predict_proba(X_val)[:, 1]
# --------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LogRegAdapter:
    """Logistic regression adapter (matches legacy settings).

    Parameters
    ----------
    solver : str
        Optimization algorithm (default "saga").
    penalty : str
        Regularization type (default "l2").
    C : float
        Inverse regularization strength (default 1.0).
    fit_intercept : bool
        Whether to fit an intercept (default False to match A−B feature design).
    max_iter : int
        Maximum iterations for the solver (default 400).
    n_jobs : int
        Number of parallel threads (default -1 = all cores).
    verbose : int
        Verbosity level (default 1).
    class_weight : Optional[dict]
        Passed through to scikit-learn if provided.
    """

    solver: str = "saga"
    penalty: str = "l2"
    C: float = 1.0
    fit_intercept: bool = False
    max_iter: int = 400
    n_jobs: int = -1
    verbose: int = 1
    class_weight: Optional[Dict[int, float]] = None

    # Adapter metadata
    name: str = field(default="logreg", init=False)
    requires_dense: bool = field(default=False, init=False)
    supports_predict_proba: bool = field(default=True, init=False)

    # Internal sklearn estimator (initialized in fit)
    _est: Optional[LogisticRegression] = field(default=None, init=False, repr=False)

    def get_params(self) -> Dict[str, Any]:
        """Return the effective sklearn parameters used by the underlying model."""
        return dict(
            solver=self.solver,
            penalty=self.penalty,
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            class_weight=self.class_weight,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X, y, *, sample_weight: Optional[np.ndarray] = None) -> "LogRegAdapter":
        """Fit logistic regression on (X, y) with legacy-default settings.

        Notes
        -----
        - `X` can be CSR sparse; scikit-learn handles it fine for `saga`.
        - `y` should be 0/1 integers (caller ensures this).
        """
        params = self.get_params()
        est = LogisticRegression(**{k: v for k, v in params.items() if v is not None})
        est.fit(X, y, sample_weight=sample_weight)
        self._est = est
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._est.predict_proba(X)

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        """Return hard predictions using a probability threshold (default 0.5)."""
        p = self.predict_proba(X)[:, 1]
        return (p >= float(threshold)).astype(np.int32)

    def decision_function(self, X) -> np.ndarray:
        """Return raw decision scores (log-odds) when available."""
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        # For LogisticRegression, decision_function returns log-odds
        return self._est.decision_function(X)

    # Convenience accessors -------------------------------------------------
    @property
    def coef_(self):  # type: ignore[override]
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._est.coef_

    @property
    def intercept_(self):  # type: ignore[override]
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._est.intercept_
