# models/adaboost.py
# Purpose: Adapter wrapping scikit-learn's AdaBoostClassifier (uses shallow trees).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class AdaBoostAdapter:
    """AdaBoost adapter (shallow decision-tree base learners)."""

    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 3
    algorithm: str = "SAMME.R"
    random_state: Optional[int] = None

    # Adapter metadata
    name: str = field(default="adaboost", init=False)
    requires_dense: bool = field(default=True, init=False)
    supports_predict_proba: bool = field(default=True, init=False)

    # Internal sklearn estimator (initialized in fit)
    _est: Optional[AdaBoostClassifier] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dense(X):
        """Ensure a dense numpy array for sklearn's tree-based models.

        Raises a ValueError with a helpful message if the matrix is too large for
        naive densification.
        """
        if sp.issparse(X):
            n, d = X.shape
            # Heuristic guardrail: >4e8 elements (~3.2GB as float64) can be risky.
            if n * d > 4e8:
                raise ValueError(
                    f"Refusing to densify a huge matrix of shape {X.shape}. "
                    "Use feature preselection or a sparse-friendly booster (e.g., XGBoost)."
                )
            return X.toarray()
        return X

    def get_params(self) -> Dict[str, Any]:
        """Return the effective sklearn parameters used by the underlying model."""
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state,
            base_estimator=DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X, y, *, sample_weight: Optional[np.ndarray] = None) -> "AdaBoostAdapter":
        """Fit AdaBoost on (X, y) using shallow decision trees.

        Notes
        -----
        - Converts CSR inputs to dense arrays for sklearn's AdaBoost.
        - y should be 0/1 integers.
        """
        Xd = self._to_dense(X)
        est = AdaBoostClassifier(**self.get_params())
        est.fit(Xd, y, sample_weight=sample_weight)
        self._est = est
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2)."""
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        Xd = self._to_dense(X)
        return self._est.predict_proba(Xd)

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        """Return hard predictions using a probability threshold (default 0.5)."""
        p = self.predict_proba(X)[:, 1]
        return (p >= float(threshold)).astype(np.int32)

    # Convenience accessors -------------------------------------------------
    @property
    def feature_importances_(self):  # type: ignore[override]
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._est.feature_importances_
