# models/nn.py
# Purpose: Lightweight neural-network adapter using sklearn's MLPClassifier.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier


@dataclass
class NNAdapter:
    """Neural network adapter (sklearn MLPClassifier) exposing fit/predict methods."""

    hidden_layer_sizes: Tuple[int, ...] = (100,)
    activation: str = "relu"
    alpha: float = 1e-3
    learning_rate_init: float = 1e-4
    max_iter: int = 200
    random_state: Optional[int] = 42
    early_stopping: bool = True
    validation_fraction: float = 0.1
    tol: float = 1e-4
    verbose: bool = False
    # Adapter metadata
    name: str = field(default="nn", init=False)
    requires_dense: bool = field(default=True, init=False)
    supports_predict_proba: bool = field(default=True, init=False)

    # Internal estimator
    _est: Optional[MLPClassifier] = field(default=None, init=False, repr=False)

    def _to_dense(self, X):
        if sp.issparse(X):
            n, d = X.shape
            # if n * d > 4e8:
            #     raise ValueError(
            #         f"Refusing to densify a huge matrix of shape {X.shape}. "
            #         "Use feature preselection or a sparse-friendly model."
            #     )
            return X.toarray()
        return X

    def get_params(self) -> Dict[str, Any]:
        return dict(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X, y, *, sample_weight: Optional[np.ndarray] = None) -> "NNAdapter":
        Xd = self._to_dense(X)

        # Basic numeric hygiene: ensure finite values
        if not np.all(np.isfinite(Xd)):
            # Replace non-finite with zeros and warn
            import warnings
            warnings.warn("Non-finite values found in input X; replacing with zeros.")
            Xd = np.nan_to_num(Xd, copy=False)

        params = {k: v for k, v in self.get_params().items() if v is not None}
        # Use 'adam' solver (robust default) and enable early stopping for stability
        params.update(dict(solver="adam", early_stopping=bool(self.early_stopping),
                           validation_fraction=float(self.validation_fraction), tol=float(self.tol), verbose=bool(self.verbose)))

        est = MLPClassifier(**params)
        est.fit(Xd, y)
        self._est = est
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._est is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        Xd = self._to_dense(X)
        return self._est.predict_proba(Xd)

    def predict(self, X, *, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= float(threshold)).astype(np.int32)

    @property
    def loss_curve_(self):
        """Proxy to underlying MLPClassifier.loss_curve_ (if available)."""
        if self._est is not None and hasattr(self._est, "loss_curve_"):
            return self._est.loss_curve_
        return None

    @property
    def loss_(self):
        """Proxy to underlying MLPClassifier.loss_ (if available)."""
        if self._est is not None and hasattr(self._est, "loss_"):
            return self._est.loss_
        return None



__all__ = ["NNAdapter"]
