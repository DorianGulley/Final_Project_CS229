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

    def fit(self, X, y, *, sample_weight: Optional[np.ndarray] = None, eval_set: Optional[Tuple[Any, Any]] = None) -> "NNAdapter":
        """Fit the neural network.

        If `eval_set=(X_val, y_val)` is provided the adapter will train epoch-by-epoch
        (using `warm_start=True` and `max_iter=1` per call) and record both the
        training `loss_curve_` and a per-epoch `val_loss_curve_` (using
        `sklearn.metrics.log_loss`). This enables plotting training vs validation
        loss to diagnose overfitting.
        """
        Xd = self._to_dense(X)

        # Basic numeric hygiene: ensure finite values
        if not np.all(np.isfinite(Xd)):
            import warnings
            warnings.warn("Non-finite values found in input X; replacing with zeros.")
            Xd = np.nan_to_num(Xd, copy=False)

        Xv = None
        yv = None
        if eval_set is not None:
            Xv = self._to_dense(eval_set[0])
            yv = eval_set[1]

        params = {k: v for k, v in self.get_params().items() if v is not None}
        params.pop("max_iter", None)

        # We'll use 'adam' solver by default; enable warm_start so we can iterate
        params.update(dict(solver="adam", tol=float(self.tol), verbose=bool(self.verbose)))

        from sklearn.metrics import log_loss

        est = MLPClassifier(**params, warm_start=True, max_iter=1)

        train_losses = []
        val_losses = [] if Xv is not None else None


        # Iterate for up to `max_iter` epochs, recording train & val loss per epoch
        for epoch in range(int(self.max_iter)):
            est.fit(Xd, y)

            # training loss for this epoch
            tr_loss = est.loss_ if hasattr(est, "loss_") else None
            if tr_loss is not None:
                train_losses.append(float(tr_loss))

            # validation loss (log loss) if requested
            if Xv is not None and yv is not None:
                try:
                    probs = est.predict_proba(Xv)
                    # log_loss expects shape (n_samples, n_classes) or (n_samples,) for binary
                    val_loss = float(log_loss(yv, probs))
                except Exception:
                    # If predict_proba fails for any reason, append NaN and continue
                    val_loss = float("nan")
                val_losses.append(val_loss)

        # Save estimator and curves
        self._est = est
        # expose arrays in familiar attribute names
        self._loss_curve = train_losses
        self._val_loss_curve = val_losses
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
        if hasattr(self, "_loss_curve") and self._loss_curve is not None:
            return self._loss_curve
        if self._est is not None and hasattr(self._est, "loss_curve_"):
            return self._est.loss_curve_
        return None

    @property
    def loss_(self):
        """Proxy to underlying MLPClassifier.loss_ (if available)."""
        if self._est is not None and hasattr(self._est, "loss_"):
            return self._est.loss_
        return None

    @property
    def val_loss_curve_(self):
        """Per-epoch validation loss (log loss) recorded when `eval_set` was provided to `fit()`.

        Returns a list of floats (length == number of recorded epochs) or `None` if not recorded.
        """
        if hasattr(self, "_val_loss_curve"):
            return self._val_loss_curve
        return None



__all__ = ["NNAdapter"]
