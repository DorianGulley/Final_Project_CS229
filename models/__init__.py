# models/__init__.py
# --------------------------------------------------------------------------------------
# Purpose
#   Central model registry and convenience imports for adapters.
#   Allows dynamic model selection by name (e.g., 'logreg', 'adaboost', 'xgb').
#   Keeps a consistent interface: fit(), predict_proba(), predict().
# --------------------------------------------------------------------------------------

from typing import Dict, Type

# Import adapters here
from models.logreg import LogRegAdapter

# Optional future imports (placeholders for extensibility)
from models.adaboost import AdaBoostAdapter
from models.xgb import XGBAdapter

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type] = {
    "logreg": LogRegAdapter,
    "adaboost": AdaBoostAdapter,
    "xgb": XGBAdapter,
}

# --------------------------------------------------------------------------------------
# Factory function
# --------------------------------------------------------------------------------------

def get_model(name: str, **kwargs):
    """Instantiate a model adapter by name.

    Parameters
    ----------
    name : str
        Model key (e.g., 'logreg').
    **kwargs
        Extra parameters forwarded to the adapter constructor.

    Returns
    -------
    object
        Instantiated model adapter.

    Raises
    ------
    KeyError
        If name is not found in MODEL_REGISTRY.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = ["get_model", "MODEL_REGISTRY", "LogRegAdapter"]