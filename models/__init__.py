# models/__init__.py
# Purpose: Central model registry and convenience imports for model adapters.

from typing import Dict, Type

# Import adapters here
from models.logreg import LogRegAdapter

# Optional future imports (placeholders for extensibility)
from models.adaboost import AdaBoostAdapter
from models.xgb import XGBAdapter
from models.nn import NNAdapter

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type] = {
    "logreg": LogRegAdapter,
    "adaboost": AdaBoostAdapter,
    "xgb": XGBAdapter,
    "nn": NNAdapter,
}

# --------------------------------------------------------------------------------------
# Factory function
# --------------------------------------------------------------------------------------

def get_model(name: str, **kwargs):
    """Instantiate a model adapter by name. Extra kwargs are forwarded to the adapter."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


__all__ = ["get_model", "MODEL_REGISTRY", "LogRegAdapter"]