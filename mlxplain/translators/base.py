"""Abstract base for model-specific translators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mlxplain.core.report import Counterfactual, FeatureDriver


class BaseTranslator(ABC):
    """Interface that every model-type translator must implement."""

    def __init__(self, language: str = "en"):
        self.language = language

    @abstractmethod
    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        """Return P(positive class) for instance at index `idx`."""

    def get_probabilities(self, model, X: np.ndarray, idx: int) -> dict[str, float]:
        """Return a dictionary mapping class labels to their probabilities."""
        instance = X[idx].reshape(1, -1)
        probs = model.predict_proba(instance)[0]
        classes = model.classes_
        return {str(c): float(p) for c, p in zip(classes, probs, strict=False)}

    @abstractmethod
    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        """Extract per-feature contributions for the given instance."""

    @abstractmethod
    def compute_counterfactuals(
        self,
        model,
        X: np.ndarray,
        idx: int,
        threshold: float,
        feature_names: list[str],
        immutable_features: list[str] | None = None,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        target_class: Any | None = None,
    ) -> list[Counterfactual]:
        """Compute minimum feature changes to flip the prediction."""
