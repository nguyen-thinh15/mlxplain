"""Abstract base for model-specific translators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mlxplain.core.report import Counterfactual, FeatureDriver


class BaseTranslator(ABC):
    """Interface that every model-type translator must implement."""

    @abstractmethod
    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        """Return P(positive class) for instance at index `idx`."""

    @abstractmethod
    def extract_drivers(
        self, model, X: np.ndarray, idx: int, feature_names: list[str]
    ) -> list[FeatureDriver]:
        """Extract per-feature contributions for the given instance."""

    @abstractmethod
    def compute_counterfactuals(
        self,
        model,
        X: np.ndarray,
        idx: int,
        threshold: float,
        feature_names: list[str],
    ) -> list[Counterfactual]:
        """Compute minimum feature changes to flip the prediction."""
