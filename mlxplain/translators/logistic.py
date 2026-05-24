"""Logistic regression translator — coefficients x values -> contributions."""

from __future__ import annotations

import numpy as np

from mlxplain.core.counterfactual import compute_counterfactuals_logistic
from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


class LogisticTranslator(BaseTranslator):
    """Translates logistic regression predictions into feature contributions."""

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        instance = X[idx].reshape(1, -1)
        return float(model.predict_proba(instance)[0, 1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        instance = X[idx]
        coefs = model.coef_[0]
        contributions = coefs * instance

        drivers = []
        for name, val, contrib in zip(feature_names, instance, contributions, strict=False):
            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(val),
                    impact=float(abs(contrib)),
                    direction="positive" if contrib >= 0 else "negative",
                )
            )

        drivers.sort(key=lambda d: d.impact, reverse=True)
        return drivers

    def compute_counterfactuals(
        self,
        model,
        X: np.ndarray,
        idx: int,
        threshold: float,
        feature_names: list[str],
    ) -> list[Counterfactual]:
        return compute_counterfactuals_logistic(model, X[idx], threshold, feature_names)
