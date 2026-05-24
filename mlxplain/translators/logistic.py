"""Logistic regression translator — coefficients x values -> contributions."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlxplain.core.counterfactual import compute_counterfactuals_logistic
from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


class LogisticTranslator(BaseTranslator):
    """Translates logistic regression predictions into feature contributions."""

    def __init__(self, language: str = "en"):
        super().__init__(language)

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        instance = X[idx].reshape(1, -1)
        probs = model.predict_proba(instance)[0]
        if hasattr(model, "classes_") and len(model.classes_) > 2:
            return float(np.max(probs))
        return float(probs[1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        instance = X[idx]
        is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2

        if is_multiclass:
            probs = model.predict_proba(instance.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
            coefs = model.coef_[pred_idx]
        else:
            coefs = model.coef_[0]

        contributions = coefs * instance

        drivers = []
        for feat_idx, (name, val, contrib) in enumerate(zip(feature_names, instance, contributions, strict=False)):
            if self.language == "vi":
                direction = "tích cực" if contrib >= 0 else "tiêu cực"
            else:
                direction = "positive" if contrib >= 0 else "negative"

            if is_multiclass:
                per_class = {
                    str(c): float(model.coef_[c_idx, feat_idx] * val) for c_idx, c in enumerate(model.classes_)
                }
            else:
                per_class = None

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(val),
                    impact=float(abs(contrib)),
                    direction=direction,
                    per_class_impacts=per_class,
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
        immutable_features: list[str] | None = None,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        target_class: Any | None = None,
    ) -> list[Counterfactual]:
        return compute_counterfactuals_logistic(
            model,
            X[idx],
            threshold,
            feature_names,
            immutable_features=immutable_features,
            feature_bounds=feature_bounds,
            target_class=target_class,
        )
