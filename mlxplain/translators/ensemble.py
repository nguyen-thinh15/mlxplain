"""Ensemble boosting translator — SHAP values → feature importances."""

from __future__ import annotations

import warnings

import numpy as np
import shap

from mlxplain.core.counterfactual import compute_counterfactuals_perturbation
from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


class EnsembleTranslator(BaseTranslator):
    """Translates XGBoost / LightGBM predictions via SHAP values."""

    def __init__(self, language: str = "en"):
        super().__init__(language)

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        instance = X[idx].reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(model.predict_proba(instance)[0, 1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        instance = X[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance.reshape(1, -1))
            except Exception:
                # Fallback to model-agnostic explainer
                # To keep it extremely fast, use a small sample (max 100 rows) of X as background data
                bg_data = X
                if len(bg_data) > 100:
                    indices = np.linspace(0, len(bg_data) - 1, 100, dtype=int)
                    bg_data = bg_data[indices]

                def predict_fn(x):
                    return model.predict_proba(x)[:, 1]

                explainer = shap.Explainer(predict_fn, bg_data)
                shap_explanation = explainer(instance.reshape(1, -1))
                shap_values = shap_explanation.values

        # Convert Explanation object to raw values if returned
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        # Parse shap_values robustly depending on output format
        if isinstance(shap_values, list):
            # Select class 1 if multiple classes, else class 0
            raw_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            raw_values = shap_values

        # Extract 1D array of contributions for this instance
        if len(raw_values.shape) == 3:
            # Shape: (n_samples, n_features, n_classes)
            values = raw_values[0, :, 1] if raw_values.shape[2] > 1 else raw_values[0, :, 0]
        elif len(raw_values.shape) == 2:
            # Shape: (n_samples, n_features)
            values = raw_values[0]
        else:
            values = raw_values

        drivers = []
        for name, val, sv in zip(feature_names, instance, values, strict=False):
            if self.language == "vi":
                direction = "tích cực" if sv >= 0 else "tiêu cực"
            else:
                direction = "positive" if sv >= 0 else "negative"

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(val),
                    impact=float(abs(sv)),
                    direction=direction,
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
        return compute_counterfactuals_perturbation(model, X[idx], threshold, feature_names)
