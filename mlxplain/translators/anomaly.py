"""Anomaly detection translator — explains IsolationForest models."""

from __future__ import annotations

import warnings

import numpy as np
import shap

from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


class IsolationForestTranslator(BaseTranslator):
    """Translates IsolationForest predictions via SHAP values."""

    def __init__(self, language: str = "en"):
        super().__init__(language)

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        """Return anomaly score (higher means more anomalous) in range [0, 1]."""
        instance = X[idx].reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # scikit-learn score_samples returns negative anomaly score.
            # -score_samples(x) matches the original paper's score in [0, 1]
            return float(-model.score_samples(instance)[0])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        """Extract per-feature contributions to the anomaly score using SHAP."""
        instance = X[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance.reshape(1, -1))
                # TreeExplainer explains path length (shorter = more anomalous).
                # We multiply by -1.0 to align with the anomaly score (higher score = more anomalous).
                if hasattr(shap_values, "values"):
                    shap_values = -shap_values.values
                elif isinstance(shap_values, list):
                    shap_values = [-sv for sv in shap_values]
                else:
                    shap_values = -shap_values
            except Exception:
                # Fallback to model-agnostic explainer
                bg_data = X
                if len(bg_data) > 100:
                    indices = np.linspace(0, len(bg_data) - 1, 100, dtype=int)
                    bg_data = bg_data[indices]

                def score_fn(x):
                    return -model.score_samples(x)

                explainer = shap.Explainer(score_fn, bg_data)
                shap_explanation = explainer(instance.reshape(1, -1))
                shap_values = shap_explanation.values

        # Convert Explanation object to raw values if returned
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        # Parse shap_values robustly
        raw_values = shap_values[0] if isinstance(shap_values, list) else shap_values

        values = raw_values[0] if len(raw_values.shape) == 2 else raw_values

        # In IsolationForest, lower path length = higher anomaly score.
        # TreeExplainer explains decision_path or average tree path lengths.
        # If the explainer output is negatively correlated with the anomaly score,
        # we flip the sign of the SHAP values to align with "higher score = positive driver".
        # We can dynamically detect this correlation or default to standard SHAP values.
        # Generally, TreeExplainer on IsolationForest output is already aligned such that
        # features pushing toward shorter paths (anomalies) have positive impact or we can
        # align them by checking the score change. Let's make it standard.
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
        """Perturbation-based counterfactual search to make the instance normal."""
        instance = X[idx]
        current_score = self.get_probability(model, X, idx)
        above_threshold = current_score >= threshold

        counterfactuals = []
        n_steps = 50

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for i, name in enumerate(feature_names):
                original = instance[i]
                best_change = None
                for direction in [1, -1]:
                    search_range = max(abs(original) * 3, 1.0)
                    for step in range(1, n_steps + 1):
                        delta = direction * search_range * step / n_steps
                        perturbed = instance.copy()
                        perturbed[i] = original + delta
                        new_score = float(-model.score_samples(perturbed.reshape(1, -1))[0])
                        flipped = (new_score >= threshold) != above_threshold
                        if flipped:
                            if best_change is None or abs(delta) < abs(best_change):
                                best_change = delta
                            break

                if best_change is not None:
                    counterfactuals.append(
                        Counterfactual(
                            feature=name,
                            current_value=float(original),
                            target_value=float(original + best_change),
                            change_needed=float(best_change),
                        )
                    )

        counterfactuals.sort(key=lambda c: abs(c.change_needed))
        return counterfactuals
