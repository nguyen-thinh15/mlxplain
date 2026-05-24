"""Ensemble boosting translator — SHAP values → feature importances."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mlxplain.core.counterfactual import compute_counterfactuals_perturbation
from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


def _canonicalize_shap_values(shap_values, n_features: int, n_classes: int) -> np.ndarray:
    """Shape-normalize all SHAP variants into a canonical 2D matrix (n_features, n_classes) for the first instance."""
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # Case 1: list of arrays
    if isinstance(shap_values, list):
        canonical = np.zeros((n_features, n_classes))
        for c_idx in range(min(n_classes, len(shap_values))):
            arr = shap_values[c_idx]
            if len(arr.shape) == 2:
                canonical[:, c_idx] = arr[0]
            elif len(arr.shape) == 1:
                canonical[:, c_idx] = arr
        return canonical

    # Case 2: 3D array of shape (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        return shap_values[0]

    # Case 3: 2D array of shape (n_samples, n_features)
    if len(shap_values.shape) == 2:
        canonical = np.zeros((n_features, n_classes))
        if n_classes == 2:
            canonical[:, 1] = shap_values[0]
            canonical[:, 0] = -shap_values[0]
        else:
            canonical[:, 0] = shap_values[0]
        return canonical

    # Case 4: 1D array of shape (n_features,)
    canonical = np.zeros((n_features, n_classes))
    if n_classes == 2:
        canonical[:, 1] = shap_values
        canonical[:, 0] = -shap_values
    else:
        canonical[:, 0] = shap_values
    return canonical


class EnsembleTranslator(BaseTranslator):
    """Translates XGBoost / LightGBM predictions via SHAP values."""

    def __init__(self, language: str = "en"):
        try:
            import shap  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "EnsembleTranslator requires the optional 'shap' backend. "
                "Please install it using: pip install mlxplain-xai[shap]"
            ) from e
        super().__init__(language)

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        instance = X[idx].reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            probs = model.predict_proba(instance)[0]
            if hasattr(model, "classes_") and len(model.classes_) > 2:
                return float(np.max(probs))
            return float(probs[1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        import shap

        instance = X[idx]
        n_features = len(feature_names)
        is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2
        n_classes = len(model.classes_) if hasattr(model, "classes_") else 2

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
                    if is_multiclass:
                        # For multiclass, predict_proba returns (n_samples, n_classes)
                        return model.predict_proba(x)
                    return model.predict_proba(x)[:, 1]

                explainer = shap.Explainer(predict_fn, bg_data)
                shap_explanation = explainer(instance.reshape(1, -1))
                shap_values = shap_explanation.values

        canonical_shap = _canonicalize_shap_values(shap_values, n_features, n_classes)

        if is_multiclass:
            probs = model.predict_proba(instance.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
        else:
            pred_idx = 1

        values = canonical_shap[:, pred_idx]

        drivers = []
        for feat_idx, (name, val, sv) in enumerate(zip(feature_names, instance, values, strict=False)):
            if self.language == "vi":
                direction = "tích cực" if sv >= 0 else "tiêu cực"
            else:
                direction = "positive" if sv >= 0 else "negative"

            if is_multiclass:
                per_class = {str(c): float(canonical_shap[feat_idx, c_idx]) for c_idx, c in enumerate(model.classes_)}
            else:
                per_class = None

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(val),
                    impact=float(abs(sv)),
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
        return compute_counterfactuals_perturbation(
            model,
            X[idx],
            threshold,
            feature_names,
            immutable_features=immutable_features,
            feature_bounds=feature_bounds,
            target_class=target_class,
        )
