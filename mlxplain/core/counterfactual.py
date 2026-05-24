"""Counterfactual computation: analytical (logistic) and perturbation (tree/ensemble)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mlxplain.core.report import Counterfactual


class ConstraintHelper:
    """Manages immutability and feature bounds checking for counterfactual search."""

    def __init__(
        self,
        feature_names: list[str],
        immutable_features: list[str] | None = None,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.feature_names = feature_names
        self.immutable_features = set(immutable_features) if immutable_features else set()
        self.feature_bounds = feature_bounds or {}

    def is_mutable(self, name: str) -> bool:
        """Check if feature is mutable."""
        return name not in self.immutable_features

    def is_within_bounds(self, name: str, value: float) -> bool:
        """Check if value is within defined bounds."""
        if name in self.feature_bounds:
            low, high = self.feature_bounds[name]
            return low <= value <= high
        return True

    def clamp(self, name: str, value: float) -> float:
        """Clamp value to defined bounds if they exist."""
        if name in self.feature_bounds:
            low, high = self.feature_bounds[name]
            return float(np.clip(value, low, high))
        return float(value)


def compute_counterfactuals_logistic(
    model,
    instance: np.ndarray,
    threshold: float,
    feature_names: list[str],
    immutable_features: list[str] | None = None,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
    target_class: Any | None = None,
) -> list[Counterfactual]:
    """Analytical counterfactuals via coefficient inversion for logistic regression.

    Supports both binary and multi-class classification.
    """
    helper = ConstraintHelper(feature_names, immutable_features, feature_bounds)
    is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2

    counterfactuals = []

    if is_multiclass:
        coefs = model.coef_  # (n_classes, n_features)
        intercepts = model.intercept_  # (n_classes,)
        scores = np.dot(coefs, instance) + intercepts
        pred_idx = int(np.argmax(scores))

        # Resolve target index
        if target_class is not None:
            classes_list = list(model.classes_)
            target_idx = classes_list.index(target_class) if target_class in classes_list else None
        else:
            target_idx = None

        if target_idx is None:
            # Default to runner-up (second highest score)
            sorted_indices = np.argsort(scores)
            target_idx = int(sorted_indices[-2])

        delta_score = scores[pred_idx] - scores[target_idx]

        for i, name in enumerate(feature_names):
            denom = coefs[target_idx, i] - coefs[pred_idx, i]
            if abs(denom) < 1e-10:
                continue
            if not helper.is_mutable(name):
                continue
            change = delta_score / denom
            target_val = instance[i] + change
            if not helper.is_within_bounds(name, target_val):
                continue
            counterfactuals.append(
                Counterfactual(
                    feature=name,
                    current_value=float(instance[i]),
                    target_value=float(target_val),
                    change_needed=float(change),
                )
            )
    else:
        coefs = model.coef_[0]
        intercept = model.intercept_[0]

        # Current log-odds
        current_log_odds = np.dot(coefs, instance) + intercept
        # Target log-odds at threshold boundary
        clamped_threshold = float(np.clip(threshold, 1e-7, 1.0 - 1e-7))
        target_log_odds = np.log(clamped_threshold / (1 - clamped_threshold))

        delta_log_odds = target_log_odds - current_log_odds

        for i, (coef, name) in enumerate(zip(coefs, feature_names, strict=False)):
            if abs(coef) < 1e-10:
                continue
            if not helper.is_mutable(name):
                continue
            change = delta_log_odds / coef
            target_val = instance[i] + change
            if not helper.is_within_bounds(name, target_val):
                continue
            counterfactuals.append(
                Counterfactual(
                    feature=name,
                    current_value=float(instance[i]),
                    target_value=float(target_val),
                    change_needed=float(change),
                )
            )

    # Sort by absolute change needed (smallest first = cheapest flip)
    counterfactuals.sort(key=lambda c: abs(c.change_needed))

    if not counterfactuals:
        warnings.warn(
            "No feasible counterfactual path exists within current feature constraints and bounds.",
            UserWarning,
            stacklevel=2,
        )

    return counterfactuals


def compute_counterfactuals_perturbation(
    model,
    instance: np.ndarray,
    threshold: float,
    feature_names: list[str],
    immutable_features: list[str] | None = None,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
    n_steps: int = 50,
    target_class: Any | None = None,
) -> list[Counterfactual]:
    """Perturbation-based counterfactuals for tree/ensemble models.

    Supports both binary and multi-class classification.
    """
    helper = ConstraintHelper(feature_names, immutable_features, feature_bounds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2

        if is_multiclass:
            original_probs = model.predict_proba(instance.reshape(1, -1))[0]
            original_pred_idx = np.argmax(original_probs)
            original_pred = model.classes_[original_pred_idx]
        else:
            current_prob = model.predict_proba(instance.reshape(1, -1))[0, 1]
            above_threshold = current_prob >= threshold

        counterfactuals = []
        for i, name in enumerate(feature_names):
            if not helper.is_mutable(name):
                continue
            original = instance[i]
            # Search in both directions
            best_change = None
            for direction in [1, -1]:
                # Search range: up to 3x the feature's absolute value (or 1 if zero)
                search_range = max(abs(original) * 3, 1.0)
                for step in range(1, n_steps + 1):
                    delta = direction * search_range * step / n_steps
                    perturbed_val = original + delta
                    if not helper.is_within_bounds(name, perturbed_val):
                        continue
                    perturbed = instance.copy()
                    perturbed[i] = perturbed_val

                    if is_multiclass:
                        new_probs = model.predict_proba(perturbed.reshape(1, -1))[0]
                        new_pred_idx = np.argmax(new_probs)
                        new_pred = model.classes_[new_pred_idx]

                        flipped = new_pred == target_class if target_class is not None else new_pred != original_pred
                    else:
                        new_prob = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
                        flipped = (new_prob >= threshold) != above_threshold

                    if flipped:
                        if best_change is None or abs(delta) < abs(best_change):
                            best_change = delta
                        break  # Found flip in this direction, try the other

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

    if not counterfactuals:
        warnings.warn(
            "No feasible counterfactual path exists within current feature constraints and bounds.",
            UserWarning,
            stacklevel=2,
        )

    return counterfactuals
