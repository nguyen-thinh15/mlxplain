"""Counterfactual computation: analytical (logistic) and perturbation (tree/ensemble)."""

from __future__ import annotations

import warnings

import numpy as np

from mlxplain.core.report import Counterfactual


def compute_counterfactuals_logistic(
    model,
    instance: np.ndarray,
    threshold: float,
    feature_names: list[str],
) -> list[Counterfactual]:
    """Analytical counterfactuals via coefficient inversion for logistic regression.

    For each feature, compute the minimum single-feature change needed to
    move the predicted probability from the current side to the other side
    of the threshold.
    """
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Current log-odds
    current_log_odds = np.dot(coefs, instance) + intercept
    # Target log-odds at threshold boundary (clamped to prevent division-by-zero/log-odds errors)
    clamped_threshold = float(np.clip(threshold, 1e-7, 1.0 - 1e-7))
    target_log_odds = np.log(clamped_threshold / (1 - clamped_threshold))

    delta_log_odds = target_log_odds - current_log_odds

    counterfactuals = []
    for i, (coef, name) in enumerate(zip(coefs, feature_names)):
        if abs(coef) < 1e-10:
            continue
        change = delta_log_odds / coef
        target_val = instance[i] + change
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
    return counterfactuals


def compute_counterfactuals_perturbation(
    model,
    instance: np.ndarray,
    threshold: float,
    feature_names: list[str],
    n_steps: int = 50,
) -> list[Counterfactual]:
    """Perturbation-based counterfactuals for tree/ensemble models.

    For each feature, searches linearly from the current value to find the
    smallest change that flips the prediction across the threshold.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        current_prob = model.predict_proba(instance.reshape(1, -1))[0, 1]
        above_threshold = current_prob >= threshold

        counterfactuals = []
        for i, name in enumerate(feature_names):
            original = instance[i]
            # Search in both directions
            best_change = None
            for direction in [1, -1]:
                # Search range: up to 3x the feature's absolute value (or 1 if zero)
                search_range = max(abs(original) * 3, 1.0)
                for step in range(1, n_steps + 1):
                    delta = direction * search_range * step / n_steps
                    perturbed = instance.copy()
                    perturbed[i] = original + delta
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
    return counterfactuals
