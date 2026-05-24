"""Clustering translator — explains KMeans cluster assignments."""

from __future__ import annotations

import numpy as np

from mlxplain.core.report import Counterfactual, FeatureDriver


class KMeansTranslator:
    """Translates KMeans cluster assignments."""

    def __init__(self, language: str = "en"):
        self.language = language

    def extract_drivers(
        self,
        instance: np.ndarray,
        assigned_centroid: np.ndarray,
        target_centroid: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureDriver]:
        """Explain why the instance belongs to the assigned cluster versus the target centroid.

        The impact of feature i is: (x_i - t_i)^2 - (x_i - c_i)^2.
        If impact > 0, this feature pushed the instance closer to the assigned centroid c (positive driver).
        If impact < 0, this feature pushed it closer to the target centroid t (negative driver).
        """
        drivers = []
        for i, (name, val) in enumerate(zip(feature_names, instance, strict=False)):
            c_val = assigned_centroid[i]
            t_val = target_centroid[i]
            impact = float((val - t_val) ** 2 - (val - c_val) ** 2)

            if self.language == "vi":
                direction = "tích cực" if impact >= 0 else "tiêu cực"
            else:
                direction = "positive" if impact >= 0 else "negative"

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(val),
                    impact=float(abs(impact)),
                    direction=direction,
                )
            )

        # Sort by impact magnitude
        drivers.sort(key=lambda d: d.impact, reverse=True)
        return drivers

    def compute_counterfactuals(
        self,
        instance: np.ndarray,
        assigned_centroid: np.ndarray,
        target_centroid: np.ndarray,
        feature_names: list[str],
    ) -> list[Counterfactual]:
        """Compute the minimum feature changes to transition the instance to the target cluster.

        Uses exact, closed-form projection onto the bisecting decision boundary hyperplane.
        """
        w = 2.0 * (assigned_centroid - target_centroid)
        dist_c = float(np.sum((instance - assigned_centroid) ** 2))
        dist_t = float(np.sum((instance - target_centroid) ** 2))
        R = dist_c - dist_t  # Negative because x is closer to c than to t

        # Add a tiny epsilon margin to guarantee crossing the decision boundary
        epsilon = 1e-5
        norm_w_sq = float(np.sum(w**2))
        if norm_w_sq < 1e-12:
            return []

        delta = ((R - epsilon) / norm_w_sq) * w

        counterfactuals = []
        for name, val, change in zip(feature_names, instance, delta, strict=False):
            counterfactuals.append(
                Counterfactual(
                    feature=name,
                    current_value=float(val),
                    target_value=float(val + change),
                    change_needed=float(change),
                )
            )

        # Sort by absolute change needed (smallest/cheapest changes first)
        counterfactuals.sort(key=lambda c: abs(c.change_needed))
        return counterfactuals
