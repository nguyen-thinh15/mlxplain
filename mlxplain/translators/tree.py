"""Tree-based model translator — decision path splits → feature contributions."""

from __future__ import annotations

import numpy as np

from mlxplain.core.counterfactual import compute_counterfactuals_perturbation
from mlxplain.core.report import Counterfactual, FeatureDriver
from mlxplain.translators.base import BaseTranslator


class TreeTranslator(BaseTranslator):
    """Translates decision tree / random forest predictions into feature contributions."""

    def __init__(self, language: str = "en"):
        super().__init__(language)

    def get_probability(self, model, X: np.ndarray, idx: int) -> float:
        instance = X[idx].reshape(1, -1)
        return float(model.predict_proba(instance)[0, 1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        instance = X[idx]

        # Determine estimators
        estimators = model.estimators_ if hasattr(model, "estimators_") else [model]

        # Accumulate feature contributions across all estimators
        all_contributions: dict[str, list[float]] = {}

        for estimator in estimators:
            tree = estimator.tree_
            feature_ids = tree.feature
            thresholds = tree.threshold
            values = tree.value
            children_left = tree.children_left
            children_right = tree.children_right

            decision_path = estimator.decision_path(instance.reshape(1, -1))
            node_indices = decision_path.indices

            for node_id in node_indices:
                feat_id = feature_ids[node_id]
                if feat_id < 0:
                    continue  # Leaf node
                name = feature_names[feat_id]
                feat_val = instance[feat_id]
                thresh = thresholds[node_id]

                # Determine child node
                child_node_id = children_left[node_id] if feat_val <= thresh else children_right[node_id]

                # Calculate probability difference
                counts_node = values[node_id].flatten()
                total_node = counts_node.sum()
                p_node = float(counts_node[1] / total_node) if total_node > 0 else 0.0

                counts_child = values[child_node_id].flatten()
                total_child = counts_child.sum()
                p_child = float(counts_child[1] / total_child) if total_child > 0 else 0.0

                delta_p = p_child - p_node

                if name not in all_contributions:
                    all_contributions[name] = []
                all_contributions[name].append(delta_p)

        # Average the contributions across all estimators
        drivers = []
        for feat_idx, name in enumerate(feature_names):
            contribs = all_contributions.get(name, [])
            # Pad with 0.0 for trees that did not split on this feature
            n_missing = len(estimators) - len(contribs)
            if n_missing > 0:
                contribs = contribs + [0.0] * n_missing

            mean_contrib = float(np.mean(contribs)) if contribs else 0.0
            if abs(mean_contrib) < 1e-10:
                continue

            if self.language == "vi":
                direction = "tích cực" if mean_contrib >= 0 else "tiêu cực"
            else:
                direction = "positive" if mean_contrib >= 0 else "negative"

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(instance[feat_idx]),
                    impact=float(abs(mean_contrib)),
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
