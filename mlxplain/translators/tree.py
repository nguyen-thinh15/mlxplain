"""Tree-based model translator — decision path splits → feature contributions."""

from __future__ import annotations

from typing import Any

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
        probs = model.predict_proba(instance)[0]
        if hasattr(model, "classes_") and len(model.classes_) > 2:
            return float(np.max(probs))
        return float(probs[1])

    def extract_drivers(self, model, X: np.ndarray, idx: int, feature_names: list[str]) -> list[FeatureDriver]:
        instance = X[idx]
        is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2
        n_classes = len(model.classes_) if hasattr(model, "classes_") else 2

        if is_multiclass:
            probs = model.predict_proba(instance.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
        else:
            pred_idx = 1

        # Determine estimators
        estimators = model.estimators_ if hasattr(model, "estimators_") else [model]

        # Accumulate feature contributions across all estimators
        all_contributions: dict[str, list[np.ndarray]] = {}

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

                # Calculate probability difference vector across all classes
                counts_node = values[node_id].flatten()
                total_node = counts_node.sum()
                p_node = counts_node / total_node if total_node > 0 else np.zeros(n_classes)

                counts_child = values[child_node_id].flatten()
                total_child = counts_child.sum()
                p_child = counts_child / total_child if total_child > 0 else np.zeros(n_classes)

                delta_p = p_child - p_node

                if name not in all_contributions:
                    all_contributions[name] = []
                all_contributions[name].append(delta_p)

        # Average the contributions across all estimators
        drivers = []
        for feat_idx, name in enumerate(feature_names):
            contribs = all_contributions.get(name, [])
            # Pad with zeros for trees that did not split on this feature
            n_missing = len(estimators) - len(contribs)
            if n_missing > 0:
                padding = [np.zeros(n_classes) for _ in range(n_missing)]
                contribs = contribs + padding

            mean_contrib_vec = np.mean(contribs, axis=0) if contribs else np.zeros(n_classes)

            # Extract contribution for the predicted class
            pred_contrib = mean_contrib_vec[pred_idx] if is_multiclass else mean_contrib_vec[1]
            if abs(pred_contrib) < 1e-10 and not is_multiclass:
                continue
            if is_multiclass and np.all(np.abs(mean_contrib_vec) < 1e-10):
                continue

            if self.language == "vi":
                direction = "tích cực" if pred_contrib >= 0 else "tiêu cực"
            else:
                direction = "positive" if pred_contrib >= 0 else "negative"

            if is_multiclass:
                per_class = {str(c): float(mean_contrib_vec[c_idx]) for c_idx, c in enumerate(model.classes_)}
            else:
                per_class = None

            drivers.append(
                FeatureDriver(
                    feature=name,
                    value=float(instance[feat_idx]),
                    impact=float(abs(pred_contrib)),
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
