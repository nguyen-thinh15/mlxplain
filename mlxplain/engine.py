"""Orchestrator: detect model → select translator → explain → visualize → report."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain.core.report import ExplanationReport
from mlxplain.core.threshold import classify
from mlxplain.domains.credit_risk.interpreter import CreditRiskDomain
from mlxplain.translators.base import BaseTranslator
from mlxplain.translators.logistic import LogisticTranslator
from mlxplain.translators.tree import TreeTranslator
from mlxplain.visualizations.charts import plot_report


def _detect_translator(model, language: str = "en") -> BaseTranslator:
    """Auto-detect the model type and return the appropriate translator."""
    # Isolation Forest (unsupervised anomaly detection)
    if isinstance(model, IsolationForest):
        from mlxplain.translators.anomaly import IsolationForestTranslator

        return IsolationForestTranslator(language=language)

    # Logistic regression (covers LogisticRegression and subclasses like LogisticRegressionCV)
    if isinstance(model, LogisticRegression):
        return LogisticTranslator(language=language)

    # Ensemble boosting (XGBoost, LightGBM) — check before tree since
    # XGBClassifier has estimators_ too
    try:
        import xgboost

        if isinstance(model, xgboost.XGBClassifier):
            from mlxplain.translators.ensemble import EnsembleTranslator

            return EnsembleTranslator(language=language)
    except ImportError:
        pass

    try:
        import lightgbm

        if isinstance(model, lightgbm.LGBMClassifier):
            from mlxplain.translators.ensemble import EnsembleTranslator

            return EnsembleTranslator(language=language)
    except ImportError:
        pass

    # Tree-based (DecisionTree, RandomForest)
    if isinstance(model, DecisionTreeClassifier | RandomForestClassifier):
        return TreeTranslator(language=language)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        "Supported: IsolationForest, LogisticRegression, DecisionTreeClassifier, "
        "RandomForestClassifier, XGBClassifier, LGBMClassifier."
    )


def _validate_inputs(
    X: np.ndarray,
    idx: int,
    feature_names: list[str] | None,
) -> None:
    """Validate inputs before translation."""
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got {X.ndim}D.")
    if idx < 0 or idx >= X.shape[0]:
        raise ValueError(f"idx={idx} is out of bounds for X with {X.shape[0]} rows.")
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_names has {len(feature_names)} entries but X has {X.shape[1]} columns.")


def explain(
    model,
    X: np.ndarray,
    idx: int = 0,
    feature_names: list[str] | None = None,
    threshold: float = 0.5,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
    top_k: int | None = None,
    language: str = "en",
    immutable_features: list[str] | None = None,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
    target_class: Any | None = None,
) -> ExplanationReport:
    """Generic domain-agnostic explanation for any supported binary or multi-class classifier.

    Args:
        model: A fitted scikit-learn-compatible classifier.
        X: Feature matrix (2d array).
        idx: Index of the instance to explain.
        feature_names: Names for each feature column. Defaults to f0, f1, ...
        threshold: Decision boundary probability. For binary, predictions with probability >= threshold
            receive `positive_label`; below receive `negative_label`. For multi-class, ignored (warns if not 0.5).
        positive_label: Label for probability >= threshold.
        negative_label: Label for probability < threshold.
        top_k: If set, keep only the top-k drivers per direction (by impact). Sorted by predicted class impact.
        language: Language for the output driver directions ("en" or "vi").
        immutable_features: List of feature names that cannot be changed.
        feature_bounds: Dictionary mapping feature name to (lower_bound, upper_bound) tuple.
        target_class: For multi-class, the target class to transition to in counterfactuals.
                      If None, defaults to the runner-up class.

    Returns:
        ExplanationReport with prediction, drivers, counterfactuals, and charts.
    """
    X = np.asarray(X, dtype=float)
    _validate_inputs(X, idx, feature_names)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    is_multiclass = hasattr(model, "classes_") and len(model.classes_) > 2

    if is_multiclass and threshold != 0.5:
        import warnings

        warnings.warn(
            "Threshold is not applicable to multi-class classification and will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    if language == "vi":
        if positive_label == "Positive":
            positive_label = "Bất thường" if isinstance(model, IsolationForest) else "Tích cực"
        if negative_label == "Negative":
            negative_label = "Bình thường" if isinstance(model, IsolationForest) else "Tiêu cực"
    else:
        if positive_label == "Positive" and isinstance(model, IsolationForest):
            positive_label = "Anomaly"
        if negative_label == "Negative" and isinstance(model, IsolationForest):
            negative_label = "Normal"

    translator = _detect_translator(model, language=language)

    # 1. Probability & Classification
    if is_multiclass:
        probs_dict = translator.get_probabilities(model, X, idx)
        prediction = max(probs_dict, key=probs_dict.get)
        probability = probs_dict[prediction]
    else:
        probs_dict = None
        probability = translator.get_probability(model, X, idx)
        prediction = classify(probability, threshold, positive_label, negative_label)

    # 2. Feature drivers
    all_drivers = translator.extract_drivers(model, X, idx, feature_names)
    positive_drivers = sorted(
        [d for d in all_drivers if d.direction in ("positive", "tích cực")],
        key=lambda d: d.impact,
        reverse=True,
    )
    negative_drivers = sorted(
        [d for d in all_drivers if d.direction in ("negative", "tiêu cực")],
        key=lambda d: d.impact,
        reverse=True,
    )

    if top_k is not None:
        positive_drivers = positive_drivers[:top_k]
        negative_drivers = negative_drivers[:top_k]

    # 3. Counterfactuals — only compute for unfavorable predictions/states
    if is_multiclass:
        if target_class is None:
            # Find runner-up (second highest probability class)
            sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1])
            target_class = sorted_probs[-2][0]

        if str(prediction) != str(target_class):
            counterfactuals = translator.compute_counterfactuals(
                model,
                X,
                idx,
                threshold,
                feature_names,
                immutable_features=immutable_features,
                feature_bounds=feature_bounds,
                target_class=target_class,
            )
        else:
            counterfactuals = []
    else:
        if probability >= threshold:
            counterfactuals = translator.compute_counterfactuals(
                model,
                X,
                idx,
                threshold,
                feature_names,
                immutable_features=immutable_features,
                feature_bounds=feature_bounds,
            )
        else:
            counterfactuals = []

    # 4. Build report
    report = ExplanationReport(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        counterfactuals=counterfactuals,
        probabilities=probs_dict,
    )

    # 5. Visualizations
    report.domain_output = {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "language": language,
    }
    if is_multiclass:
        report.domain_output["classes"] = [str(c) for c in model.classes_]
        if target_class is not None:
            report.domain_output["target_class"] = str(target_class)

    report.figures = plot_report(report)

    return report


def explain_risk(
    model,
    X: np.ndarray,
    idx: int = 0,
    feature_names: list[str] | None = None,
    threshold: float = 0.5,
    top_k: int | None = None,
    language: str = "en",
    immutable_features: list[str] | None = None,
    feature_bounds: dict[str, tuple[float, float]] | None = None,
    target_class: Any | None = None,
) -> ExplanationReport:
    """Credit risk convenience wrapper.

    Calls explain() with generic labels, then applies CreditRiskDomain
    interpretation to produce credit-specific output.
    """
    report = explain(
        model,
        X,
        idx=idx,
        feature_names=feature_names,
        threshold=threshold,
        top_k=top_k,
        language=language,
        immutable_features=immutable_features,
        feature_bounds=feature_bounds,
        target_class=target_class,
    )
    domain = CreditRiskDomain()
    if hasattr(model, "classes_") and len(model.classes_) > 2:
        return domain.interpret_multiclass(report)
    return domain.interpret(report)


def explain_cluster(
    model,
    X: np.ndarray,
    idx: int = 0,
    feature_names: list[str] | None = None,
    target_cluster: int | None = None,
    top_k: int | None = None,
    language: str = "en",
) -> ExplanationReport:
    """Explain a KMeans clustering assignment for a data instance.

    Args:
        model: A fitted KMeans clustering model.
        X: Feature matrix (2d array).
        idx: Index of the instance to explain.
        feature_names: Names for each feature column. Defaults to f0, f1, ...
        target_cluster: Index of the target cluster to transition to.
                        If None, defaults to the runner-up (second closest) cluster.
        top_k: If set, keep only the top-k drivers.
        language: Language for outputs ("en" or "vi").

    Returns:
        ExplanationReport detailing cluster drivers, closed-form L2 projection counterfactuals, and charts.
    """
    X = np.asarray(X, dtype=float)
    _validate_inputs(X, idx, feature_names)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    from sklearn.cluster import KMeans

    if not isinstance(model, KMeans):
        raise ValueError(f"Model must be an instance of KMeans, got {type(model).__name__}.")

    instance = X[idx]
    assigned_cluster = int(model.predict(instance.reshape(1, -1))[0])

    # Calculate distances to all centroids
    centroids = model.cluster_centers_
    distances = [float(np.sum((instance - c) ** 2)) for c in centroids]

    # Find runner-up cluster if target not specified
    sorted_indices = np.argsort(distances)
    if target_cluster is None:
        target_cluster = int(sorted_indices[1]) if len(centroids) > 1 else assigned_cluster

    if target_cluster < 0 or target_cluster >= len(centroids):
        raise ValueError(f"target_cluster={target_cluster} is out of bounds for KMeans with {len(centroids)} clusters.")

    # Compute distance-based similarity score (pseudo-probability)
    dist_c = distances[assigned_cluster]
    dist_t = distances[target_cluster]

    probability = 0.5 if dist_c + dist_t < 1e-10 else dist_t / (dist_c + dist_t)

    from mlxplain.translators.clustering import KMeansTranslator

    translator = KMeansTranslator(language=language)

    assigned_centroid = centroids[assigned_cluster]
    target_centroid = centroids[target_cluster]

    all_drivers = translator.extract_drivers(instance, assigned_centroid, target_centroid, feature_names)

    positive_drivers = sorted(
        [d for d in all_drivers if d.direction in ("positive", "tích cực")],
        key=lambda d: d.impact,
        reverse=True,
    )
    negative_drivers = sorted(
        [d for d in all_drivers if d.direction in ("negative", "tiêu cực")],
        key=lambda d: d.impact,
        reverse=True,
    )

    if top_k is not None:
        positive_drivers = positive_drivers[:top_k]
        negative_drivers = negative_drivers[:top_k]

    counterfactuals = translator.compute_counterfactuals(instance, assigned_centroid, target_centroid, feature_names)

    # Format prediction labels
    if language == "vi":
        prediction = f"Nhóm {assigned_cluster}"
        pos_lbl = f"Nhóm {assigned_cluster}"
        neg_lbl = f"Nhóm {target_cluster}"
    else:
        prediction = f"Cluster {assigned_cluster}"
        pos_lbl = f"Cluster {assigned_cluster}"
        neg_lbl = f"Cluster {target_cluster}"

    report = ExplanationReport(
        prediction=prediction,
        probability=probability,
        threshold=0.5,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        counterfactuals=counterfactuals,
    )

    report.domain_output = {
        "positive_label": pos_lbl,
        "negative_label": neg_lbl,
        "language": language,
        "assigned_cluster": assigned_cluster,
        "target_cluster": target_cluster,
    }

    report.figures = plot_report(report)
    return report
