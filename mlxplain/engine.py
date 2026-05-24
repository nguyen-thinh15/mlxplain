"""Orchestrator: detect model → select translator → explain → visualize → report."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain.core.report import ExplanationReport
from mlxplain.core.threshold import classify
from mlxplain.domains.credit_risk.interpreter import CreditRiskDomain
from mlxplain.translators.base import BaseTranslator
from mlxplain.translators.ensemble import EnsembleTranslator
from mlxplain.translators.logistic import LogisticTranslator
from mlxplain.translators.tree import TreeTranslator
from mlxplain.visualizations.charts import plot_report


def _detect_translator(model, language: str = "en") -> BaseTranslator:
    """Auto-detect the model type and return the appropriate translator."""
    # Logistic regression (covers LogisticRegression and subclasses like LogisticRegressionCV)
    if isinstance(model, LogisticRegression):
        return LogisticTranslator(language=language)

    # Ensemble boosting (XGBoost, LightGBM) — check before tree since
    # XGBClassifier has estimators_ too
    try:
        import xgboost

        if isinstance(model, xgboost.XGBClassifier):
            return EnsembleTranslator(language=language)
    except ImportError:
        pass

    try:
        import lightgbm

        if isinstance(model, lightgbm.LGBMClassifier):
            return EnsembleTranslator(language=language)
    except ImportError:
        pass

    # Tree-based (DecisionTree, RandomForest)
    if isinstance(model, DecisionTreeClassifier | RandomForestClassifier):
        return TreeTranslator(language=language)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        "Supported: LogisticRegression, DecisionTreeClassifier, "
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
) -> ExplanationReport:
    """Generic domain-agnostic explanation for any supported binary classifier.

    Args:
        model: A fitted scikit-learn-compatible binary classifier.
        X: Feature matrix (2d array).
        idx: Index of the instance to explain.
        feature_names: Names for each feature column. Defaults to f0, f1, ...
        threshold: Decision boundary probability.
        positive_label: Label for probability >= threshold.
        negative_label: Label for probability < threshold.
        top_k: If set, keep only the top-k drivers per direction (by impact).
        language: Language for the output driver directions ("en" or "vi").

    Returns:
        ExplanationReport with prediction, drivers, counterfactuals, and charts.
    """
    X = np.asarray(X, dtype=float)
    _validate_inputs(X, idx, feature_names)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    if language == "vi":
        if positive_label == "Positive":
            positive_label = "Tích cực"
        if negative_label == "Negative":
            negative_label = "Tiêu cực"

    translator = _detect_translator(model, language=language)

    # 1. Probability
    probability = translator.get_probability(model, X, idx)

    # 2. Classification
    prediction = classify(probability, threshold, positive_label, negative_label)

    # 3. Feature drivers
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

    # 4. Counterfactuals — only compute for unfavorable predictions
    if probability >= threshold:
        counterfactuals = translator.compute_counterfactuals(model, X, idx, threshold, feature_names)
    else:
        counterfactuals = []

    # 5. Build report
    report = ExplanationReport(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        counterfactuals=counterfactuals,
    )

    # 6. Visualizations
    report.domain_output = {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "language": language,
    }
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
    )
    domain = CreditRiskDomain()
    return domain.interpret(report)
