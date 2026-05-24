"""Unit and integration tests for multi-class classification support."""

import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlxplain import explain
from mlxplain.core.report import ExplanationReport

has_shap = importlib.util.find_spec("shap") is not None and importlib.util.find_spec("xgboost") is not None

if has_shap:
    import xgboost as xgb


@pytest.fixture
def multiclass_data():
    """Generate synthetic 3-class data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    score = X[:, 0] * 1.5 + X[:, 1] * 1.0 - X[:, 2] * 2.0
    y = np.zeros(100, dtype=object)
    y[score > 0.8] = "A"
    y[(score >= -0.8) & (score <= 0.8)] = "B"
    y[score < -0.8] = "C"
    return X, y.astype(str), ["income", "tenure", "debt"]


def test_multiclass_logistic_explanation(multiclass_data):
    """Test multi-class explanation with Logistic Regression."""
    X, y, names = multiclass_data
    model = LogisticRegression(random_state=42).fit(X, y)

    # Explain a Grade B applicant, target Grade A
    report = explain(model, X, idx=0, feature_names=names, target_class="A")

    assert isinstance(report, ExplanationReport)
    assert report.probabilities is not None
    assert set(report.probabilities.keys()) == {"A", "B", "C"}
    assert report.prediction in {"A", "B", "C"}
    assert abs(report.probability - report.probabilities[report.prediction]) < 1e-6

    # Verify drivers
    for d in report.positive_drivers + report.negative_drivers:
        assert d.per_class_impacts is not None
        assert set(d.per_class_impacts.keys()) == {"A", "B", "C"}

    # Verify counterfactuals
    if report.prediction != "A":
        assert len(report.counterfactuals) > 0
        for c in report.counterfactuals:
            assert c.feature in names
            assert c.change_needed != 0.0


def test_multiclass_random_forest_explanation(multiclass_data):
    """Test multi-class explanation with Random Forest."""
    X, y, names = multiclass_data
    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y)

    report = explain(model, X, idx=0, feature_names=names, target_class="A")

    assert isinstance(report, ExplanationReport)
    assert report.probabilities is not None
    assert set(report.probabilities.keys()) == {"A", "B", "C"}

    # Verify drivers
    for d in report.positive_drivers + report.negative_drivers:
        assert d.per_class_impacts is not None
        assert set(d.per_class_impacts.keys()) == {"A", "B", "C"}


@pytest.mark.skipif(not has_shap, reason="shap is required for multi-class ensemble explanation")
def test_multiclass_ensemble_explanation(multiclass_data):
    """Test multi-class explanation with XGBoost."""
    X, y, names = multiclass_data

    # Map string labels 'A', 'B', 'C' to integers 0, 1, 2 for XGBoost
    y_encoded = np.zeros(len(y), dtype=int)
    y_encoded[y == "A"] = 0
    y_encoded[y == "B"] = 1
    y_encoded[y == "C"] = 2

    model = xgb.XGBClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y_encoded)

    report = explain(model, X, idx=0, feature_names=names, target_class=0)

    assert isinstance(report, ExplanationReport)
    assert report.probabilities is not None
    assert set(report.probabilities.keys()) == {"0", "1", "2"}


def test_multiclass_visual_charts_generation(multiclass_data):
    """Test that multi-class reports generate correct visualizations (heatmap & probabilities)."""
    X, y, names = multiclass_data
    model = LogisticRegression(random_state=42).fit(X, y)

    report = explain(model, X, idx=0, feature_names=names)

    assert "probabilities" in report.figures
    assert "heatmap" in report.figures
    assert "drivers" in report.figures
    assert "counterfactuals" in report.figures
    assert "gauge" not in report.figures
