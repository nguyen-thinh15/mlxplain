"""End-to-end tests for LogisticRegression explanation."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlxplain import explain
from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver


@pytest.fixture
def logistic_model():
    """Train a simple logistic regression on synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, ["income", "tenure", "debt"]


def test_explain_returns_report(logistic_model):
    model, X, names = logistic_model
    report = explain(model, X, idx=0, feature_names=names)
    assert isinstance(report, ExplanationReport)


def test_probability_in_range(logistic_model):
    model, X, names = logistic_model
    report = explain(model, X, idx=0, feature_names=names)
    assert 0.0 <= report.probability <= 1.0


def test_drivers_are_present(logistic_model):
    model, X, names = logistic_model
    report = explain(model, X, idx=0, feature_names=names)
    all_drivers = report.positive_drivers + report.negative_drivers
    assert len(all_drivers) > 0
    assert all(isinstance(d, FeatureDriver) for d in all_drivers)


def test_drivers_sorted_by_impact(logistic_model):
    model, X, names = logistic_model
    report = explain(model, X, idx=0, feature_names=names)
    for drivers in [report.positive_drivers, report.negative_drivers]:
        impacts = [d.impact for d in drivers]
        assert impacts == sorted(impacts, reverse=True)


def test_counterfactuals_are_present(logistic_model):
    model, X, names = logistic_model
    # Use idx=1 which is above threshold (prob≈0.90) — counterfactuals are only
    # computed for unfavorable predictions (probability >= threshold)
    report = explain(model, X, idx=1, feature_names=names)
    assert report.probability >= 0.5
    assert len(report.counterfactuals) > 0
    assert all(isinstance(c, Counterfactual) for c in report.counterfactuals)


def test_figures_generated(logistic_model):
    model, X, names = logistic_model
    report = explain(model, X, idx=0, feature_names=names)
    assert "gauge" in report.figures
    assert "drivers" in report.figures
    assert "counterfactuals" in report.figures
