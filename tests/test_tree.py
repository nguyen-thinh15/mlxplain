"""End-to-end tests for DecisionTree and RandomForest explanations."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlxplain import explain
from mlxplain.core.report import ExplanationReport, FeatureDriver


@pytest.fixture
def tree_model():
    """Train a decision tree on synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    model.fit(X, y)
    return model, X, ["age", "balance", "num_products"]


def test_tree_explain_returns_report(tree_model):
    model, X, names = tree_model
    report = explain(model, X, idx=0, feature_names=names)
    assert isinstance(report, ExplanationReport)


def test_tree_probability_in_range(tree_model):
    model, X, names = tree_model
    report = explain(model, X, idx=0, feature_names=names)
    assert 0.0 <= report.probability <= 1.0


def test_tree_drivers_present(tree_model):
    model, X, names = tree_model
    report = explain(model, X, idx=0, feature_names=names)
    all_drivers = report.positive_drivers + report.negative_drivers
    assert len(all_drivers) > 0
    assert all(isinstance(d, FeatureDriver) for d in all_drivers)


def test_tree_counterfactuals(tree_model):
    model, X, names = tree_model
    report = explain(model, X, idx=0, feature_names=names)
    # Counterfactuals should exist (perturbation-based)
    assert isinstance(report.counterfactuals, list)


def test_tree_figures(tree_model):
    model, X, names = tree_model
    report = explain(model, X, idx=0, feature_names=names)
    assert "gauge" in report.figures
    assert "drivers" in report.figures


def test_random_forest_explain_returns_report():
    """Verify that explain() works for RandomForestClassifier without crashes."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(random_state=42, n_estimators=3, max_depth=3)
    model.fit(X, y)

    report = explain(model, X, idx=0, feature_names=["income", "tenure", "debt"])
    assert isinstance(report, ExplanationReport)
    assert 0.0 <= report.probability <= 1.0
    all_drivers = report.positive_drivers + report.negative_drivers
    assert len(all_drivers) > 0
    assert all(isinstance(d, FeatureDriver) for d in all_drivers)
    assert "gauge" in report.figures
    assert "drivers" in report.figures
