"""Tests for feature bounds and immutability constraints on counterfactual search."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlxplain import explain
from mlxplain.core.counterfactual import ConstraintHelper


@pytest.fixture
def logistic_model():
    """Train a simple logistic regression on synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    # Target label: pushes positive class above 0
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, ["income", "tenure", "debt"]


@pytest.fixture
def forest_model():
    """Train a random forest on synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model, X, ["income", "tenure", "debt"]


def test_constraint_helper():
    """Verify basic methods on the ConstraintHelper utility."""
    helper = ConstraintHelper(
        feature_names=["a", "b", "c"],
        immutable_features=["a"],
        feature_bounds={"b": (0.0, 5.0)},
    )
    assert not helper.is_mutable("a")
    assert helper.is_mutable("b")
    assert helper.is_mutable("c")

    assert helper.is_within_bounds("b", 3.0)
    assert not helper.is_within_bounds("b", -1.0)
    assert not helper.is_within_bounds("b", 6.0)
    assert helper.is_within_bounds("c", 1000.0)  # unbound feature

    assert helper.clamp("b", -1.0) == 0.0
    assert helper.clamp("b", 6.0) == 5.0
    assert helper.clamp("b", 3.0) == 3.0
    assert helper.clamp("c", 1000.0) == 1000.0


def test_logistic_immutability(logistic_model):
    """Assert that analytical counterfactuals skip immutable features."""
    model, X, names = logistic_model
    # Instance 1 has high positive probability (unfavorable/risk prediction)
    report = explain(model, X, idx=1, feature_names=names, threshold=0.5, immutable_features=["income"])
    assert report.probability >= 0.5
    assert len(report.counterfactuals) > 0

    # Ensure no counterfactual is suggested for the immutable feature
    for cf in report.counterfactuals:
        assert cf.feature != "income"


def test_forest_immutability(forest_model):
    """Assert that perturbation-based counterfactuals skip immutable features."""
    model, X, names = forest_model
    # Instance 1 is unfavorable
    report = explain(model, X, idx=1, feature_names=names, threshold=0.5, immutable_features=["income"])
    assert report.probability >= 0.5
    assert len(report.counterfactuals) > 0

    # Ensure no counterfactual is suggested for the immutable feature
    for cf in report.counterfactuals:
        assert cf.feature != "income"


def test_logistic_bounds(logistic_model):
    """Assert that analytical counterfactuals obey bounds."""
    model, X, names = logistic_model
    # Set narrow bounds for tenure such that its counterfactual is forced out of bounds
    narrow_bounds = {"tenure": (-0.1, 0.1)}
    report = explain(model, X, idx=1, feature_names=names, threshold=0.5, feature_bounds=narrow_bounds)
    assert len(report.counterfactuals) > 0

    # Check that any counterfactual suggested for tenure respects the bounds
    for cf in report.counterfactuals:
        if cf.feature == "tenure":
            assert -0.1 <= cf.target_value <= 0.1


def test_forest_bounds(forest_model):
    """Assert that perturbation-based counterfactuals obey bounds."""
    model, X, names = forest_model
    # Set narrow bounds for tenure
    narrow_bounds = {"tenure": (-0.1, 0.1)}
    report = explain(model, X, idx=1, feature_names=names, threshold=0.5, feature_bounds=narrow_bounds)
    assert len(report.counterfactuals) > 0

    for cf in report.counterfactuals:
        if cf.feature == "tenure":
            assert -0.1 <= cf.target_value <= 0.1


def test_infeasible_warning(logistic_model):
    """Verify that a UserWarning is raised and counterfactuals is empty when infeasible."""
    model, X, names = logistic_model
    # Make all features immutable
    with pytest.warns(UserWarning, match="No feasible counterfactual path exists"):
        report = explain(
            model,
            X,
            idx=1,
            feature_names=names,
            threshold=0.5,
            immutable_features=["income", "tenure", "debt"],
        )
    assert len(report.counterfactuals) == 0
