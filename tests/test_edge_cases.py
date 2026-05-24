"""Tests for input validation edge cases."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlxplain import explain


@pytest.fixture
def fitted_model():
    """A simple fitted logistic regression for validation tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(random_state=42).fit(X, y)
    return model, X


def test_idx_out_of_bounds(fitted_model):
    model, X = fitted_model
    with pytest.raises(ValueError, match="idx=100 is out of bounds"):
        explain(model, X, idx=100, feature_names=["a", "b", "c"])


def test_idx_negative(fitted_model):
    model, X = fitted_model
    with pytest.raises(ValueError, match="idx=-1 is out of bounds"):
        explain(model, X, idx=-1, feature_names=["a", "b", "c"])


def test_feature_names_length_mismatch(fitted_model):
    model, X = fitted_model
    with pytest.raises(ValueError, match="feature_names has 2 entries but X has 3 columns"):
        explain(model, X, idx=0, feature_names=["a", "b"])


def test_1d_input_rejected(fitted_model):
    model, _ = fitted_model
    with pytest.raises(ValueError, match="X must be a 2D array"):
        explain(model, np.array([1.0, 2.0, 3.0]), idx=0)


def test_favorable_prediction_has_no_counterfactuals(fitted_model):
    """When probability < threshold, counterfactuals should be empty."""
    model, X = fitted_model
    report = explain(model, X, idx=0, feature_names=["a", "b", "c"])
    if report.probability < report.threshold:
        assert report.counterfactuals == []


def test_top_k_limits_drivers(fitted_model):
    """top_k should limit drivers per direction."""
    model, X = fitted_model
    report = explain(model, X, idx=0, feature_names=["a", "b", "c"], top_k=1)
    assert len(report.positive_drivers) <= 1
    assert len(report.negative_drivers) <= 1
