"""Tests for credit risk domain: explain_risk() produces credit-specific output."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlxplain import explain_risk


@pytest.fixture
def credit_model():
    """Train a model on synthetic credit-like data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    # Higher debt (col 2) → higher default risk
    y = (X[:, 2] > 0.5).astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, ["income", "tenure", "debt_ratio"]


def test_credit_risk_labels(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    assert report.prediction in ("Approved", "Declined")


def test_credit_risk_domain_output(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    assert "decision" in report.domain_output
    assert "risk_factors" in report.domain_output
    assert "mitigating_factors" in report.domain_output
    assert "cure_paths" in report.domain_output


def test_credit_memo_summary(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    assert "CREDIT DECISION:" in report.summary
    assert "Default Probability:" in report.summary


def test_credit_risk_figures(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    assert "gauge" in report.figures
    assert "drivers" in report.figures

