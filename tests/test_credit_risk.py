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
    # Ensure normalized percentages are printed
    assert "% of risk impact" in report.summary or "% of mitigating impact" in report.summary


def test_credit_risk_figures(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    assert "gauge" in report.figures
    assert "drivers" in report.figures


def test_memo_impact_normalization_sums_to_100_per_group(credit_model):
    model, X, names = credit_model
    report = explain_risk(model, X, idx=0, feature_names=names)
    import re

    # Extract all percentages from risk factors and mitigating factors
    risk_percentages = [float(p) for p in re.findall(r"(\d+\.\d+)% of risk impact", report.summary)]
    mit_percentages = [float(p) for p in re.findall(r"(\d+\.\d+)% of mitigating impact", report.summary)]

    if risk_percentages:
        assert abs(sum(risk_percentages) - 100.0) < 1e-1
    if mit_percentages:
        assert abs(sum(mit_percentages) - 100.0) < 1e-1


def test_memo_handles_zero_total_impact_gracefully():
    from mlxplain.core.report import ExplanationReport, FeatureDriver

    report = ExplanationReport(
        prediction="Declined",
        probability=0.75,
        threshold=0.5,
        positive_drivers=[FeatureDriver("debt", 10.0, 0.0, "positive")],
        negative_drivers=[FeatureDriver("income", 50.0, 0.0, "negative")],
        counterfactuals=[],
    )
    report.domain_output = {
        "positive_label": "Declined",
        "negative_label": "Approved",
        "decision": "Declined",
        "risk_factors": report.positive_drivers,
        "mitigating_factors": report.negative_drivers,
        "cure_paths": report.counterfactuals,
        "language": "en",
    }
    from mlxplain.domains.credit_risk.templates import generate_credit_memo

    summary = generate_credit_memo(report)
    assert "0.0% of risk impact" in summary
    assert "0.0% of mitigating impact" in summary


def test_memo_handles_single_driver_per_group():
    from mlxplain.core.report import ExplanationReport, FeatureDriver

    report = ExplanationReport(
        prediction="Declined",
        probability=0.75,
        threshold=0.5,
        positive_drivers=[FeatureDriver("debt", 10.0, 5.0, "positive")],
        negative_drivers=[FeatureDriver("income", 50.0, 2.0, "negative")],
        counterfactuals=[],
    )
    report.domain_output = {
        "positive_label": "Declined",
        "negative_label": "Approved",
        "decision": "Declined",
        "risk_factors": report.positive_drivers,
        "mitigating_factors": report.negative_drivers,
        "cure_paths": report.counterfactuals,
        "language": "en",
    }
    from mlxplain.domains.credit_risk.templates import generate_credit_memo

    summary = generate_credit_memo(report)
    assert "100.0% of risk impact" in summary
    assert "100.0% of mitigating impact" in summary


def test_interpret_multiclass_grading():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from mlxplain import explain_risk

    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    score = X[:, 0] * 1.5 + X[:, 1] * 1.0 - X[:, 2] * 2.0
    y = np.zeros(100, dtype=object)
    y[score > 0.5] = "A"
    y[(score >= -0.5) & (score <= 0.5)] = "B"
    y[score < -0.5] = "C"

    model = LogisticRegression(random_state=42).fit(X, y.astype(str))
    report = explain_risk(model, X, idx=0, feature_names=["income", "tenure", "debt"], target_class="A")

    assert "CREDIT GRADE DECISION: Grade" in report.summary
    assert "Probability of Grade" in report.summary
    assert "KEY FACTORS FOR ASSIGNED GRADE" in report.summary
