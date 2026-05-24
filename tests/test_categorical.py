"""Tests for explaining models with encoded categorical features (ordinal and one-hot)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlxplain import explain
from mlxplain.core.report import ExplanationReport


def test_ordinal_encoded_categorical_features():
    """Verify that ordinal-encoded categorical features are explained and bounded correctly."""
    rng = np.random.RandomState(42)
    n_samples = 150

    # 3 features:
    # 0: income (continuous)
    # 1: education (ordinal: 0=High School, 1=Bachelor's, 2=Master's, 3=PhD)
    # 2: debt (continuous)
    income = rng.uniform(20, 150, n_samples)
    education = rng.choice([0, 1, 2, 3], size=n_samples)
    debt = rng.uniform(5, 80, n_samples)

    X = np.column_stack([income, education, debt])
    feature_names = ["income", "education", "debt"]

    # Higher income + education pushes toward approval (1), higher debt pushes toward decline (0)
    # Scaled to make education significant
    y = (income * 0.1 + education * 1.5 - debt * 0.2 > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Select an instance predicted as unfavorable (0)
    # Let's find one where predicted probability of 1 is low (so it is classified as 0)
    preds = model.predict(X)
    target_idx = -1
    for i, pred in enumerate(preds):
        if pred == 0 and X[i, 1] == 1:  # Bachelor's
            target_idx = i
            break

    if target_idx == -1:
        target_idx = np.where(preds == 0)[0][0]

    # Explain without constraints
    report = explain(model=model, X=X, idx=target_idx, feature_names=feature_names, threshold=0.5)

    assert isinstance(report, ExplanationReport)
    assert report.probability < 0.5

    # Verify education is present in positive/negative drivers
    all_drivers = report.positive_drivers + report.negative_drivers
    driver_names = [d.feature for d in all_drivers]
    assert "education" in driver_names

    # Explain with bounds to ensure the ordinal category target value stays within realistic ranks
    # e.g., cannot go above PhD (3)
    bounds = {"education": (0, 3)}
    report_bounded = explain(
        model=model, X=X, idx=target_idx, feature_names=feature_names, threshold=0.5, feature_bounds=bounds
    )

    for cf in report_bounded.counterfactuals:
        if cf.feature == "education":
            assert 0 <= cf.target_value <= 3

    # Explain making the education category immutable
    report_immutable = explain(
        model=model, X=X, idx=target_idx, feature_names=feature_names, threshold=0.5, immutable_features=["education"]
    )

    for cf in report_immutable.counterfactuals:
        assert cf.feature != "education"


def test_one_hot_encoded_categorical_features():
    """Verify explaining models with one-hot encoded categorical features using constraints."""
    rng = np.random.RandomState(42)
    n_samples = 150

    # 4 features:
    # 0: income (continuous)
    # 1: employed (binary one-hot)
    # 2: unemployed (binary one-hot)
    # 3: self_employed (binary one-hot)
    income = rng.uniform(20, 150, n_samples)

    # Mutually exclusive one-hot columns
    categories = rng.choice([0, 1, 2], size=n_samples)
    employed = (categories == 0).astype(float)
    unemployed = (categories == 1).astype(float)
    self_employed = (categories == 2).astype(float)

    X = np.column_stack([income, employed, unemployed, self_employed])
    feature_names = ["income", "employed", "unemployed", "self_employed"]

    # approval formula
    y = (income * 0.1 + employed * 3.0 - unemployed * 4.0 > 2.0).astype(int)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    # Pick a candidate who is unemployed (unemployed = 1.0) and predicted unfavorable
    preds = model.predict(X)
    target_idx = -1
    for i, pred in enumerate(preds):
        if pred == 0 and X[i, 2] == 1.0:
            target_idx = i
            break

    if target_idx == -1:
        target_idx = np.where(preds == 0)[0][0]

    # Explain without constraints
    report = explain(model=model, X=X, idx=target_idx, feature_names=feature_names, threshold=0.5)

    assert isinstance(report, ExplanationReport)
    assert report.probability < 0.5

    # Drivers should contain the one-hot feature names
    all_drivers = [d.feature for d in report.positive_drivers + report.negative_drivers]
    assert "unemployed" in all_drivers

    # To prevent invalid continuous counterfactual changes for binary one-hot features,
    # we can set their bounds to (0, 1) or mark them immutable to only suggest changes
    # for the numerical features (e.g. income).
    report_imm = explain(
        model=model,
        X=X,
        idx=target_idx,
        feature_names=feature_names,
        threshold=0.5,
        immutable_features=["employed", "unemployed", "self_employed"],
    )

    # Ensure no counterfactuals are suggested for the immutable one-hot categories
    for cf in report_imm.counterfactuals:
        assert cf.feature not in ["employed", "unemployed", "self_employed"]
        assert cf.feature == "income"

    # Bounds constraint test
    bounds = {"employed": (0.0, 1.0), "unemployed": (0.0, 1.0), "self_employed": (0.0, 1.0)}
    report_bounded = explain(
        model=model, X=X, idx=target_idx, feature_names=feature_names, threshold=0.5, feature_bounds=bounds
    )

    for cf in report_bounded.counterfactuals:
        if cf.feature in bounds:
            assert 0.0 <= cf.target_value <= 1.0
