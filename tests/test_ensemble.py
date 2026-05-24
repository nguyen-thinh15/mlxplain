"""End-to-end tests for EnsembleTranslator (XGBoost and LightGBM explanations)."""

import numpy as np
import pytest
import xgboost as xgb
import lightgbm as lgb

from mlxplain import explain
from mlxplain.core.report import ExplanationReport, FeatureDriver


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y, ["income", "tenure", "debt"]


def test_xgboost_explain_returns_report(synthetic_data):
    X, y, names = synthetic_data
    model = xgb.XGBClassifier(random_state=42, n_estimators=3, max_depth=3)
    model.fit(X, y)

    report = explain(model, X, idx=0, feature_names=names)
    assert isinstance(report, ExplanationReport)
    assert 0.0 <= report.probability <= 1.0
    all_drivers = report.positive_drivers + report.negative_drivers
    assert len(all_drivers) > 0
    assert all(isinstance(d, FeatureDriver) for d in all_drivers)
    assert "gauge" in report.figures
    assert "drivers" in report.figures


def test_lightgbm_explain_returns_report(synthetic_data):
    X, y, names = synthetic_data
    model = lgb.LGBMClassifier(random_state=42, n_estimators=3, max_depth=3, verbose=-1)
    model.fit(X, y)

    report = explain(model, X, idx=0, feature_names=names)
    assert isinstance(report, ExplanationReport)
    assert 0.0 <= report.probability <= 1.0
    all_drivers = report.positive_drivers + report.negative_drivers
    assert len(all_drivers) > 0
    assert all(isinstance(d, FeatureDriver) for d in all_drivers)
    assert "gauge" in report.figures
    assert "drivers" in report.figures

