"""Tests for the native language parameter support in translators, engine, and credit risk domain."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain import explain, explain_risk
from mlxplain.translators.logistic import LogisticTranslator
from mlxplain.translators.tree import TreeTranslator

has_shap = importlib.util.find_spec("shap") is not None


@pytest.fixture
def sample_data():
    """Simple synthetic data and models for language testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3)
    # y designed to trigger a decline on idx=0 with a threshold of 0.40
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    feature_names = ["income", "tenure", "debt"]
    return X, y, feature_names


def test_translators_vietnamese_directions(sample_data):
    X, y, names = sample_data

    # 1. Logistic
    model_lr = LogisticRegression(random_state=42).fit(X, y)
    trans_lr = LogisticTranslator(language="vi")
    drivers_lr = trans_lr.extract_drivers(model_lr, X, 0, names)
    assert all(d.direction in ("tích cực", "tiêu cực") for d in drivers_lr)

    # 2. Decision Tree
    model_dt = DecisionTreeClassifier(random_state=42).fit(X, y)
    trans_dt = TreeTranslator(language="vi")
    drivers_dt = trans_dt.extract_drivers(model_dt, X, 0, names)
    assert all(d.direction in ("tích cực", "tiêu cực") for d in drivers_dt)

    # 3. Ensemble (model-agnostic fallback via RF)
    if has_shap:
        from mlxplain.translators.ensemble import EnsembleTranslator

        model_rf = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, y)
        trans_rf = EnsembleTranslator(language="vi")
        drivers_rf = trans_rf.extract_drivers(model_rf, X, 0, names)
        assert all(d.direction in ("tích cực", "tiêu cực") for d in drivers_rf)


def test_explain_vietnamese_orchestration(sample_data):
    X, y, names = sample_data
    model = LogisticRegression(random_state=42).fit(X, y)

    # Explain in Vietnamese
    report = explain(model, X, idx=0, feature_names=names, threshold=0.40, language="vi")

    # Drivers list must be correctly filtered and filled despite Vietnamese direction strings
    assert len(report.positive_drivers) + len(report.negative_drivers) > 0
    assert all(d.direction in ("tích cực", "tiêu cực") for d in report.positive_drivers + report.negative_drivers)
    assert report.prediction in ("Tích cực", "Tiêu cực")


def test_explain_risk_vietnamese_memo(sample_data):
    X, y, names = sample_data
    model = LogisticRegression(random_state=42).fit(X, y)

    # Trigger a decline decision (X[0] default probability should exceed 0.40)
    report = explain_risk(model, X, idx=0, feature_names=names, threshold=0.40, language="vi")

    assert report.prediction in ("Từ chối", "Phê duyệt")
    memo = report.summary

    # Verify key Vietnamese headings are present in the summary memo
    assert "QUYẾT ĐỊNH TÍN DỤNG:" in memo
    assert "Xác suất Nợ xấu:" in memo

    if "YẾU TỐ RỦI RO (RISK FACTORS):" in memo:
        assert "mức độ ảnh hưởng:" in memo
        assert "tổng mức rủi ro" in memo

    if "PHƯƠNG ÁN KHẮC PHỤC" in memo:
        assert "tăng từ" in memo or "giảm từ" in memo


def test_multiclass_vietnamese_localization():
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
    report = explain_risk(model, X, idx=0, feature_names=["income", "tenure", "debt"], target_class="A", language="vi")

    assert "QUYẾT ĐỊNH PHÂN HẠNG TÍN DỤNG: Hạng" in report.summary
    assert "Xác suất của Hạng" in report.summary
    assert "PHƯƠNG ÁN KHẮC PHỤC ĐỂ ĐẠT Hạng A" in report.summary
