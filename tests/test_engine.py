"""Tests for the engine: model detection and unified explain() API."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain.engine import _detect_translator, explain
from mlxplain.translators.logistic import LogisticTranslator
from mlxplain.translators.tree import TreeTranslator


@pytest.fixture
def simple_data():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_detect_logistic(simple_data):
    X, y = simple_data
    model = LogisticRegression().fit(X, y)
    translator = _detect_translator(model)
    assert isinstance(translator, LogisticTranslator)


def test_detect_tree(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    translator = _detect_translator(model)
    assert isinstance(translator, TreeTranslator)


def test_detect_unsupported():
    class FakeModel:
        pass

    with pytest.raises(ValueError, match="Unsupported model type"):
        _detect_translator(FakeModel())


def test_explain_default_feature_names(simple_data):
    X, y = simple_data
    model = LogisticRegression().fit(X, y)
    report = explain(model, X, idx=0)
    # Should auto-generate f0, f1
    all_features = {d.feature for d in report.positive_drivers + report.negative_drivers}
    assert all_features.issubset({"f0", "f1"})


def test_explain_custom_labels(simple_data):
    X, y = simple_data
    model = LogisticRegression().fit(X, y)
    report = explain(model, X, idx=0, positive_label="High", negative_label="Low")
    assert report.prediction in ("High", "Low")
