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


def test_explain_multiclass_threshold_warning(simple_data):
    X, y = simple_data
    # Convert target to 3 classes
    y_multi = np.copy(y)
    y_multi[10:20] = 2
    model = LogisticRegression(random_state=42).fit(X, y_multi)

    with pytest.warns(UserWarning, match="Threshold is not applicable to multi-class"):
        explain(model, X, idx=0, threshold=0.45)


def test_explain_multiclass_runner_up_default(simple_data):
    X, y = simple_data
    y_multi = np.copy(y)
    y_multi[10:20] = 2
    model = LogisticRegression(random_state=42).fit(X, y_multi)

    report = explain(model, X, idx=0)
    # Default target_class should be runner-up and not match prediction
    assert report.probabilities is not None
    sorted_probs = sorted(report.probabilities.items(), key=lambda item: item[1])
    expected_runner_up = sorted_probs[-2][0]

    assert report.domain_output["target_class"] == expected_runner_up
