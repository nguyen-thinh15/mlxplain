"""Tests for unsupervised model explainability (IsolationForest and KMeans)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from mlxplain import explain, explain_cluster


@pytest.fixture
def synthetic_anomaly_data():
    """Generate synthetic data: 45 normal points + 5 outliers on feature_a only."""
    np.random.seed(42)
    # 45 normal points centered at 0
    normal = np.random.normal(0, 1, (45, 3))
    # 5 outliers that are normal on feature_b and feature_c, but extreme on feature_a
    outliers = np.random.normal(0, 1, (5, 3))
    outliers[:, 0] = np.random.normal(10, 1, 5)
    X = np.vstack([normal, outliers])
    feature_names = ["feature_a", "feature_b", "feature_c"]
    return X, feature_names


@pytest.fixture
def synthetic_cluster_data():
    """Generate synthetic clustering data with 3 distinct clusters."""
    np.random.seed(42)
    c1 = np.random.normal([0, 0], 0.2, (20, 2))
    c2 = np.random.normal([5, 5], 0.2, (20, 2))
    c3 = np.random.normal([10, 0], 0.2, (20, 2))
    X = np.vstack([c1, c2, c3])
    feature_names = ["x_coord", "y_coord"]
    return X, feature_names


def test_isolation_forest_english(synthetic_anomaly_data):
    """Test IsolationForest explanation in English."""
    X, feature_names = synthetic_anomaly_data
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    # Explain an anomalous point (index 45 is the first outlier)
    report = explain(model, X, idx=45, feature_names=feature_names, language="en")

    assert report.prediction in ("Anomaly", "Normal")
    # Anomalous point should be classified as Anomaly with default threshold 0.5
    assert report.prediction == "Anomaly"
    assert report.probability >= 0.5

    # Check drivers
    assert len(report.positive_drivers) + len(report.negative_drivers) == 3
    # The anomaly feature must be the top positive driver
    assert report.positive_drivers[0].feature == "feature_a"
    # Drivers should be sorted by impact per direction
    for drivers in [report.positive_drivers, report.negative_drivers]:
        impacts = [d.impact for d in drivers]
        assert impacts == sorted(impacts, reverse=True)

    # Check counterfactuals
    assert len(report.counterfactuals) > 0
    # Apply the first counterfactual change and verify it reduces anomaly score
    cf = report.counterfactuals[0]
    idx_feat = feature_names.index(cf.feature)
    perturbed = X[45].copy()
    perturbed[idx_feat] = cf.target_value

    original_score = -model.score_samples(X[45].reshape(1, -1))[0]
    perturbed_score = -model.score_samples(perturbed.reshape(1, -1))[0]
    assert perturbed_score < original_score

    # Check figures
    assert "gauge" in report.figures
    assert "drivers" in report.figures
    assert "counterfactuals" in report.figures


def test_isolation_forest_vietnamese(synthetic_anomaly_data):
    """Test IsolationForest explanation in Vietnamese."""
    X, feature_names = synthetic_anomaly_data
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    # Explain an anomalous point in Vietnamese
    report = explain(model, X, idx=45, feature_names=feature_names, language="vi")

    assert report.prediction == "Bất thường"
    assert report.domain_output["positive_label"] == "Bất thường"
    assert report.domain_output["negative_label"] == "Bình thường"

    # The anomaly feature must be the top positive driver in Vietnamese
    assert report.positive_drivers[0].feature == "feature_a"

    # Directions should be translated
    for d in report.positive_drivers:
        assert d.direction == "tích cực"
    for d in report.negative_drivers:
        assert d.direction == "tiêu cực"


def test_kmeans_clustering_english(synthetic_cluster_data):
    """Test KMeans clustering explanation in English."""
    X, feature_names = synthetic_cluster_data
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)

    # Explain point at index 0 (which is in cluster c1)
    report = explain_cluster(model, X, idx=0, feature_names=feature_names, language="en")

    assigned_cluster = model.predict(X[0].reshape(1, -1))[0]
    assert report.prediction == f"Cluster {assigned_cluster}"
    assert report.probability >= 0.5  # Should be closer to assigned than runner-up

    # Drivers
    assert len(report.positive_drivers) + len(report.negative_drivers) == 2
    # Ensure drivers are sorted by impact per direction
    for drivers in [report.positive_drivers, report.negative_drivers]:
        impacts = [d.impact for d in drivers]
        assert impacts == sorted(impacts, reverse=True)

    # Counterfactuals: verify that applying counterfactual changes moves instance to the target cluster
    assert len(report.counterfactuals) > 0
    target_cluster = report.domain_output["target_cluster"]

    # Let's apply ALL counterfactual changes to reconstruct target instance
    perturbed = X[0].copy()
    for cf in report.counterfactuals:
        idx_feat = feature_names.index(cf.feature)
        perturbed[idx_feat] = cf.target_value

    new_pred = model.predict(perturbed.reshape(1, -1))[0]
    assert new_pred == target_cluster

    # Figures
    assert "gauge" in report.figures
    assert "drivers" in report.figures
    assert "counterfactuals" in report.figures


def test_kmeans_clustering_vietnamese(synthetic_cluster_data):
    """Test KMeans clustering explanation in Vietnamese."""
    X, feature_names = synthetic_cluster_data
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)

    # Explain in Vietnamese
    report = explain_cluster(model, X, idx=0, feature_names=feature_names, language="vi")

    assigned_cluster = model.predict(X[0].reshape(1, -1))[0]
    target_cluster = report.domain_output["target_cluster"]

    assert report.prediction == f"Nhóm {assigned_cluster}"
    assert report.domain_output["positive_label"] == f"Nhóm {assigned_cluster}"
    assert report.domain_output["negative_label"] == f"Nhóm {target_cluster}"

    # Directions should be translated
    for d in report.positive_drivers:
        assert d.direction == "tích cực"
    for d in report.negative_drivers:
        assert d.direction == "tiêu cực"
