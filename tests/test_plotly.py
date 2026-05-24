"""Automated tests for interactive Plotly charts."""

import importlib.util
from unittest.mock import patch

import pytest

from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver
from mlxplain.visualizations.plotly_charts import (
    _ensure_plotly,
    plot_counterfactuals_plotly,
    plot_drivers_waterfall_plotly,
    plot_gauge_plotly,
    plot_per_class_driver_heatmap_plotly,
    plot_probabilities_bar_plotly,
    plot_report_plotly,
)

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


@pytest.fixture
def sample_report():
    pos_drivers = [FeatureDriver(feature="FICO", value=720, impact=0.25, direction="positive")]
    neg_drivers = [FeatureDriver(feature="Income", value=50000, impact=0.15, direction="negative")]
    cfs = [Counterfactual(feature="Income", current_value=50000, target_value=60000, change_needed=10000)]
    return ExplanationReport(
        prediction="Declined",
        probability=0.75,
        threshold=0.5,
        positive_drivers=pos_drivers,
        negative_drivers=neg_drivers,
        counterfactuals=cfs,
        summary="Sample Memo",
    )


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_gauge_generation():
    fig = plot_gauge_plotly(0.75, 0.5, "Approved", "Declined")
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_drivers_generation():
    fig = plot_drivers_waterfall_plotly(
        [FeatureDriver(feature="FICO", value=720, impact=0.25, direction="positive")],
        [FeatureDriver(feature="Income", value=50000, impact=0.15, direction="negative")],
    )
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_drivers_empty():
    fig = plot_drivers_waterfall_plotly([], [])
    assert fig is not None
    assert len(fig.data) == 0  # Should display an annotation instead of a bar chart


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_counterfactuals():
    fig = plot_counterfactuals_plotly(
        [Counterfactual(feature="Income", current_value=50000, target_value=60000, change_needed=10000)]
    )
    assert fig is not None
    assert len(fig.data) == 2  # Current and Required bars
    assert fig.data[0].name == "Current Value"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_probabilities_bar():
    fig = plot_probabilities_bar_plotly({"A": 0.8, "B": 0.2})
    assert fig is not None
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [0.8, 0.2]


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_per_class_driver_heatmap():
    d = FeatureDriver(
        feature="FICO", value=720, impact=0.2, direction="positive", per_class_impacts={"A": 0.1, "B": -0.1}
    )
    fig = plot_per_class_driver_heatmap_plotly([d], ["A", "B"])
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly package not installed")
def test_plotly_report_orchestration(sample_report):
    figs = plot_report_plotly(sample_report)
    assert "gauge" in figs
    assert "drivers" in figs
    assert "counterfactuals" in figs


def test_ensure_plotly_import_error():
    with (
        patch("builtins.__import__", side_effect=ImportError("plotly missing")),
        pytest.raises(ImportError, match="The 'plotly' package is required"),
    ):
        _ensure_plotly()
