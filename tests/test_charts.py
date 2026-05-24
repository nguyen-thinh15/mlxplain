"""Tests for visualization charts — each chart returns a valid matplotlib Figure."""

from matplotlib.figure import Figure

from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver
from mlxplain.visualizations.charts import (
    plot_counterfactuals,
    plot_drivers_waterfall,
    plot_gauge,
    plot_report,
)


def test_plot_gauge_returns_figure():
    fig = plot_gauge(0.7, 0.5)
    assert isinstance(fig, Figure)


def test_plot_gauge_custom_labels():
    fig = plot_gauge(0.3, 0.5, positive_label="Declined", negative_label="Approved")
    assert isinstance(fig, Figure)


def test_plot_drivers_waterfall():
    pos = [FeatureDriver("income", 50000, 0.3, "positive")]
    neg = [FeatureDriver("debt", 20000, 0.5, "negative")]
    fig = plot_drivers_waterfall(pos, neg)
    assert isinstance(fig, Figure)


def test_plot_drivers_waterfall_empty():
    fig = plot_drivers_waterfall([], [])
    assert isinstance(fig, Figure)


def test_plot_counterfactuals():
    cfs = [Counterfactual("debt", 20000, 15000, -5000)]
    fig = plot_counterfactuals(cfs)
    assert isinstance(fig, Figure)


def test_plot_counterfactuals_empty():
    fig = plot_counterfactuals([])
    assert isinstance(fig, Figure)


def test_plot_report():
    report = ExplanationReport(
        prediction="Positive",
        probability=0.7,
        threshold=0.5,
        positive_drivers=[FeatureDriver("a", 1.0, 0.5, "positive")],
        negative_drivers=[FeatureDriver("b", 2.0, 0.3, "negative")],
        counterfactuals=[Counterfactual("a", 1.0, 0.5, -0.5)],
        domain_output={"positive_label": "Positive", "negative_label": "Negative"},
    )
    figures = plot_report(report)
    assert "gauge" in figures
    assert "drivers" in figures
    assert "counterfactuals" in figures
    assert all(isinstance(f, Figure) for f in figures.values())


def test_plot_probabilities_bar():
    from mlxplain.visualizations.charts import plot_probabilities_bar

    probs = {"Grade A": 0.2, "Grade B": 0.7, "Grade C": 0.1}
    fig = plot_probabilities_bar(probs)
    assert isinstance(fig, Figure)


def test_plot_per_class_driver_heatmap():
    from mlxplain.visualizations.charts import plot_per_class_driver_heatmap

    drivers = [
        FeatureDriver("income", 50.0, 0.4, "positive", {"A": 0.4, "B": -0.2, "C": -0.2}),
        FeatureDriver("debt", 10.0, 0.3, "negative", {"A": -0.3, "B": 0.1, "C": 0.2}),
    ]
    fig = plot_per_class_driver_heatmap(drivers, ["A", "B", "C"])
    assert isinstance(fig, Figure)
