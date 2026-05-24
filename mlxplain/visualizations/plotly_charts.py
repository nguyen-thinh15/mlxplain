"""Domain-agnostic interactive Plotly charts for explanation reports."""

from __future__ import annotations

from typing import Any

from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver


def _ensure_plotly():
    """Ensure plotly is installed; otherwise raise a clear, helpful ImportError."""
    try:
        import plotly
        import plotly.graph_objects as go

        return plotly, go
    except ImportError as e:
        raise ImportError(
            "The 'plotly' package is required for interactive visualizations. "
            'Install it using: pip install "mlxplain-xai[plotly]"'
        ) from e


def plot_gauge_plotly(
    probability: float,
    threshold: float,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> Any:
    """Horizontal interactive gauge showing probability relative to threshold."""
    _, go = _ensure_plotly()

    color = "#F44336" if probability >= threshold else "#4CAF50"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"size": 24, "color": color}, "valueformat": ".3f"},
            title={"text": "Prediction Probability Gauge", "font": {"size": 14, "color": "#424242"}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "gray"},
                "bar": {"color": color, "thickness": 0.4},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E0E0E0",
                "steps": [
                    {"range": [0, threshold], "color": "rgba(76, 175, 80, 0.15)"},
                    {"range": [threshold, 1], "color": "rgba(244, 67, 54, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "gray", "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )

    # Add custom annotations to label green/red zones
    fig.add_annotation(
        x=threshold / 2,
        y=0.15,
        text=negative_label,
        showarrow=False,
        font={"color": "#2E7D32", "weight": "bold", "size": 12},
    )
    fig.add_annotation(
        x=threshold + (1 - threshold) / 2,
        y=0.15,
        text=positive_label,
        showarrow=False,
        font={"color": "#C62828", "weight": "bold", "size": 12},
    )

    fig.update_layout(
        height=220,
        margin={"t": 50, "b": 30, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_drivers_waterfall_plotly(
    positive_drivers: list[FeatureDriver],
    negative_drivers: list[FeatureDriver],
) -> Any:
    """Horizontal interactive bar chart of positive vs. negative feature contributions."""
    _, go = _ensure_plotly()

    all_drivers = sorted(positive_drivers + negative_drivers, key=lambda d: d.impact, reverse=True)
    if not all_drivers:
        fig = go.Figure()
        fig.add_annotation(text="No drivers to display", showarrow=False, font={"size": 14})
        fig.update_layout(height=150, margin={"t": 30, "b": 30, "l": 30, "r": 30})
        return fig

    names = [d.feature for d in all_drivers]
    impacts = [d.impact if d.direction in ("positive", "tích cực") else -d.impact for d in all_drivers]
    colors = ["#F44336" if d.direction in ("positive", "tích cực") else "#4CAF50" for d in all_drivers]

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=names,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="<b>%{y}</b><br>Impact: %{x:+.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Feature Drivers Impact", "font": {"size": 14, "color": "#424242"}},
        xaxis={"title": "Feature Impact", "zeroline": True, "zerolinecolor": "black", "zerolinewidth": 0.5},
        yaxis={"autorange": "reversed"},
        height=max(200, len(names) * 35),
        margin={"t": 40, "b": 40, "l": 150, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_counterfactuals_plotly(counterfactuals: list[Counterfactual]) -> Any:
    """Grouped interactive bar chart showing current vs. required values for each counterfactual."""
    _, go = _ensure_plotly()

    if not counterfactuals:
        fig = go.Figure()
        fig.add_annotation(text="No counterfactuals to display", showarrow=False, font={"size": 14})
        fig.update_layout(height=150, margin={"t": 30, "b": 30, "l": 30, "r": 30})
        return fig

    # Show top 5 cheapest changes
    cfs = counterfactuals[:5]
    names = [c.feature for c in cfs]
    current = [c.current_value for c in cfs]
    target = [c.target_value for c in cfs]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Current Value",
                x=current,
                y=names,
                orientation="h",
                marker={"color": "#90CAF9"},
                hovertemplate="Current: %{x:.3f}<extra></extra>",
            ),
            go.Bar(
                name="Required Value",
                x=target,
                y=names,
                orientation="h",
                marker={"color": "#FFB74D"},
                hovertemplate="Required: %{x:.3f}<extra></extra>",
            ),
        ]
    )

    fig.update_layout(
        title={"text": "Counterfactual Changes", "font": {"size": 14, "color": "#424242"}},
        xaxis={"title": "Value"},
        yaxis={"autorange": "reversed"},
        barmode="group",
        height=max(200, len(names) * 45),
        margin={"t": 40, "b": 40, "l": 150, "r": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "right", "x": 1},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_probabilities_bar_plotly(probabilities: dict[str, float]) -> Any:
    """Interactive horizontal bar chart showing predicted probabilities for each class."""
    _, go = _ensure_plotly()

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[0])
    classes = [item[0] for item in sorted_probs]
    probs = [item[1] for item in sorted_probs]

    max_prob = max(probs)
    colors = ["#1976D2" if p == max_prob else "#E0E0E0" for p in probs]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker={"color": colors},
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
            hovertemplate="Class: %{y}<br>Probability: %{x:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Class Probabilities", "font": {"size": 14, "color": "#424242"}},
        xaxis={"range": [0, 1.05], "visible": False},
        yaxis={"autorange": "reversed"},
        height=max(180, len(classes) * 40),
        margin={"t": 40, "b": 20, "l": 100, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_per_class_driver_heatmap_plotly(
    feature_drivers: list[FeatureDriver],
    classes: list[str],
) -> Any:
    """Plot an interactive Plotly heatmap of feature contributions per class."""
    _, go = _ensure_plotly()

    valid_drivers = [d for d in feature_drivers if d.per_class_impacts is not None]
    if not valid_drivers or not classes:
        fig = go.Figure()
        fig.add_annotation(text="No driver heatmap data available", showarrow=False, font={"size": 14})
        fig.update_layout(height=150, margin={"t": 30, "b": 30, "l": 30, "r": 30})
        return fig

    features = [d.feature for d in valid_drivers]

    data = []
    for d in valid_drivers:
        row = [d.per_class_impacts.get(str(c), 0.0) for c in classes]
        data.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=data,
            x=classes,
            y=features,
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            text=[[f"{val:+.3f}" for val in row] for row in data],
            texttemplate="%{text}",
            hovertemplate="Feature: %{y}<br>Class: %{x}<br>Impact: %{z:+.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Per-Class Feature Impact Heatmap", "font": {"size": 14, "color": "#424242"}},
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed"},
        height=max(220, len(features) * 45),
        margin={"t": 80, "b": 20, "l": 150, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_report_plotly(report: ExplanationReport) -> dict[str, Any]:
    """Generate all interactive Plotly charts from a report. Returns dict of name -> Plotly Figure."""
    positive_label = report.domain_output.get("positive_label", "Positive")
    negative_label = report.domain_output.get("negative_label", "Negative")

    if report.probabilities is not None:
        classes = report.domain_output.get("classes", list(report.probabilities.keys()))
        figures = {
            "probabilities": plot_probabilities_bar_plotly(report.probabilities),
            "heatmap": plot_per_class_driver_heatmap_plotly(report.positive_drivers + report.negative_drivers, classes),
            "drivers": plot_drivers_waterfall_plotly(report.positive_drivers, report.negative_drivers),
            "counterfactuals": plot_counterfactuals_plotly(report.counterfactuals),
        }
    else:
        figures = {
            "gauge": plot_gauge_plotly(report.probability, report.threshold, positive_label, negative_label),
            "drivers": plot_drivers_waterfall_plotly(report.positive_drivers, report.negative_drivers),
            "counterfactuals": plot_counterfactuals_plotly(report.counterfactuals),
        }
    return figures
