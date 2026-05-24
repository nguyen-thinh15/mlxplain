"""Domain-agnostic matplotlib charts for explanation reports."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver


def plot_gauge(
    probability: float,
    threshold: float,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> Figure:
    """Horizontal gauge showing probability relative to threshold."""
    fig = Figure(figsize=(8, 1.5))
    ax = fig.add_subplot(111)

    # Draw colored zones
    ax.barh(0, threshold, height=0.4, color="#4CAF50", alpha=0.3, left=0)
    ax.barh(0, 1 - threshold, height=0.4, color="#F44336", alpha=0.3, left=threshold)

    # Threshold line
    ax.axvline(threshold, color="gray", linewidth=2, linestyle="--", zorder=3)
    ax.text(threshold, 0.35, f"Threshold: {threshold:.2f}", ha="center", fontsize=8, color="gray")

    # Probability marker
    color = "#F44336" if probability >= threshold else "#4CAF50"
    ax.plot(probability, 0, "D", markersize=14, color=color, zorder=4)
    ax.text(probability, -0.35, f"{probability:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Labels
    ax.text(threshold / 2, 0, negative_label, ha="center", va="center", fontsize=9, color="#2E7D32")
    ax.text(threshold + (1 - threshold) / 2, 0, positive_label, ha="center", va="center", fontsize=9, color="#C62828")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Predicted Probability")
    ax.set_title("Prediction Gauge")
    fig.tight_layout()
    return fig


def plot_drivers_waterfall(
    positive_drivers: list[FeatureDriver],
    negative_drivers: list[FeatureDriver],
) -> Figure:
    """Horizontal waterfall of positive vs. negative feature contributions."""
    # Combine and sort by impact
    all_drivers = sorted(positive_drivers + negative_drivers, key=lambda d: d.impact, reverse=True)
    if not all_drivers:
        fig = Figure(figsize=(8, 2))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No drivers to display", ha="center", va="center")
        return fig

    names = [d.feature for d in all_drivers]
    impacts = [d.impact if d.direction in ("positive", "tích cực") else -d.impact for d in all_drivers]
    colors = ["#F44336" if d.direction in ("positive", "tích cực") else "#4CAF50" for d in all_drivers]

    fig = Figure(figsize=(8, max(2, len(names) * 0.4)))
    ax = fig.add_subplot(111)
    y_pos = range(len(names))
    ax.barh(y_pos, impacts, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature Impact")
    ax.set_title("Feature Drivers")
    fig.tight_layout()
    return fig


def plot_counterfactuals(counterfactuals: list[Counterfactual]) -> Figure:
    """Grouped bars showing current vs. required values for each counterfactual."""
    if not counterfactuals:
        fig = Figure(figsize=(8, 2))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No counterfactuals to display", ha="center", va="center")
        return fig

    # Show top 5 cheapest changes
    cfs = counterfactuals[:5]
    names = [c.feature for c in cfs]
    current = [c.current_value for c in cfs]
    target = [c.target_value for c in cfs]

    fig = Figure(figsize=(8, max(2, len(names) * 0.5)))
    ax = fig.add_subplot(111)
    y_pos = range(len(names))
    bar_height = 0.3

    ax.barh([y - bar_height / 2 for y in y_pos], current, height=bar_height, color="#90CAF9", label="Current")
    ax.barh([y + bar_height / 2 for y in y_pos], target, height=bar_height, color="#FFB74D", label="Required")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Value")
    ax.set_title("Counterfactual Changes")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_probabilities_bar(probabilities: dict[str, float]) -> Figure:
    """Horizontal bar chart showing predicted probabilities for each class."""
    fig = Figure(figsize=(8, max(2.0, len(probabilities) * 0.45)))
    ax = fig.add_subplot(111)

    # Sort classes by name/label
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[0])
    classes = [item[0] for item in sorted_probs]
    probs = [item[1] for item in sorted_probs]

    # Highlight the class with max probability
    max_prob = max(probs)
    colors = ["#1976D2" if p == max_prob else "#E0E0E0" for p in probs]

    y_pos = range(len(classes))
    bars = ax.barh(y_pos, probs, color=colors, height=0.6, edgecolor="none")

    # Add text labels inside/beside the bars
    for bar, p in zip(bars, probs, strict=False):
        width = bar.get_width()
        text_color = "white" if p == max_prob else "#424242"
        if width > 0.15:
            ax.text(
                width - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}",
                va="center",
                ha="right",
                fontsize=8,
                color=text_color,
                fontweight="bold",
            )
        else:
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}",
                va="center",
                ha="left",
                fontsize=8,
                color="#424242",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", fontsize=9)
    ax.set_title("Class Probabilities", fontsize=10, fontweight="bold", pad=12)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.get_xaxis().set_visible(False)

    fig.tight_layout()
    return fig


def plot_per_class_driver_heatmap(
    feature_drivers: list[FeatureDriver],
    classes: list[str],
) -> Figure:
    """Plot a premium heatmap of feature contributions/impacts per class."""
    if not feature_drivers or not classes:
        fig = Figure(figsize=(8, 2))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No driver data for heatmap", ha="center", va="center")
        return fig

    # Filter drivers that have per_class_impacts
    valid_drivers = [d for d in feature_drivers if d.per_class_impacts is not None]
    if not valid_drivers:
        fig = Figure(figsize=(8, 2))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Per-class impacts missing", ha="center", va="center")
        return fig

    features = [d.feature for d in valid_drivers]

    data = np.zeros((len(features), len(classes)))
    for f_idx, d in enumerate(valid_drivers):
        for c_idx, c in enumerate(classes):
            data[f_idx, c_idx] = d.per_class_impacts.get(str(c), 0.0)

    fig = Figure(figsize=(8, max(2.5, len(features) * 0.5)))
    ax = fig.add_subplot(111)

    vmax = max(1e-5, float(np.max(np.abs(data))))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel("Contribution / Impact", rotation=-90, va="bottom", fontsize=8)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)

    for f_idx in range(len(features)):
        for c_idx in range(len(classes)):
            val = data[f_idx, c_idx]
            text_color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(
                c_idx,
                f_idx,
                f"{val:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="semibold",
            )

    ax.set_title("Per-Class Feature Impact Heatmap", fontsize=10, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_report(report: ExplanationReport) -> dict[str, Figure]:
    """Generate all charts from a report. Returns dict of name → Figure."""
    positive_label = report.domain_output.get("positive_label", "Positive")
    negative_label = report.domain_output.get("negative_label", "Negative")

    if report.probabilities is not None:
        classes = report.domain_output.get("classes", list(report.probabilities.keys()))
        figures = {
            "probabilities": plot_probabilities_bar(report.probabilities),
            "heatmap": plot_per_class_driver_heatmap(report.positive_drivers + report.negative_drivers, classes),
            "drivers": plot_drivers_waterfall(report.positive_drivers, report.negative_drivers),
            "counterfactuals": plot_counterfactuals(report.counterfactuals),
        }
    else:
        figures = {
            "gauge": plot_gauge(report.probability, report.threshold, positive_label, negative_label),
            "drivers": plot_drivers_waterfall(report.positive_drivers, report.negative_drivers),
            "counterfactuals": plot_counterfactuals(report.counterfactuals),
        }
    return figures
