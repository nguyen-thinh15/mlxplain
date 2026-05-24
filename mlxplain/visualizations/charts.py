"""Domain-agnostic matplotlib charts for explanation reports."""

from __future__ import annotations

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
    impacts = [d.impact if d.direction == "positive" else -d.impact for d in all_drivers]
    colors = ["#F44336" if d.direction == "positive" else "#4CAF50" for d in all_drivers]

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


def plot_report(report: ExplanationReport) -> dict[str, Figure]:
    """Generate all three charts from a report. Returns dict of name → Figure."""
    positive_label = report.domain_output.get("positive_label", "Positive")
    negative_label = report.domain_output.get("negative_label", "Negative")

    figures = {
        "gauge": plot_gauge(report.probability, report.threshold, positive_label, negative_label),
        "drivers": plot_drivers_waterfall(report.positive_drivers, report.negative_drivers),
        "counterfactuals": plot_counterfactuals(report.counterfactuals),
    }
    return figures
