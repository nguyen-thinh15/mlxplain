"""Core data structures for explanation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.figure import Figure


@dataclass
class FeatureDriver:
    """A single feature's contribution to the prediction."""

    feature: str
    value: float
    impact: float
    direction: str  # "positive" or "negative"


@dataclass
class Counterfactual:
    """A single feature change needed to flip the prediction."""

    feature: str
    current_value: float
    target_value: float
    change_needed: float


@dataclass
class ExplanationReport:
    """Complete explanation output returned by all API calls."""

    prediction: str
    probability: float
    threshold: float
    positive_drivers: list[FeatureDriver]
    negative_drivers: list[FeatureDriver]
    counterfactuals: list[Counterfactual]
    figures: dict[str, Figure] = field(default_factory=dict)
    summary: str = ""
    domain_output: dict[str, Any] = field(default_factory=dict)
