"""mlxplain — ML model explainability with pluggable domain interpretation."""

from mlxplain._version import __version__
from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver
from mlxplain.engine import explain, explain_cluster, explain_risk

__all__ = [
    "Counterfactual",
    "ExplanationReport",
    "FeatureDriver",
    "__version__",
    "explain",
    "explain_cluster",
    "explain_risk",
]
