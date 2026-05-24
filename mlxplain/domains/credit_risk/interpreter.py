"""Credit risk domain interpreter — maps generic explanations to credit underwriting terms."""

from __future__ import annotations

from mlxplain.core.report import ExplanationReport
from mlxplain.domains.base import BaseDomain
from mlxplain.domains.credit_risk.templates import generate_credit_memo


class CreditRiskDomain(BaseDomain):
    """Interprets generic ML explanations through a credit risk lens.

    Mapping:
        positive prediction → Declined (high default risk)
        negative prediction → Approved (low risk)
        positive drivers    → Risk Factors
        negative drivers    → Mitigating Factors
        counterfactuals     → Cure Paths
    """

    positive_label = "Declined"
    negative_label = "Approved"

    def interpret(self, report: ExplanationReport) -> ExplanationReport:
        # Re-classify with credit risk labels
        decision = self.positive_label if report.probability >= report.threshold else self.negative_label

        report.prediction = decision
        report.domain_output = {
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "decision": decision,
            "risk_factors": report.positive_drivers,
            "mitigating_factors": report.negative_drivers,
            "cure_paths": report.counterfactuals,
        }
        report.summary = generate_credit_memo(report)
        return report
