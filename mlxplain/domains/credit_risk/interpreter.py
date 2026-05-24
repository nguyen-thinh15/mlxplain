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
        language = report.domain_output.get("language", "en")
        pos_label = "Từ chối" if language == "vi" else self.positive_label
        neg_label = "Phê duyệt" if language == "vi" else self.negative_label

        # Re-classify with credit risk labels
        decision = pos_label if report.probability >= report.threshold else neg_label

        report.prediction = decision
        report.domain_output = {
            "positive_label": pos_label,
            "negative_label": neg_label,
            "decision": decision,
            "risk_factors": report.positive_drivers,
            "mitigating_factors": report.negative_drivers,
            "cure_paths": report.counterfactuals,
            "language": language,
        }
        report.summary = generate_credit_memo(report)
        return report
