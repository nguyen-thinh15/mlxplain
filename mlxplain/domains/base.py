"""Abstract base for domain-specific interpreters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlxplain.core.report import ExplanationReport


class BaseDomain(ABC):
    """Interface for pluggable domain interpreters."""

    positive_label: str
    negative_label: str

    @abstractmethod
    def interpret(self, report: ExplanationReport) -> ExplanationReport:
        """Enrich a generic report with domain-specific context.

        Should populate report.summary and report.domain_output.
        Returns the same report object (mutated).
        """
