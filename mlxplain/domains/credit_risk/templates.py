"""Credit memo text generation templates."""

from __future__ import annotations

from mlxplain.core.report import ExplanationReport


def generate_credit_memo(report: ExplanationReport) -> str:
    """Generate a human-readable credit memo from an enriched report."""
    domain = report.domain_output
    decision = domain.get("decision", report.prediction)
    risk_factors = domain.get("risk_factors", [])
    mitigating_factors = domain.get("mitigating_factors", [])
    cure_paths = domain.get("cure_paths", [])
    language = domain.get("language", "en")

    if language == "vi":
        lines = [
            f"QUYẾT ĐỊNH TÍN DỤNG: {decision}",
            f"Xác suất Nợ xấu: {report.probability:.1%} (ngưỡng: {report.threshold:.1%})",
            "",
        ]

        if risk_factors:
            lines.append("YẾU TỐ RỦI RO (RISK FACTORS):")
            for rf in risk_factors:
                lines.append(f"  - {rf.feature}: {rf.value:.4g} (mức độ ảnh hưởng: {rf.impact:.4g})")
            lines.append("")

        if mitigating_factors:
            lines.append("YẾU TỐ GIẢM THIỂU RỦI RO (MITIGATING FACTORS):")
            for mf in mitigating_factors:
                lines.append(f"  + {mf.feature}: {mf.value:.4g} (mức độ ảnh hưởng: {mf.impact:.4g})")
            lines.append("")

        if cure_paths:
            lines.append("PHƯƠNG ÁN KHẮC PHỤC (CURE PATHS - thay đổi cần thiết để được duyệt):")
            for cp in cure_paths:
                direction = "tăng" if cp.change_needed > 0 else "giảm"
                to_word = "lên" if direction == "tăng" else "xuống"
                lines.append(f"  → {cp.feature}: {direction} từ {cp.current_value:.4g} {to_word} {cp.target_value:.4g}")
            lines.append("")
    else:
        lines = [
            f"CREDIT DECISION: {decision}",
            f"Default Probability: {report.probability:.1%} (threshold: {report.threshold:.1%})",
            "",
        ]

        if risk_factors:
            lines.append("RISK FACTORS:")
            for rf in risk_factors:
                lines.append(f"  - {rf.feature}: {rf.value:.4g} (impact: {rf.impact:.4g})")
            lines.append("")

        if mitigating_factors:
            lines.append("MITIGATING FACTORS:")
            for mf in mitigating_factors:
                lines.append(f"  + {mf.feature}: {mf.value:.4g} (impact: {mf.impact:.4g})")
            lines.append("")

        if cure_paths:
            lines.append("CURE PATHS (changes needed for approval):")
            for cp in cure_paths:
                direction = "increase" if cp.change_needed > 0 else "decrease"
                lines.append(f"  → {cp.feature}: {direction} from {cp.current_value:.4g} to {cp.target_value:.4g}")
            lines.append("")

    return "\n".join(lines)
