"""Credit memo text generation templates."""

from __future__ import annotations

from mlxplain.core.report import ExplanationReport


def generate_credit_memo(report: ExplanationReport) -> str:
    """Generate a human-readable credit memo from an enriched report with normalized impact percentages."""
    domain = report.domain_output
    decision = domain.get("decision", report.prediction)
    risk_factors = domain.get("risk_factors", [])
    mitigating_factors = domain.get("mitigating_factors", [])
    cure_paths = domain.get("cure_paths", [])
    language = domain.get("language", "en")

    # Calculate total impacts per group for normalization
    total_risk_impact = sum(rf.impact for rf in risk_factors)
    total_mitigating_impact = sum(mf.impact for mf in mitigating_factors)

    is_multiclass = report.probabilities is not None

    if language == "vi":
        if is_multiclass:
            lines = [
                f"QUYẾT ĐỊNH PHÂN HẠNG TÍN DỤNG: {decision}",
                f"Xác suất của Hạng {report.prediction}: {report.probability:.1%}",
                "",
            ]
        else:
            lines = [
                f"QUYẾT ĐỊNH TÍN DỤNG: {decision}",
                f"Xác suất Nợ xấu: {report.probability:.1%} (ngưỡng: {report.threshold:.1%})",
                "",
            ]

        if risk_factors:
            if is_multiclass:
                lines.append("CÁC YẾU TỐ ĐỊNH HÌNH HẠNG TÍN DỤNG (KEY FACTORS FOR ASSIGNED GRADE):")
            else:
                lines.append("YẾU TỐ RỦI RO (RISK FACTORS):")
            for rf in risk_factors:
                rf_pct = (rf.impact / total_risk_impact * 100) if total_risk_impact > 0 else 0.0
                lines.append(
                    f"  - {rf.feature}: {rf.value:.4g} (mức độ ảnh hưởng: chiếm {rf_pct:.1f}% tổng mức rủi ro)"
                )
            lines.append("")

        if mitigating_factors:
            if is_multiclass:
                lines.append("YẾU TỐ DI DỊCH HẠNG (DRIVERS PUSHING AWAY FROM ASSIGNED GRADE):")
            else:
                lines.append("YẾU TỐ GIẢM THIỂU RỦI RO (MITIGATING FACTORS):")
            for mf in mitigating_factors:
                mf_pct = (mf.impact / total_mitigating_impact * 100) if total_mitigating_impact > 0 else 0.0
                lines.append(
                    f"  + {mf.feature}: {mf.value:.4g} (mức độ ảnh hưởng: chiếm {mf_pct:.1f}% tổng mức giảm thiểu)"
                )
            lines.append("")

        if cure_paths:
            if is_multiclass:
                target_label = domain.get("target_class_label", "hạng mục tiêu")
                lines.append(f"PHƯƠNG ÁN KHẮC PHỤC ĐỂ ĐẠT {target_label} (CURE PATHS FOR TARGET GRADE):")
            else:
                lines.append("PHƯƠNG ÁN KHẮC PHỤC (CURE PATHS - thay đổi cần thiết để được duyệt):")
            for cp in cure_paths:
                direction = "tăng" if cp.change_needed > 0 else "giảm"
                to_word = "lên" if direction == "tăng" else "xuống"
                lines.append(f"  → {cp.feature}: {direction} từ {cp.current_value:.4g} {to_word} {cp.target_value:.4g}")
            lines.append("")
    else:
        if is_multiclass:
            lines = [
                f"CREDIT GRADE DECISION: {decision}",
                f"Probability of Grade {report.prediction}: {report.probability:.1%}",
                "",
            ]
        else:
            lines = [
                f"CREDIT DECISION: {decision}",
                f"Default Probability: {report.probability:.1%} (threshold: {report.threshold:.1%})",
                "",
            ]

        if risk_factors:
            if is_multiclass:
                lines.append("KEY FACTORS FOR ASSIGNED GRADE:")
            else:
                lines.append("RISK FACTORS:")
            for rf in risk_factors:
                rf_pct = (rf.impact / total_risk_impact * 100) if total_risk_impact > 0 else 0.0
                lines.append(f"  - {rf.feature}: {rf.value:.4g} (impact: {rf_pct:.1f}% of risk impact)")
            lines.append("")

        if mitigating_factors:
            if is_multiclass:
                lines.append("DRIVERS PUSHING AWAY FROM ASSIGNED GRADE:")
            else:
                lines.append("MITIGATING FACTORS:")
            for mf in mitigating_factors:
                mf_pct = (mf.impact / total_mitigating_impact * 100) if total_mitigating_impact > 0 else 0.0
                lines.append(f"  + {mf.feature}: {mf.value:.4g} (impact: {mf_pct:.1f}% of mitigating impact)")
            lines.append("")

        if cure_paths:
            if is_multiclass:
                target_label = domain.get("target_class_label", "target grade")
                lines.append(f"CURE PATHS FOR TARGET GRADE ({target_label}):")
            else:
                lines.append("CURE PATHS (changes needed for approval):")
            for cp in cure_paths:
                direction = "increase" if cp.change_needed > 0 else "decrease"
                lines.append(f"  → {cp.feature}: {direction} from {cp.current_value:.4g} to {cp.target_value:.4g}")
            lines.append("")

    return "\n".join(lines)
