"""Premium HTML report exporter for explanation reports."""

from __future__ import annotations

import base64
import io
from typing import Any

from mlxplain.core.report import ExplanationReport


def _fig_to_base64(fig: Any) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    # Save with high DPI for premium display on high-res monitors
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str


def export_html(
    report: ExplanationReport,
    output_path: str,
    include_plotly: bool = False,
) -> None:
    """Export an ExplanationReport as a premium, self-contained HTML dashboard.

    Matplotlib figures are automatically converted to embedded base64 images,
    ensuring a completely portable single-file layout.
    If include_plotly is True and report.plotly_figures is available, interactive
    Plotly figures are embedded instead of static matplotlib charts, loading the
    PlotlyJS library from CDN in the document head.

    For PDF export, the HTML report is print-ready; open it in a modern browser
    and select 'Print to PDF' (configured with print media CSS margins).
    """
    language = report.domain_output.get("language", "en")
    is_vi = language == "vi"

    # Localization strings
    labels = {
        "title": "mlxplain Decision Dossier" if not is_vi else "Hồ sơ Quyết định mlxplain",
        "prediction": "Prediction Decision" if not is_vi else "Quyết định Dự báo",
        "probability": "Predicted Probability" if not is_vi else "Xác suất Dự báo",
        "threshold": "Decision Threshold" if not is_vi else "Ngưỡng Quyết định",
        "drivers": "Feature Drivers" if not is_vi else "Nhân tố Tác động",
        "counterfactuals": "Counterfactual Cure Paths" if not is_vi else "Đường khắc phục Tối thiểu",
        "summary": "Explanation Summary" if not is_vi else "Tóm tắt Giải thích",
        "current_val": "Current" if not is_vi else "Hiện tại",
        "required_val": "Required" if not is_vi else "Cần thiết",
        "change_needed": "Adjustment" if not is_vi else "Điều chỉnh",
        "no_drivers": "No drivers to display." if not is_vi else "Không có nhân tố nào.",
        "no_cfs": "No counterfactual adjustment required or feasible."
        if not is_vi
        else "Không cần hoặc không thể thực hiện điều chỉnh khắc phục.",
        "status_unfavorable": "Requires Adjustment" if not is_vi else "Cần Điều chỉnh",
        "status_favorable": "Favorable Outcome" if not is_vi else "Kết quả Tốt",
        "feature": "Feature" if not is_vi else "Đặc trưng",
        "impact": "Impact" if not is_vi else "Mức tác động",
        "direction": "Direction" if not is_vi else "Chiều tác động",
        "value": "Value" if not is_vi else "Giá trị",
    }

    # Format the decision details
    pred_str = str(report.prediction)
    prob_str = f"{report.probability:.3f}"
    thresh_str = f"{report.threshold:.2f}"

    # Determine status class for coloring
    is_favorable = report.probability < report.threshold
    status_text = labels["status_favorable"] if is_favorable else labels["status_unfavorable"]
    status_class = "status-favorable" if is_favorable else "status-unfavorable"

    # Embed matplotlib charts as base64 images
    embedded_images = {}
    for name, fig in report.figures.items():
        embedded_images[name] = _fig_to_base64(fig)

    # Handle Plotly embedding
    plotly_divs = {}
    plotly_header_scripts = ""
    if include_plotly and report.plotly_figures:
        try:
            import plotly.io as pio

            plotly_header_scripts = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            for name, fig in report.plotly_figures.items():
                # Renders a standard div with include_plotlyjs=False to avoid duplications
                div_str = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                plotly_divs[name] = div_str
        except ImportError:
            # Fall back to matplotlib if plotly cannot be imported
            include_plotly = False

    # Standard layout divs generator
    def get_chart_html(chart_name: str) -> str:
        if include_plotly and chart_name in plotly_divs:
            return plotly_divs[chart_name]
        elif chart_name in embedded_images:
            return (
                f'<img class="chart-img" '
                f'src="data:image/png;base64,{embedded_images[chart_name]}" '
                f'alt="{chart_name} chart" />'
            )
        return f'<div class="no-chart">{labels["no_drivers"]}</div>'

    # Build Feature Drivers Table HTML
    drivers_tbody = ""
    all_drivers = sorted(
        report.positive_drivers + report.negative_drivers,
        key=lambda d: d.impact,
        reverse=True,
    )
    if all_drivers:
        for d in all_drivers:
            impact_sign = "+" if d.direction in ("positive", "tích cực") else "-"
            dir_color = "impact-positive" if impact_sign == "+" else "impact-negative"
            drivers_tbody += f"""
            <tr>
                <td><strong>{d.feature}</strong></td>
                <td>{d.value:.4f}</td>
                <td class="{dir_color}">{impact_sign}{d.impact:.4f}</td>
                <td><span class="badge {dir_color}">{d.direction.upper()}</span></td>
            </tr>
            """
    else:
        drivers_tbody = f'<tr><td colspan="4" class="text-center">{labels["no_drivers"]}</td></tr>'

    # Build Counterfactuals Table HTML
    cf_tbody = ""
    if report.counterfactuals:
        for c in report.counterfactuals:
            change_sign = "+" if c.change_needed > 0 else ""
            cf_tbody += f"""
            <tr>
                <td><strong>{c.feature}</strong></td>
                <td>{c.current_value:.4f}</td>
                <td>{c.target_value:.4f}</td>
                <td class="impact-negative">{change_sign}{c.change_needed:.4f}</td>
            </tr>
            """
    else:
        cf_tbody = f'<tr><td colspan="4" class="text-center">{labels["no_cfs"]}</td></tr>'

    # Multi-class rendering block if probabilities dict is active
    multi_class_section = ""
    if report.probabilities is not None:
        sorted_probs = sorted(report.probabilities.items(), key=lambda x: x[1], reverse=True)
        prob_rows = ""
        for cls, prob in sorted_probs:
            is_max = prob == max(report.probabilities.values())
            highlight = "class-winner" if is_max else ""
            prob_rows += f"""
            <div class="class-prob-row {highlight}">
                <span class="class-name">{cls}</span>
                <div class="class-prob-bar-bg">
                    <div class="class-prob-bar-fill" style="width: {prob * 100}%;"></div>
                </div>
                <span class="class-prob-val">{prob:.1%}</span>
            </div>
            """
        multi_class_section = f"""
        <div class="card">
            <h3>Class Probabilities</h3>
            <div class="class-prob-container">
                {prob_rows}
            </div>
        </div>
        """

    # Structured Domain Memo output (if it has customized tables or groups)
    memo_section = ""
    if report.summary:
        memo_section = f"""
        <div class="card summary-card">
            <h2>{labels["summary"]}</h2>
            <div class="memo-content">
                {report.summary.replace(chr(10), "<br>")}
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{labels["title"]} - {pred_str}</title>
    {plotly_header_scripts}
    <style>
        /* CSS Premium Design System */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

        :root {{
            --bg-color: #0B0F19;
            --surface-color: #161C2C;
            --surface-accent: #232C43;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
            --accent-green: #10B981;
            --accent-green-bg: rgba(16, 185, 129, 0.15);
            --accent-red: #EF4444;
            --accent-red-bg: rgba(239, 68, 68, 0.15);
            --accent-blue: #3B82F6;
            --shadow-premium: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --border-glow: rgba(59, 130, 246, 0.15);
        }}

        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-color: #F8FAFC;
                --surface-color: #FFFFFF;
                --surface-accent: #F1F5F9;
                --text-main: #0F172A;
                --text-muted: #64748B;
                --border-glow: rgba(59, 130, 246, 0.05);
            }}
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }}

        header {{
            max-width: 1200px;
            margin: 0 auto 2rem auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--surface-accent);
            padding-bottom: 1.5rem;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }}

        header h1 {{
            font-size: 2.2rem;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, var(--text-main) 30%, var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-pill {{
            font-size: 0.85rem;
            background: var(--surface-accent);
            color: var(--text-muted);
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-weight: 500;
            border: 1px solid var(--border-glow);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 900px) {{
            .container {{
                grid-template-columns: 1fr;
            }}
        }}

        .main-column {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .side-column {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .card {{
            background: var(--surface-color);
            border-radius: 16px;
            padding: 1.8rem;
            box-shadow: var(--shadow-premium);
            border: 1px solid var(--surface-accent);
            transition: transform 0.2s, border-color 0.2s;
        }}

        .card:hover {{
            border-color: var(--border-glow);
        }}

        .card h2, .card h3 {{
            margin-bottom: 1.2rem;
            font-size: 1.3rem;
            border-left: 4px solid var(--accent-blue);
            padding-left: 0.6rem;
            color: var(--text-main);
        }}

        /* Status Panels */
        .status-panel {{
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }}

        .status-metric {{
            flex: 1;
            text-align: center;
            padding: 1rem;
            background: var(--surface-accent);
            border-radius: 12px;
        }}

        .status-metric .metric-lbl {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}

        .status-metric .metric-val {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
        }}

        .status-badge-container {{
            text-align: center;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.4rem;
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
        }}

        .status-favorable {{
            background: var(--accent-green-bg);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .status-unfavorable {{
            background: var(--accent-red-bg);
            color: var(--accent-red);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
        }}

        th, td {{
            padding: 0.8rem 1rem;
            text-align: left;
            font-size: 0.9rem;
        }}

        th {{
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--surface-accent);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            border-bottom: 1px solid var(--surface-accent);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .badge {{
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }}

        .impact-positive {{
            color: var(--accent-red);
        }}

        .badge.impact-positive {{
            background: var(--accent-red-bg);
            color: var(--accent-red);
        }}

        .impact-negative {{
            color: var(--accent-green);
        }}

        .badge.impact-negative {{
            background: var(--accent-green-bg);
            color: var(--accent-green);
        }}

        .text-center {{
            text-align: center;
        }}

        /* Charts & Visualizations */
        .chart-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }}

        .no-chart {{
            padding: 3rem;
            text-align: center;
            color: var(--text-muted);
            background: var(--surface-accent);
            border-radius: 8px;
            font-size: 0.9rem;
        }}

        /* Class probabilities */
        .class-prob-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.8rem;
            font-size: 0.85rem;
            padding: 0.4rem;
            border-radius: 6px;
        }}

        .class-prob-row.class-winner {{
            background: rgba(59, 130, 246, 0.08);
            font-weight: 600;
        }}

        .class-name {{
            width: 80px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .class-prob-bar-bg {{
            flex: 1;
            height: 8px;
            background: var(--surface-accent);
            border-radius: 4px;
            overflow: hidden;
        }}

        .class-prob-bar-fill {{
            height: 100%;
            background: var(--accent-blue);
            border-radius: 4px;
        }}

        .class-prob-val {{
            width: 45px;
            text-align: right;
        }}

        /* Memo Summary Content */
        .memo-content {{
            font-size: 0.95rem;
            color: var(--text-main);
            white-space: pre-wrap;
            background: var(--surface-accent);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px dashed rgba(156, 163, 175, 0.3);
            font-family: inherit;
        }}

        /* Page viewports print styles */
        @media print {{
            body {{
                background: white;
                color: black;
                padding: 0;
            }}
            :root {{
                --bg-color: #FFFFFF;
                --surface-color: #FFFFFF;
                --surface-accent: #E2E8F0;
                --text-main: #000000;
                --text-muted: #475569;
                --shadow-premium: none;
            }}
            .card {{
                box-shadow: none;
                border: 1px solid #CBD5E1;
                page-break-inside: avoid;
            }}
            header h1 {{
                background: none;
                -webkit-text-fill-color: initial;
                color: black;
            }}
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>{labels["title"]}</h1>
            <p class="brand-pill">mlxplain v{report.domain_output.get("version", "0.2.0")}</p>
        </div>
        <div class="status-badge-container {status_class}">
            {status_text}
        </div>
    </header>

    <div class="container">

        <div class="main-column">

            <!-- Decision info card -->
            <div class="card">
                <div class="status-panel">
                    <div class="status-metric">
                        <div class="metric-lbl">{labels["prediction"]}</div>
                        <div class="metric-val" style="color: var(--accent-blue);">{pred_str}</div>
                    </div>
                    <div class="status-metric">
                        <div class="metric-lbl">{labels["probability"]}</div>
                        <div class="metric-val">{prob_str}</div>
                    </div>
                    <div class="status-metric">
                        <div class="metric-lbl">{labels["threshold"]}</div>
                        <div class="metric-val">{thresh_str}</div>
                    </div>
                </div>
            </div>

            <!-- Figures Section -->
            <div class="card">
                <h2>Visual Insights</h2>
                <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                    <div>
                        {get_chart_html("gauge" if report.probabilities is None else "probabilities")}
                    </div>
                    <div>
                        {get_chart_html("drivers")}
                    </div>
                    {
        f"<div>{get_chart_html('heatmap')}</div>" if "heatmap" in report.figures or "heatmap" in plotly_divs else ""
    }
                    <div>
                        {get_chart_html("counterfactuals")}
                    </div>
                </div>
            </div>

            {memo_section}

        </div>

        <div class="side-column">

            {multi_class_section}

            <!-- Drivers detailed list -->
            <div class="card">
                <h2>{labels["drivers"]}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>{labels["feature"]}</th>
                            <th>{labels["value"]}</th>
                            <th>{labels["impact"]}</th>
                            <th>{labels["direction"]}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {drivers_tbody}
                    </tbody>
                </table>
            </div>

            <!-- Counterfactuals detailed list -->
            <div class="card">
                <h2>{labels["counterfactuals"]}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>{labels["feature"]}</th>
                            <th>{labels["current_val"]}</th>
                            <th>{labels["required_val"]}</th>
                            <th>{labels["change_needed"]}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cf_tbody}
                    </tbody>
                </table>
            </div>

        </div>

    </div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
