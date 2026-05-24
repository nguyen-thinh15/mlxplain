#!/usr/bin/env python3
"""Example demonstrating mlxplain across all three XAI translators in credit risk.

Business Scenario: Advanced credit risk analysis with a high-dimensional 12-feature dataset, comparing three model types (Logistic Regression, Random Forest, XGBoost) and exporting glassmorphic HTML dashboards.
mlxplain Capability: Translating multi-model pipelines, saving high-resolution vector diagnostics (gauge, waterfall, counterfactuals), and exporting premium standalone HTML credit dossiers.
Expected Runtime: < 5 seconds.
Required Dependencies: numpy, scikit-learn, matplotlib, xgboost (optional), shap (optional), mlxplain.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import uuid
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlxplain import explain_risk

# Check optional dependencies: xgboost and shap
has_deps = importlib.util.find_spec("xgboost") is not None and importlib.util.find_spec("shap") is not None

if has_deps:
    import xgboost as xgb
else:
    xgb = None


def generate_dossier_html(
    report,
    feature_names,
    applicant_values,
    applicant_id,
    model_name,
    xai_paradigm_title,
    xai_paradigm_desc,
    gauge_svg,
    drivers_svg,
    counterfactuals_svg,
) -> str:
    """Generate a premium glassmorphic HTML dossier."""
    if report.probabilities is not None:
        raise NotImplementedError("Advanced HTML dossiers are currently deferred for multi-class reports.")

    decision = report.prediction
    probability = report.probability
    threshold = report.threshold
    summary_text = report.summary

    # Color definitions and badges
    if decision == "Declined":
        status_badge = '<span class="badge badge-declined">DECLINED / HIGH RISK</span>'
        status_border = "border-declined"
        status_text_color = "text-declined"
    else:
        status_badge = '<span class="badge badge-approved">APPROVED / STABLE</span>'
        status_border = "border-approved"
        status_text_color = "text-approved"

    # Identify which features are risk drivers vs. mitigating strengths
    pos_feats = {d.feature: d for d in report.positive_drivers}
    neg_feats = {d.feature: d for d in report.negative_drivers}

    # Generate the applicant feature list HTML
    feature_list_html = []
    # Benchmark values/ranges for each feature for premium aesthetics
    benchmarks = {
        "Annual Income ($k)": "Avg: $80k (Higher is better)",
        "Debt-to-Income (%)": "Benchmark: < 36% (Lower is better)",
        "Credit Score (FICO)": "Prime: > 670 (Higher is better)",
        "Employment Duration (yrs)": "Stable: > 3 yrs (Higher is better)",
        "Savings Balance ($k)": "Reserve: > $10k (Higher is better)",
        "Requested Loan Amount ($k)": "Portfolio Limit: $100k",
        "Loan-to-Value Ratio (%)": "Safe: < 80% (Lower is better)",
        "Open Credit Lines": "Healthy: 4 - 8 lines",
        "On-Time Payment (%)": "Target: > 98% (Higher is better)",
        "Derogatory Public Records": "Target: 0 marks (Lower is better)",
        "Credit Card Utilization (%)": "Target: < 30% (Lower is better)",
        "Years of Credit History": "Established: > 8 yrs (Higher is better)",
    }

    descriptions = {
        "Annual Income ($k)": "Total annual verified gross earnings.",
        "Debt-to-Income (%)": "Percentage of monthly income used to pay debts.",
        "Credit Score (FICO)": "Industry-standard creditworthiness score.",
        "Employment Duration (yrs)": "Continuous employment with current employer.",
        "Savings Balance ($k)": "Liquid savings held in interest-bearing accounts.",
        "Requested Loan Amount ($k)": "Total capital requested in current application.",
        "Loan-to-Value Ratio (%)": "Ratio of the loan amount to asset collateral value.",
        "Open Credit Lines": "Number of currently active credit cards and loans.",
        "On-Time Payment (%)": "Percentage of historical payments made on schedule.",
        "Derogatory Public Records": "Count of bankruptcies, liens, or collections.",
        "Credit Card Utilization (%)": "Percentage of revolving credit limits currently used.",
        "Years of Credit History": "Time elapsed since the applicant's first credit line.",
    }

    for name, val in zip(feature_names, applicant_values, strict=False):
        status_class = ""
        driver_badge = ""

        if name in pos_feats:
            status_class = "feature-row-risk"
            driver_badge = (
                f'<span class="driver-tag tag-risk">Risk Factor (Impact: {pos_feats[name].impact:.3f})</span>'
            )
        elif name in neg_feats:
            status_class = "feature-row-mitigating"
            driver_badge = (
                f'<span class="driver-tag tag-mitigating">Mitigating (Impact: {neg_feats[name].impact:.3f})</span>'
            )

        # Format feature value
        if "%" in name:
            formatted_val = f"{val:.1f}%"
        elif "$" in name or "Income" in name or "Amount" in name or "Balance" in name:
            formatted_val = f"${val:.1f}k"
        elif "Duration" in name or "History" in name or "yrs" in name:
            formatted_val = f"{val:.1f} yrs"
        else:
            formatted_val = f"{int(val)}" if val.is_integer() else f"{val:.1f}"

        feature_list_html.append(f"""
        <div class="feature-card {status_class}">
            <div class="feature-card-header">
                <span class="feature-name">{name}</span>
                <span class="feature-value">{formatted_val}</span>
            </div>
            <div class="feature-description">{descriptions.get(name, "")}</div>
            <div class="feature-footer">
                <span class="feature-benchmark">{benchmarks.get(name, "")}</span>
                {driver_badge}
            </div>
        </div>
        """)

    feature_list_html_str = "\n".join(feature_list_html)

    # Actionable Cure Paths checklist
    cure_paths_html = []
    if report.counterfactuals:
        for idx, cp in enumerate(report.counterfactuals[:4]):
            direction = "increase" if cp.change_needed > 0 else "decrease"
            unit = "%" if "%" in cp.feature else "k" if "$" in cp.feature else ""
            prefix = "$" if "$" in cp.feature and "%" not in cp.feature else ""

            curr_fmt = f"{prefix}{cp.current_value:.1f}{unit}"
            targ_fmt = f"{prefix}{cp.target_value:.1f}{unit}"
            change_fmt = f"{prefix}{abs(cp.change_needed):.1f}{unit}"

            cure_paths_html.append(f"""
            <div class="cure-path-item">
                <div class="cure-checkbox-wrapper">
                    <span class="cure-index">{idx + 1}</span>
                </div>
                <div class="cure-details">
                    <h4>Adjust <strong>{cp.feature}</strong></h4>
                    <p>Request the applicant to <strong>{direction}</strong> this value from {curr_fmt} to <strong>{targ_fmt}</strong> (a shift of {change_fmt}).</p>
                </div>
            </div>
            """)
    else:
        cure_paths_html.append("""
        <div class="no-cure-paths">
            <p>No cure paths required. This application is already approved and meets underwriting criteria.</p>
        </div>
        """)
    cure_paths_html_str = "\n".join(cure_paths_html)

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Risk Intelligence Dossier - ID {applicant_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --panel-bg: rgba(22, 30, 49, 0.65);
            --panel-border: rgba(255, 255, 255, 0.08);
            --panel-shadow: rgba(0, 0, 0, 0.4);

            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;

            --approved: #10b981;
            --approved-bg: rgba(16, 185, 129, 0.1);
            --approved-border: rgba(16, 185, 129, 0.3);

            --declined: #f43f5e;
            --declined-bg: rgba(244, 63, 94, 0.1);
            --declined-border: rgba(244, 63, 94, 0.3);

            --blue-glow: rgba(59, 130, 246, 0.15);
            --accent-blue: #3b82f6;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            line-height: 1.5;
            background-image:
                radial-gradient(at 10% 20%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 90% 10%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
                radial-gradient(at 50% 80%, rgba(244, 63, 94, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header Styling */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 1.5rem;
        }}

        .header-title h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #f8fafc 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-title p {{
            color: var(--text-secondary);
            font-size: 1rem;
            margin-top: 0.25rem;
        }}

        .header-meta {{
            text-align: right;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .header-meta strong {{
            color: var(--text-secondary);
        }}

        /* Navigation between dossiers */
        .dossier-nav {{
            display: flex;
            gap: 10px;
            margin-bottom: 1.5rem;
        }}

        .dossier-nav a {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.4rem 0.8rem;
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            font-size: 0.8rem;
            background: rgba(30, 41, 59, 0.4);
            transition: all 0.2s ease;
        }}

        .dossier-nav a.active {{
            background: var(--accent-blue);
            color: #fff;
            border-color: var(--accent-blue);
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        }}

        .dossier-nav a:hover:not(.active) {{
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
        }}

        /* Dashboard Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}

        @media (min-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 380px 1fr;
            }}
        }}

        /* Glassmorphic Panel Base */
        .panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px var(--panel-shadow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .panel:hover {{
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5), 0 0 15px var(--blue-glow);
        }}

        .panel-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 0.75rem;
            color: var(--text-primary);
        }}

        /* Left Column Content */
        .decision-panel {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid var(--panel-border);
        }}

        .decision-panel.border-declined {{
            border-top: 5px solid var(--declined);
        }}
        .decision-panel.border-approved {{
            border-top: 5px solid var(--approved);
        }}

        .badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            font-weight: 700;
            border-radius: 9999px;
            letter-spacing: 0.05em;
            margin-bottom: 1.25rem;
        }}

        .badge-declined {{
            background-color: var(--declined-bg);
            color: var(--declined);
            border: 1px solid var(--declined-border);
        }}

        .badge-approved {{
            background-color: var(--approved-bg);
            color: var(--approved);
            border: 1px solid var(--approved-border);
        }}

        .metric-value {{
            font-size: 3rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.25rem;
            letter-spacing: -0.03em;
        }}

        .text-declined {{ color: var(--declined); }}
        .text-approved {{ color: var(--approved); }}

        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1.5rem;
        }}

        .gauge-container {{
            width: 100%;
            display: flex;
            justify-content: center;
            margin: 0.5rem 0 1rem 0;
        }}

        .gauge-container svg {{
            width: 100%;
            height: auto;
            max-height: 110px;
        }}

        /* Applicant Feature Cards */
        .features-container {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .feature-card {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            transition: all 0.2s ease;
        }}

        .feature-card:hover {{
            background: rgba(30, 41, 59, 0.7);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .feature-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
        }}

        .feature-name {{
            font-size: 0.9rem;
            color: var(--text-primary);
        }}

        .feature-value {{
            font-size: 1rem;
            color: #fff;
        }}

        .feature-description {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin: 0.25rem 0 0.5rem 0;
        }}

        .feature-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            border-top: 1px dashed rgba(255, 255, 255, 0.05);
            padding-top: 0.4rem;
        }}

        .feature-benchmark {{
            color: var(--text-muted);
        }}

        .driver-tag {{
            font-weight: 600;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
        }}

        .tag-risk {{
            background: rgba(244, 63, 94, 0.12);
            color: var(--declined);
        }}

        .tag-mitigating {{
            background: rgba(16, 185, 129, 0.12);
            color: var(--approved);
        }}

        .feature-card.feature-row-risk {{
            border-left: 3px solid var(--declined);
            background: rgba(244, 63, 94, 0.03);
        }}

        .feature-card.feature-row-mitigating {{
            border-left: 3px solid var(--approved);
            background: rgba(16, 185, 129, 0.03);
        }}

        /* Right Column Panels */
        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .chart-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            background: rgba(15, 23, 42, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.04);
            margin-bottom: 1rem;
        }}

        .chart-card svg {{
            width: 100%;
            height: auto;
            max-height: 380px;
        }}

        /* Actionable Cure Paths */
        .cure-paths-container {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .cure-path-item {{
            display: flex;
            gap: 1.25rem;
            background: rgba(59, 130, 246, 0.03);
            border: 1px solid rgba(59, 130, 246, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.2s ease;
        }}

        .cure-path-item:hover {{
            background: rgba(59, 130, 246, 0.06);
            border-color: rgba(59, 130, 246, 0.25);
            transform: translateX(4px);
        }}

        .cure-index {{
            width: 32px;
            height: 32px;
            background: var(--accent-blue);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            flex-shrink: 0;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        }}

        .cure-details h4 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            color: var(--text-primary);
        }}

        .cure-details p {{
            font-size: 0.88rem;
            color: var(--text-secondary);
        }}

        .cure-details strong {{
            color: #fff;
        }}

        /* Underwriting Memo Terminal Block */
        .memo-terminal {{
            background: #05070c;
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 1.25rem;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            color: #38bdf8;
            overflow-x: auto;
            white-space: pre-wrap;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
        }}

        .memo-terminal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 0.5rem;
            margin-bottom: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .memo-dots {{
            display: flex;
            gap: 6px;
        }}

        .memo-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .dot-r {{ background-color: #ef4444; }}
        .dot-y {{ background-color: #f59e0b; }}
        .dot-g {{ background-color: #10b981; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Dashboard Header -->
        <header>
            <div class="header-title">
                <h1>mlxplain // Credit Underwriting Dossier</h1>
                <p>Advanced Explainable AI Risk Intelligence Dashboard</p>
            </div>
            <div class="header-meta">
                <p>DOSSIER ID: <strong>D-{applicant_id[:8].upper()}</strong></p>
                <p>GENERATED: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
                <p>MODEL: <strong>{model_name}</strong></p>
            </div>
        </header>

        <!-- Dossier Navigation Links -->
        <div class="dossier-nav">
            <a href="logistic_dossier.html" class="{"active" if "Logistic" in model_name else ""}">1. Logistic Regression</a>
            <a href="tree_dossier.html" class="{"active" if "RandomForest" in model_name else ""}">2. Random Forest</a>
            <a href="ensemble_dossier.html" class="{"active" if "XGBoost" in model_name else ""}">3. XGBoost Ensemble</a>
        </div>

        <div class="dashboard-grid">

            <!-- Left Column: Decision & Applicant Profile -->
            <div class="left-column">

                <!-- Decision Summary Card -->
                <div class="panel decision-panel {status_border}">
                    {status_badge}
                    <div class="metric-value {status_text_color}">{probability:.1%}</div>
                    <div class="metric-label">Default Probability</div>

                    <div class="gauge-container">
                        {gauge_svg}
                    </div>

                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Approval Threshold: {threshold:.0%} Default Risk Limit
                    </div>
                </div>

                <!-- Applicant Profile Details -->
                <div class="panel">
                    <div class="panel-title">
                        <span>Applicant Attributes</span>
                        <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">{len(feature_names)} features</span>
                    </div>
                    <div class="features-container">
                        {feature_list_html_str}
                    </div>
                </div>

            </div>

            <!-- Right Column: Charts & In-depth Analytics -->
            <div class="right-column">

                <!-- Underwriting Decision Drivers -->
                <div class="panel">
                    <div class="panel-title">
                        <span>Underwriting Decision Drivers</span>
                        <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Method: {xai_paradigm_title}</span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">
                        {xai_paradigm_desc}
                    </p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">
                        <span style="color: var(--declined); font-weight: 600;">Red factors</span> increase predicted risk towards Default/Decline,
                        while <span style="color: var(--approved); font-weight: 600;">Green factors</span> mitigate risk towards Approval.
                    </p>
                    <div class="chart-card">
                        {drivers_svg}
                    </div>
                </div>

                <!-- Actionable Cure Paths & Counterfactuals -->
                <div class="panel">
                    <div class="panel-title">
                        <span>Strategic Cure Paths (Counterfactuals)</span>
                        <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Paradigms: Minimum Single-Feature Flips</span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">
                        The following attributes can be adjusted to successfully reverse this unfavorable decision.
                        The chart displays the gap between the applicant's current values (<span style="color: #90CAF9; font-weight: 600;">blue</span>)
                        and the minimum target values required for approval (<span style="color: #FFB74D; font-weight: 600;">orange</span>).
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 1.5rem; margin-bottom: 1rem;">
                        <div class="chart-card" style="margin-bottom: 0;">
                            {counterfactuals_svg}
                        </div>
                        <div class="cure-paths-container">
                            {cure_paths_html_str}
                        </div>
                    </div>
                </div>

                <!-- Official Credit Memo Text Output -->
                <div class="panel">
                    <div class="panel-title">
                        <span>Raw Underwriting Memo</span>
                        <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">System Integration Output</span>
                    </div>
                    <div class="memo-terminal-header">
                        <div class="memo-dots">
                            <span class="memo-dot dot-r"></span>
                            <span class="memo-dot dot-y"></span>
                            <span class="memo-dot dot-g"></span>
                        </div>
                        <span>copy_underwrite_memo.sh</span>
                    </div>
                    <pre class="memo-terminal">{summary_text}</pre>
                </div>

            </div>

        </div>
    </div>
</body>
</html>
"""
    return html_content


def read_svg_content(path: str) -> str:
    """Read a saved SVG file and return it as clean inlinable XML/SVG."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Find start of <svg> tag
    svg_start = content.find("<svg")
    if svg_start != -1:
        return content[svg_start:]
    return content


def main():
    print("=" * 70)
    print("mlxplain - High-Dimensional Advanced Credit Underwriting Suite")
    print("=" * 70)

    # 1. Generate high-fidelity synthetic credit dataset with 12 features
    print("Synthesizing 12-feature credit scoring dataset (1,000 applicants)...")
    np.random.seed(42)
    n_samples = 1000

    # Draw feature variables
    income = np.random.normal(80, 25, n_samples).clip(25, 250)  # Annual Income (k$)
    dti = np.random.normal(32, 8, n_samples).clip(5, 65)  # Debt-to-Income (%)
    credit_score = np.random.normal(685, 45, n_samples).clip(450, 850)  # Credit Score (FICO)
    employment_yrs = np.random.normal(6, 4, n_samples).clip(0, 30)  # Employment Duration (yrs)
    savings = np.random.exponential(15, n_samples).clip(0, 150)  # Savings Balance (k$)
    loan_amount = np.random.normal(35, 15, n_samples).clip(5, 100)  # Requested Loan Amount (k$)
    ltv = np.random.normal(78, 12, n_samples).clip(20, 110)  # Loan-to-Value Ratio (%)
    open_lines = np.random.poisson(6, n_samples).clip(1, 20)  # Open Credit Lines
    on_time_pct = (100 - np.random.exponential(2, n_samples)).clip(60, 100)  # On-Time Payment History (%)
    derogatories = np.random.poisson(0.4, n_samples).clip(0, 5)  # Derogatory Public Records
    utilization = np.random.beta(2, 5, n_samples) * 100  # Credit Card Utilization (%)
    credit_history_yrs = np.random.normal(12, 5, n_samples).clip(1, 40)  # Years of Credit History

    X = np.column_stack(
        [
            income,
            dti,
            credit_score,
            employment_yrs,
            savings,
            loan_amount,
            ltv,
            open_lines,
            on_time_pct,
            derogatories,
            utilization,
            credit_history_yrs,
        ]
    )

    feature_names = [
        "Annual Income ($k)",
        "Debt-to-Income (%)",
        "Credit Score (FICO)",
        "Employment Duration (yrs)",
        "Savings Balance ($k)",
        "Requested Loan Amount ($k)",
        "Loan-to-Value Ratio (%)",
        "Open Credit Lines",
        "On-Time Payment (%)",
        "Derogatory Public Records",
        "Credit Card Utilization (%)",
        "Years of Credit History",
    ]

    # Non-linear probability of default logit equation
    # Simulates realistic underwriting thresholds and multi-variable interactions
    logit = -0.5
    logit += (dti - 30) * 0.08
    logit -= (credit_score - 670) * 0.015
    logit -= employment_yrs * 0.12
    logit -= savings * 0.06
    logit += (loan_amount - 30) * 0.02
    logit += (ltv - 75) * 0.04
    logit -= (on_time_pct - 97) * 0.15
    logit += derogatories * 1.3
    logit += (utilization - 30) * 0.05
    logit -= credit_history_yrs * 0.04
    logit -= (income - 75) * 0.012

    # Interactions
    logit += np.where((employment_yrs < 1.5) & (ltv > 90), 1.2, 0.0)
    logit += np.where((dti > 45) & (credit_score < 630), 1.8, 0.0)
    logit += np.where(credit_score < 560, 3.5, 0.0)  # Critically low credit score hit

    prob = 1 / (1 + np.exp(-logit))
    y = (prob >= 0.5).astype(int)

    # 2. Train All Three Models
    print("Training scikit-learn LogisticRegression model...")
    model_lr = LogisticRegression(max_iter=1500, random_state=42)
    model_lr.fit(X, y)

    print("Training scikit-learn RandomForestClassifier model...")
    model_rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    model_rf.fit(X, y)

    model_xgb = None
    if xgb is not None:
        print("Training XGBoost XGBClassifier model...")
        model_xgb = xgb.XGBClassifier(
            n_estimators=30, max_depth=4, learning_rate=0.08, random_state=42, eval_metric="logloss"
        )
        model_xgb.fit(X, y)
    else:
        print("XGBoost is not installed in the current shell context. Skipping XGBoost model training.")

    # 3. Choose the same borderline applicant (index 10 is perfect for all models)
    candidate_idx = 10
    applicant_values = X[candidate_idx]
    applicant_id = str(uuid.uuid4())

    print(f"\n[+] Selected Declined Borderline Applicant at index: {candidate_idx}")
    print(f"[+] Unique Applicant ID: D-{applicant_id[:8].upper()}")
    print("-" * 50)
    for name, val in zip(feature_names, applicant_values, strict=False):
        fmt = (
            f"{val:.1f}%"
            if "%" in name
            else f"${val:.1f}k"
            if "$" in name
            else f"{val:.1f} yrs"
            if "yrs" in name or "History" in name
            else f"{int(val)}"
        )
        print(f"  {name:<30}: {fmt}")
    print("-" * 50)

    # 4. Generate Explanations, Save SVGs and Dossiers for each model
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # List of models to process
    configs = [
        {
            "key": "logistic",
            "name": "Logistic Regression (analytical coefficients XAI)",
            "model": model_lr,
            "prefix": "logistic",
            "title": "Logistic Regression Coefficients",
            "desc": "This waterfall chart shows the raw coefficient contribution (weight * value) for each feature. The counterfactual cure paths are computed using direct **analytical coefficient inversion**, solving mathematically for the exact minimum single-feature shifts needed to cross the decision boundary.",
        },
        {
            "key": "tree",
            "name": "RandomForest Classifier (path split probability XAI)",
            "model": model_rf,
            "prefix": "tree",
            "title": "Random Forest Tree Splits",
            "desc": "This waterfall chart shows the average split-level class probability differences accumulated along all decision paths across all trees in the forest. The counterfactual cure paths are computed using **perturbation-based split-boundary searches**, testing linear shifts until split boundaries are crossed.",
        },
    ]

    if model_xgb is not None:
        configs.append(
            {
                "key": "ensemble",
                "name": "XGBoost Classifier (game-theoretic SHAP XAI)",
                "model": model_xgb,
                "prefix": "ensemble",
                "title": "XGBoost + SHAP Values",
                "desc": "This waterfall chart shows **SHAP (Shapley Additive exPlanations)** values, representing the mathematically rigorous game-theoretic contribution of each feature to the final prediction. The counterfactual cure paths are computed using **iterative perturbation searches** along the highest-impact SHAP feature dimensions.",
            }
        )

    for cfg in configs:
        print(f"\nProcessing {cfg['name']}...")
        print("  - Generating explanation report...")
        report = explain_risk(
            cfg["model"],
            X,
            idx=candidate_idx,
            feature_names=feature_names,
            threshold=0.45,
        )

        # Print the credit risk memo to the console for this model
        print("\n" + "=" * 60)
        print(f"MEMO FOR MODEL: {cfg['key'].upper()}")
        print("=" * 60)
        print(report.summary)
        print("=" * 60 + "\n")

        print("  - Rendering high-resolution vector charts (SVGs)...")
        gauge_path = os.path.join(output_dir, f"{cfg['prefix']}_gauge.svg")
        drivers_path = os.path.join(output_dir, f"{cfg['prefix']}_drivers.svg")
        cf_path = os.path.join(output_dir, f"{cfg['prefix']}_counterfactuals.svg")

        report.figures["gauge"].savefig(gauge_path, format="svg", bbox_inches="tight", transparent=True)
        report.figures["drivers"].savefig(drivers_path, format="svg", bbox_inches="tight", transparent=True)
        report.figures["counterfactuals"].savefig(cf_path, format="svg", bbox_inches="tight", transparent=True)

        # Also save high-resolution PNGs to docs/images/ for README displays to align with the 12-feature datasets
        docs_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")
        os.makedirs(docs_images_dir, exist_ok=True)
        report.figures["gauge"].savefig(
            os.path.join(docs_images_dir, f"{cfg['prefix']}_gauge.jpg"), dpi=150, bbox_inches="tight", transparent=True
        )
        report.figures["drivers"].savefig(
            os.path.join(docs_images_dir, f"{cfg['prefix']}_drivers.jpg"),
            dpi=150,
            bbox_inches="tight",
            transparent=True,
        )
        report.figures["counterfactuals"].savefig(
            os.path.join(docs_images_dir, f"{cfg['prefix']}_counterfactuals.jpg"),
            dpi=150,
            bbox_inches="tight",
            transparent=True,
        )

        print("  - Compiling premium HTML dossier...")
        gauge_svg = read_svg_content(gauge_path)
        drivers_svg = read_svg_content(drivers_path)
        counterfactuals_svg = read_svg_content(cf_path)

        dossier_html = generate_dossier_html(
            report=report,
            feature_names=feature_names,
            applicant_values=applicant_values,
            applicant_id=applicant_id,
            model_name=cfg["name"],
            xai_paradigm_title=cfg["title"],
            xai_paradigm_desc=cfg["desc"],
            gauge_svg=gauge_svg,
            drivers_svg=drivers_svg,
            counterfactuals_svg=counterfactuals_svg,
        )

        dossier_path = os.path.join(output_dir, f"{cfg['prefix']}_dossier.html")
        with open(dossier_path, "w", encoding="utf-8") as f:
            f.write(dossier_html)
        print(f"  - Standalone dossier saved at: {dossier_path}")

    # Create a default dossier.html which symlinks/copies/points to ensemble (or tree/logistic as fallback)
    default_dossier_name = "ensemble_dossier.html" if model_xgb is not None else "tree_dossier.html"
    dossier_path = os.path.join(output_dir, "dossier.html")
    with contextlib.suppress(Exception):
        if os.path.exists(dossier_path):
            os.remove(dossier_path)
        # Create a tiny meta redirect html file
        with open(dossier_path, "w", encoding="utf-8") as f:
            f.write(
                f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url={default_dossier_name}"></head><body>Redirecting to default dashboard...</body></html>'
            )

    print("\n[✓] ALL SUCCESSFUL!")
    print(f"  - Visual SVG vector charts saved in: {output_dir}/")
    print(f"  - Three Standalone Dossier Dashboards saved in: {output_dir}/")
    print("      * logistic_dossier.html")
    print("      * tree_dossier.html")
    print("      * ensemble_dossier.html")
    print("\nOpen 'examples/output/dossier.html' directly in any browser to review the visual dashboards.")
    print("=" * 70)


if __name__ == "__main__":
    main()
