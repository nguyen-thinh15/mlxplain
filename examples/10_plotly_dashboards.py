#!/usr/bin/env python3
"""Example demonstrating mlxplain's premium interactive Plotly dashboards and HTML dossiers.

Business Scenario: Empowering loan underwriters and risk management executives with gorgeous, interactive glassmorphic web dossier dashboards to audit model decisions.
mlxplain Capability: Generating interactive Plotly-based diagnostic widgets (gauges, waterfalls, heatmaps, grouped bars) and compiling them alongside localized underwriting credit memos into a self-contained, shareable HTML dossier.
Expected Runtime: < 2 seconds.
Required Dependencies: numpy, scikit-learn, matplotlib, plotly (optional), mlxplain.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mlxplain import explain_risk
from mlxplain.core.exporter import export_html

# Try to import plotly and its visual orchestrator optionally
try:
    import plotly

    from mlxplain.visualizations.plotly_charts import plot_report_plotly
except ImportError:
    plotly = None
    plot_report_plotly = None


def main():
    print("=" * 80)
    print("       mlxplain // Premium Interactive Plotly Web Dossiers")
    print("=" * 80)

    # 1. Create a synthetic credit scoring dataset
    print("\n[1/4] Generating synthetic credit dataset...")
    np.random.seed(42)
    n_samples = 300

    # Features: [Income ($k), Debt-to-Income (%), Credit Score (FICO), Months Employed]
    income = np.random.normal(75, 20, n_samples).clip(20, 150)
    dti = np.random.normal(32, 8, n_samples).clip(5, 75)
    credit_score = np.random.normal(680, 45, n_samples).clip(300, 850)
    months_employed = np.random.normal(48, 24, n_samples).clip(0, 240)

    X = np.column_stack([income, dti, credit_score, months_employed])
    feature_names = ["Annual Income ($k)", "Debt-to-Income (%)", "Credit Score (FICO)", "Employment Duration (mths)"]

    # Simulating simple non-linear default risk rules
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if dti[i] > 40 or credit_score[i] < 620 or (months_employed[i] < 12 and income[i] < 45):
            y[i] = 1

    # 2. Train a RandomForestClassifier model
    print("\n[2/4] Training scikit-learn RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    model.fit(X, y)

    # 3. Select a declined applicant (index 1 is a perfect fit)
    declined_idx = np.where(y == 1)[0][1]
    applicant_values = X[declined_idx]
    print(f"\n[3/4] Selected declined applicant at index: {declined_idx}")
    for name, val in zip(feature_names, applicant_values, strict=False):
        print(f"  * {name:<30}: {val:.1f}")

    # 4. Generate credit explanation report
    print("\n[4/4] Generating underwriting explanation report...")
    report = explain_risk(
        model,
        X,
        idx=declined_idx,
        feature_names=feature_names,
        threshold=0.45,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    html_output_path = os.path.join(output_dir, "plotly_dossier.html")

    # Check if Plotly is installed
    if plotly is not None and plot_report_plotly is not None:
        print("\n[✓] Plotly is installed. Compiling dynamic interactive figures...")
        # Attach the interactive Plotly figures to the report object
        report.plotly_figures = plot_report_plotly(report)

        # Export as HTML dossier including plotly scripts
        export_html(report, html_output_path, include_plotly=True)
        print("Success! Standing-out interactive Plotly web dossier exported successfully to:")
        print(f"  - {html_output_path}")
    else:
        print("\n[!] Plotly is not installed. To activate interactive visualizations, run:")
        print("      pip install 'mlxplain-xai[plotly]'")
        print("\nFalling back to self-contained high-DPI static Matplotlib HTML embedding...")

        # Export as standard HTML dossier with embedded base64 matplotlib charts
        export_html(report, html_output_path, include_plotly=False)
        print("Success! Portable static Matplotlib web dossier exported successfully to:")
        print(f"  - {html_output_path}")

    print("\nOpen 'examples/output/plotly_dossier.html' in your browser to experience the dashboard!")
    print("=" * 80)


if __name__ == "__main__":
    main()
