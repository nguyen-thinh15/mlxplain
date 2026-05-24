#!/usr/bin/env python3
"""Example demonstrating model-agnostic explanation consistency across three different ML models.

Business Scenario: Credit underwriting compliance audits where risk explanation consistency must be verified side-by-side across multiple machine learning architectures.
mlxplain Capability: Translating and comparing explanations across diverse model structures (Logistic Regression, Decision Trees, and XGBoost/Random Forest Ensembles) using a unified explanation schema, outputting a composite 1x3 comparative waterfall chart.
Expected Runtime: < 3 seconds.
Required Dependencies: numpy, scikit-learn, matplotlib, xgboost (optional), shap (optional), mlxplain.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain import explain_risk

# Try to import optional dependencies: xgboost and shap
try:
    import shap
    import xgboost as xgb
except ImportError:
    xgb = None
    shap = None


def draw_drivers_waterfall_to_ax(ax, positive_drivers, negative_drivers, title):
    """Draw a horizontal waterfall drivers chart to a specific Matplotlib axis."""
    # Combine and sort by impact magnitude
    all_drivers = sorted(positive_drivers + negative_drivers, key=lambda d: d.impact, reverse=True)
    if not all_drivers:
        ax.text(0.5, 0.5, "No drivers to display", ha="center", va="center")
        return

    # Use first 8 highest impact features to prevent overcrowded charts
    all_drivers = all_drivers[:8]

    names = [d.feature for d in all_drivers]
    impacts = [d.impact if d.direction in ("positive", "tích cực") else -d.impact for d in all_drivers]
    colors = ["#F44336" if d.direction in ("positive", "tích cực") else "#4CAF50" for d in all_drivers]

    y_pos = range(len(names))
    ax.barh(y_pos, impacts, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature Impact", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def main():
    print("=" * 80)
    print("       mlxplain // Side-by-Side Multi-Model Explanation Comparison")
    print("=" * 80)

    # 1. Create a synthetic credit scoring dataset
    print("\n[1/5] Generating synthetic credit dataset...")
    np.random.seed(42)
    n_samples = 300

    # Features: [Income (k$), Debt-to-Income (%), Credit Score, Months Employed]
    income = np.random.normal(70, 20, n_samples).clip(20, 150)
    dti = np.random.normal(35, 10, n_samples).clip(5, 80)
    credit_score = np.random.normal(680, 50, n_samples).clip(300, 850)
    months_employed = np.random.normal(48, 24, n_samples).clip(0, 240)

    X = np.column_stack([income, dti, credit_score, months_employed])
    feature_names = ["Income (k$)", "Debt-to-Income (%)", "Credit Score", "Months Employed"]

    # Complex rule classification
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if dti[i] > 38 or credit_score[i] < 630 or (months_employed[i] < 12 and income[i] < 50):
            y[i] = 1

    # 2. Train three models (Logistic Regression, Decision Tree, and Random Forest/XGBoost)
    print("\n[2/5] Training machine learning models...")
    print("  - Fitting LogisticRegression...")
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X, y)

    print("  - Fitting DecisionTreeClassifier...")
    model_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    model_dt.fit(X, y)

    # Use XGBoost if optional extra is installed, otherwise fall back to RandomForest
    if xgb is not None and shap is not None:
        print("  - Fitting XGBoost XGBClassifier (shap-compatible)...")
        model_ensemble = xgb.XGBClassifier(
            n_estimators=15, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss"
        )
        model_ensemble.fit(X, y)
        ensemble_label = "XGBoost Ensemble"
    else:
        print("  - XGBoost/SHAP not available. Fitting RandomForestClassifier as fallback...")
        model_ensemble = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        model_ensemble.fit(X, y)
        ensemble_label = "Random Forest"

    # 3. Select a borderline-declined applicant (DTI=45.2, CS=610, etc.)
    # Applicant index 0 or first high-risk index
    high_risk_idx = np.where(y == 1)[0][0]
    applicant_values = X[high_risk_idx]
    print(f"\n[3/5] Selected borderline declined applicant at index: {high_risk_idx}")
    for name, val in zip(feature_names, applicant_values, strict=False):
        print(f"  * {name:<20}: {val:.1f}")

    # 4. Generate explanations under all three models
    print("\n[4/5] Computing explanation reports side-by-side...")
    report_lr = explain_risk(model_lr, X, idx=high_risk_idx, feature_names=feature_names, threshold=0.45)
    report_dt = explain_risk(model_dt, X, idx=high_risk_idx, feature_names=feature_names, threshold=0.45)
    report_ens = explain_risk(model_ensemble, X, idx=high_risk_idx, feature_names=feature_names, threshold=0.45)

    print("\n--- Model Predictions Comparison ---")
    print(f"  * Logistic Regression default prob: {report_lr.probability:.1%}")
    print(f"  * Decision Tree default prob:       {report_dt.probability:.1%}")
    print(f"  * {ensemble_label} default prob:       {report_ens.probability:.1%}")

    # 5. Build 1x3 Composite Grid using matplotlib's GridSpec
    print("\n[5/5] Building stitched horizontal composite drivers chart...")
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    draw_drivers_waterfall_to_ax(ax0, report_lr.positive_drivers, report_lr.negative_drivers, "Logistic Regression")
    draw_drivers_waterfall_to_ax(ax1, report_dt.positive_drivers, report_dt.negative_drivers, "Decision Tree")
    draw_drivers_waterfall_to_ax(ax2, report_ens.positive_drivers, report_ens.negative_drivers, ensemble_label)

    fig.suptitle(
        f"Risk Driver Comparison Side-by-Side (Applicant Row {high_risk_idx})", y=1.02, fontsize=12, fontweight="bold"
    )

    # Save composite outputs
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    docs_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images", "examples")
    os.makedirs(docs_images_dir, exist_ok=True)

    # Save to both paths
    comp_output_path = os.path.join(output_dir, "model_comparison_grid.jpg")
    comp_docs_path = os.path.join(docs_images_dir, "model_comparison_grid.jpg")

    fig.savefig(comp_output_path, dpi=150, bbox_inches="tight")
    fig.savefig(comp_docs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Success! Composite comparison image generated and saved to:")
    print(f"  - {comp_output_path}")
    print(f"  - {comp_docs_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
