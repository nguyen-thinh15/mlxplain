#!/usr/bin/env python3
"""Example demonstrating mlxplain's FICO and history constraints for counterfactual safety.

Business Scenario: Credit risk compliance regulations requiring that counterfactual cure paths only recommend feasible changes (e.g. keeping credit score <= 850, not changing years of history, and preventing negative income).
mlxplain Capability: Specifying `immutable_features` lists and `feature_bounds` ranges to constrain both analytical (logistic) and perturbation-based counterfactual generators, outputting clear warnings for infeasible constraints.
Expected Runtime: < 1 second.
Required Dependencies: numpy, scikit-learn, matplotlib, mlxplain.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from mlxplain import explain_risk


def main():
    print("=" * 80)
    print("       mlxplain // Feasible & Constrained Counterfactual Compliance")
    print("=" * 80)

    # 1. Create a synthetic credit scoring dataset
    print("\n[1/4] Generating synthetic credit dataset...")
    np.random.seed(42)
    n_samples = 300

    # Features: [Income ($k), Debt-to-Income (%), Credit Score, Years of Credit History]
    income = np.random.normal(70, 20, n_samples).clip(20, 150)
    dti = np.random.normal(35, 10, n_samples).clip(5, 80)
    credit_score = np.random.normal(680, 50, n_samples).clip(300, 850)
    history_yrs = np.random.normal(12, 5, n_samples).clip(1, 40)

    X = np.column_stack([income, dti, credit_score, history_yrs])
    feature_names = ["Income ($k)", "Debt-to-Income (%)", "Credit Score", "Years of Credit History"]

    # Logit function to determine default probability (positive label = Declined)
    logit = (dti * 0.08) - (credit_score - 600) * 0.015 - (history_yrs * 0.1) - (income * 0.01) + 2.5
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)

    # 2. Train a Logistic Regression model
    print("\n[2/4] Training scikit-learn LogisticRegression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    # 3. Select a declined applicant (default probability >= threshold)
    declined_idx = np.where(y == 1)[0][0]
    applicant_values = X[declined_idx]
    print(f"\n[3/4] Selected declined applicant at index: {declined_idx}")
    for name, val in zip(feature_names, applicant_values, strict=False):
        print(f"  * {name:<30}: {val:.1f}")

    # 4. Generate explanations with and without constraints
    print("\n[4/4] Generating counterfactual cure paths side-by-side...")

    # A. Unconstrained explanation (default)
    print("  - Running UNCONSTRAINED search...")
    report_unconstrained = explain_risk(
        model,
        X,
        idx=declined_idx,
        feature_names=feature_names,
        threshold=0.45,
    )

    # B. Constrained explanation
    # Constraints:
    # 1. Years of Credit History is completely immutable (cannot be changed!)
    # 2. Credit Score cannot exceed the industry limit of 850
    # 3. Income cannot decrease below $35k
    print("  - Running CONSTRAINED search...")
    immutable_features = ["Years of Credit History"]
    feature_bounds = {
        "Credit Score": [300.0, 850.0],
        "Income ($k)": [35.0, 200.0],
        "Debt-to-Income (%)": [0.0, 100.0],
    }

    report_constrained = explain_risk(
        model,
        X,
        idx=declined_idx,
        feature_names=feature_names,
        threshold=0.45,
        immutable_features=immutable_features,
        feature_bounds=feature_bounds,
    )

    print("\n--- UNCONSTRAINED Cure Paths Recommended ---")
    for cf in report_unconstrained.counterfactuals:
        print(
            f"  → Adjust {cf.feature:<30}: shift {cf.current_value:.1f} to {cf.target_value:.1f} (change: {cf.change_needed:+.1f})"
        )

    print("\n--- CONSTRAINED (Feasible/Compliant) Cure Paths Recommended ---")
    if not report_constrained.counterfactuals:
        print("  * Warning: No feasible counterfactual path exists under these constraints!")
    for cf in report_constrained.counterfactuals:
        print(
            f"  ✓ Adjust {cf.feature:<30}: shift {cf.current_value:.1f} to {cf.target_value:.1f} (change: {cf.change_needed:+.1f})"
        )

    # Save double counterfactual comparison charts
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    docs_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images", "examples")
    os.makedirs(docs_images_dir, exist_ok=True)

    # Plot comparisons using standard matplotlib subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Draw unconstrained counterfactuals
    if report_unconstrained.counterfactuals:
        cfs = report_unconstrained.counterfactuals[:4]
        names = [c.feature for c in cfs]
        curr = [c.current_value for c in cfs]
        targ = [c.target_value for c in cfs]
        y_pos = range(len(names))
        ax1.barh([y - 0.15 for y in y_pos], curr, height=0.3, color="#90CAF9", label="Current")
        ax1.barh([y + 0.15 for y in y_pos], targ, height=0.3, color="#EF5350", label="Target (Unconstrained)")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("Value")
        ax1.set_title("Unconstrained Cure Recommendations (Tries to change History)")
        ax1.legend()
        ax1.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No counterfactuals", ha="center", va="center")

    # Draw constrained counterfactuals
    if report_constrained.counterfactuals:
        cfs = report_constrained.counterfactuals[:4]
        names = [c.feature for c in cfs]
        curr = [c.current_value for c in cfs]
        targ = [c.target_value for c in cfs]
        y_pos = range(len(names))
        ax2.barh([y - 0.15 for y in y_pos], curr, height=0.3, color="#90CAF9", label="Current")
        ax2.barh([y + 0.15 for y in y_pos], targ, height=0.3, color="#66BB6A", label="Target (Constrained)")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Value")
        ax2.set_title("Compliant Cure Recommendations (History Locked, FICO Safe)")
        ax2.legend()
        ax2.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No feasible constrained paths found", ha="center", va="center")

    fig.suptitle(
        f"Counterfactual Constraint Feasibility Comparison (Applicant Row {declined_idx})",
        fontsize=12,
        fontweight="bold",
    )

    comp_output_path = os.path.join(output_dir, "constrained_counterfactuals.jpg")
    comp_docs_path = os.path.join(docs_images_dir, "constrained_counterfactuals.jpg")

    fig.savefig(comp_output_path, dpi=150, bbox_inches="tight")
    fig.savefig(comp_docs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n[✓] Visual chart comparisons successfully saved to:")
    print(f"  - {comp_output_path}")
    print(f"  - {comp_docs_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
