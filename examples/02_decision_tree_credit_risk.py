#!/usr/bin/env python3
"""Example demonstrating mlxplain with a Tree-Based model in credit risk.

This example:
1. Generates a synthetic credit underwriting dataset.
2. Trains a scikit-learn DecisionTreeClassifier model.
3. Explains a high-risk applicant prediction using `explain_risk()`.
4. Outputs the human-readable credit decision memo.
5. Saves visual charts (gauge, waterfall, counterfactuals) to the output folder.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mlxplain import explain_risk


def main():
    # 1. Create a synthetic credit scoring dataset
    print("Generating synthetic credit dataset...")
    np.random.seed(42)
    n_samples = 200

    # Features: [Income (k$), Debt-to-Income (%), Credit Score, Months Employed]
    income = np.random.normal(70, 20, n_samples).clip(20, 150)
    dti = np.random.normal(35, 10, n_samples).clip(5, 80)
    credit_score = np.random.normal(680, 50, n_samples).clip(300, 850)
    months_employed = np.random.normal(48, 24, n_samples).clip(0, 240)

    X = np.column_stack([income, dti, credit_score, months_employed])
    feature_names = ["Income (k$)", "Debt-to-Income (%)", "Credit Score", "Months Employed"]

    # Simple tree-like rule classification (positive label = Declined / high risk of default)
    # High DTI or low credit score or low months employed -> high probability of default (y=1)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if dti[i] > 42 or credit_score[i] < 630 or (months_employed[i] < 12 and income[i] < 50):
            y[i] = 1

    # 2. Train a Decision Tree model
    print("Training scikit-learn DecisionTreeClassifier model...")
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    # 3. Select an applicant who gets declined
    declined_indices = np.where(y == 1)[0]
    high_risk_idx = declined_indices[0]
    applicant_values = X[high_risk_idx]
    print(f"\nSelected declined applicant at index {high_risk_idx}:")
    for name, val in zip(feature_names, applicant_values, strict=False):
        print(f"  {name}: {val:.1f}")

    # 4. Generate credit risk explanation report
    print("\nGenerating credit risk explanation memo and charts...")
    report = explain_risk(
        model,
        X,
        idx=high_risk_idx,
        feature_names=feature_names,
        threshold=0.50,
    )

    # 5. Print the business-friendly credit underwriting memo
    print("\n" + "=" * 60)
    print("CREDIT UNDERWRITING MEMO (Tree-Based Explanation)")
    print("=" * 60)
    print(report.summary)
    print("=" * 60 + "\n")

    # 6. Save the generated charts
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save each chart to the examples/output/ folder
    report.figures["gauge"].savefig(os.path.join(output_dir, "tree_gauge.png"), dpi=150)
    report.figures["drivers"].savefig(os.path.join(output_dir, "tree_drivers.png"), dpi=150)
    report.figures["counterfactuals"].savefig(os.path.join(output_dir, "tree_counterfactuals.png"), dpi=150)

    print("Success! Matplotlib figures successfully saved to:")
    print(f"  - {os.path.join(output_dir, 'tree_gauge.png')}")
    print(f"  - {os.path.join(output_dir, 'tree_drivers.png')}")
    print(f"  - {os.path.join(output_dir, 'tree_counterfactuals.png')}")


if __name__ == "__main__":
    main()
