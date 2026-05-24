#!/usr/bin/env python3
"""Example demonstrating mlxplain with an Ensemble Boosting model (XGBoost) in credit risk.

This example:
1. Generates a synthetic credit underwriting dataset.
2. Trains an XGBoost XGBClassifier.
3. Explains a high-risk applicant prediction using `explain_risk()`.
4. Uses SHAP values internally (via EnsembleTranslator) to extract and rank feature drivers.
5. Outputs the human-readable credit decision memo.
6. Saves visual charts (gauge, waterfall, counterfactuals) to the output folder.
"""

from __future__ import annotations

import os
import numpy as np
from mlxplain import explain_risk

# Try to import xgboost
try:
    import xgboost as xgb
except ImportError:
    xgb = None


def main():
    if xgb is None:
        print("XGBoost is not installed. To run this ensemble example, install it via:")
        print("  pip install xgboost")
        print("\nAlternatively, run the other examples (Logistic Regression or Decision Tree).")
        return

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

    # Non-linear logit function to determine default probability (y=1 for high default risk)
    # XGBoost is excellent for learning such non-linear boundaries.
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Default risk is high if DTI is high and credit score is low, OR if credit score is critically low
        if (dti[i] > 38 and credit_score[i] < 660) or (credit_score[i] < 580) or (income[i] < 35 and dti[i] > 45):
            y[i] = 1

    # 2. Train an XGBoost model
    print("Training XGBoost XGBClassifier model...")
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X, y)

    # 3. Select a high-risk applicant who gets declined
    declined_indices = np.where(y == 1)[0]
    high_risk_idx = declined_indices[0]
    applicant_values = X[high_risk_idx]
    print(f"\nSelected declined applicant at index {high_risk_idx}:")
    for name, val in zip(feature_names, applicant_values):
        print(f"  {name}: {val:.1f}")

    # 4. Generate credit risk explanation report
    print("\nGenerating credit risk explanation memo and SHAP-based charts...")
    report = explain_risk(
        model,
        X,
        idx=high_risk_idx,
        feature_names=feature_names,
        threshold=0.50,
    )

    # 5. Print the credit underwriting memo
    print("\n" + "=" * 60)
    print("CREDIT UNDERWRITING MEMO (Ensemble Boosting & SHAP)")
    print("=" * 60)
    print(report.summary)
    print("=" * 60 + "\n")

    # 6. Save the generated charts
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save each chart to the examples/output/ folder
    report.figures["gauge"].savefig(os.path.join(output_dir, "ensemble_gauge.png"), dpi=150)
    report.figures["drivers"].savefig(os.path.join(output_dir, "ensemble_drivers.png"), dpi=150)
    report.figures["counterfactuals"].savefig(os.path.join(output_dir, "ensemble_counterfactuals.png"), dpi=150)

    print(f"Success! Matplotlib figures successfully saved to:")
    print(f"  - {os.path.join(output_dir, 'ensemble_gauge.png')}")
    print(f"  - {os.path.join(output_dir, 'ensemble_drivers.png')}")
    print(f"  - {os.path.join(output_dir, 'ensemble_counterfactuals.png')}")


if __name__ == "__main__":
    main()
