#!/usr/bin/env python3
"""Example demonstrating multi-class credit grading explainability in mlxplain.

Business Scenario: Explaining bank credit risk grades (Grade A, B, C) for loan applicants, and calculating path upgrades to premium status.
mlxplain Capability: Multi-class argmax decision translation, analytical closed-form multinomial logistic counterfactual upgrades, multi-class heatmaps, and full English/Vietnamese localization.
Expected Runtime: < 2 seconds.
Required Dependencies: numpy, scikit-learn, matplotlib, mlxplain.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlxplain import explain_risk


def main():
    print("=" * 80)
    # Visual guidelines: Bold header, premium console interface
    print("      mlxplain // Multi-Class Credit Grading & Underwriting Suite")
    print("=" * 80)

    # 1. Synthesize credit dataset with three classes (A, B, C)
    print("\n[1/5] Synthesizing credit grading dataset (500 applicants, 3 features)...")
    rng = np.random.RandomState(42)
    X = rng.randn(500, 3)
    feature_names = ["Annual Income ($k)", "Years Employed", "Debt-to-Income (%)"]

    # Assign grades based on simple rules:
    # High income, high tenure, low debt -> A
    # Mid-range -> B
    # Low income, low tenure, high debt -> C
    score = X[:, 0] * 1.5 + X[:, 1] * 1.0 - X[:, 2] * 2.0

    y = np.zeros(500, dtype=object)
    y[score > 1.0] = "A"
    y[(score >= -1.0) & (score <= 1.0)] = "B"
    y[score < -1.0] = "C"

    # Print distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Grade distribution: {dict(zip(unique, counts, strict=False))}")

    # 2. Train a multinomial Logistic Regression model
    print("\n[2/5] Training multinomial Logistic Regression model...")
    model_lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
    model_lr.fit(X, y.astype(str))

    # Pick a candidate who is currently Grade B, but near the boundary
    # Let's find an index where predicted is B
    preds = model_lr.predict(X)
    target_idx = -1
    for i, pred in enumerate(preds):
        if pred == "B":
            target_idx = i
            break

    if target_idx == -1:
        target_idx = 0

    print(f"Selected applicant at index {target_idx} for underwriting explanation.")
    print(f"Applicant attributes: {dict(zip(feature_names, X[target_idx], strict=False))}")

    # 3. Call explain_risk in English
    print("\n[3/5] Generating English explanation report via explain_risk (Target: Grade A)...")
    report_en = explain_risk(model=model_lr, X=X, idx=target_idx, feature_names=feature_names, target_class="A")

    print("\n--- OFFICIAL CREDIT GRADED UNDERWRITING MEMO (ENGLISH) ---")
    print(report_en.summary)

    # 4. Call explain_risk in Vietnamese
    print("\n[4/5] Generating Vietnamese explanation report via explain_risk (Target: Hạng A)...")
    report_vi = explain_risk(
        model=model_lr, X=X, idx=target_idx, feature_names=feature_names, target_class="A", language="vi"
    )

    print("\n--- BIÊN BẢN PHÂN HẠNG TÍN DỤNG CHÍNH THỨC (VIETNAMESE) ---")
    print(report_vi.summary)

    # 5. Save the premium multi-class visual charts
    print("\n[5/5] Saving premium multi-class visualization charts to examples/output/...")
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)

    # English figures
    for name, fig in report_en.figures.items():
        path = os.path.join(output_dir, f"multiclass_lr_{name}_en.jpg")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    # Vietnamese figures
    for name, fig in report_vi.figures.items():
        path = os.path.join(output_dir, f"multiclass_lr_{name}_vi.jpg")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    print("\nExecution completed successfully! Premium charts and localized memos generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
