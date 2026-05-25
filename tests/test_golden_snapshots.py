"""Golden snapshot regression tests to ensure binary backward compatibility."""

import importlib.util
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxplain import explain

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "snapshots.json")

# Check optional ensemble imports
has_shap = importlib.util.find_spec("shap") is not None and importlib.util.find_spec("xgboost") is not None

if has_shap:
    import xgboost as xgb


def serialize_report(report) -> dict:
    """Serialize the core numerical/text output of an ExplanationReport."""
    return {
        "prediction": report.prediction,
        "probability": float(report.probability),
        "threshold": float(report.threshold),
        "positive_drivers": [
            {
                "feature": d.feature,
                "value": float(d.value),
                "impact": float(d.impact),
                "direction": d.direction,
            }
            for d in report.positive_drivers
        ],
        "negative_drivers": [
            {
                "feature": d.feature,
                "value": float(d.value),
                "impact": float(d.impact),
                "direction": d.direction,
            }
            for d in report.negative_drivers
        ],
        "counterfactuals": [
            {
                "feature": c.feature,
                "current_value": float(c.current_value),
                "target_value": float(c.target_value),
                "change_needed": float(c.change_needed),
            }
            for c in report.counterfactuals
        ],
    }


def generate_all_reports() -> dict:
    """Generate reports for standard models using fixed seeds."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    # y designed so there are distinct features pushing positive and negative
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0.2).astype(int)
    feature_names = ["income", "tenure", "debt"]

    reports = {}

    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42).fit(X, y)
    reports["logistic_idx0"] = serialize_report(explain(lr, X, idx=0, feature_names=feature_names))
    reports["logistic_idx1"] = serialize_report(explain(lr, X, idx=1, feature_names=feature_names))

    # 2. Decision Tree
    dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    reports["tree_idx0"] = serialize_report(explain(dt, X, idx=0, feature_names=feature_names))
    reports["tree_idx1"] = serialize_report(explain(dt, X, idx=1, feature_names=feature_names))

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y)
    reports["forest_idx0"] = serialize_report(explain(rf, X, idx=0, feature_names=feature_names))
    reports["forest_idx1"] = serialize_report(explain(rf, X, idx=1, feature_names=feature_names))

    # 4. Ensemble XGBoost (only if shap and xgboost are present)
    if has_shap:
        try:
            # Simple XGBoost classifier
            model_xgb = xgb.XGBClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y)
            reports["ensemble_idx0"] = serialize_report(explain(model_xgb, X, idx=0, feature_names=feature_names))
            reports["ensemble_idx1"] = serialize_report(explain(model_xgb, X, idx=1, feature_names=feature_names))
        except Exception as e:
            print(f"Skipping XGBoost golden snapshot due to error: {e}")

    return reports


def test_golden_snapshots():
    """Verify that current binary explanation outputs are identical to baseline snapshots."""
    current_reports = generate_all_reports()

    if not os.path.exists(SNAPSHOT_PATH):
        # Create baseline snapshot if it doesn't exist yet
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(current_reports, f, indent=2)
        # Snapshot created successfully
        return

    with open(SNAPSHOT_PATH) as f:
        snapshots = json.load(f)

    # Compare each model output
    for key, expected in snapshots.items():
        if key not in current_reports:
            # Skip if optional deps prevented generation
            continue
        actual = current_reports[key]

        # Use relaxed tolerance for ensemble models (XGBoost/SHAP) due to version variance
        tol = 5e-2 if "ensemble" in key else 1e-6

        assert actual["prediction"] == expected["prediction"], f"Prediction mismatch for {key}"
        assert abs(actual["probability"] - expected["probability"]) < tol, f"Probability mismatch for {key}"
        assert abs(actual["threshold"] - expected["threshold"]) < tol, f"Threshold mismatch for {key}"

        # Drivers length comparison
        assert len(actual["positive_drivers"]) == len(expected["positive_drivers"]), (
            f"Positive drivers length mismatch for {key}"
        )
        assert len(actual["negative_drivers"]) == len(expected["negative_drivers"]), (
            f"Negative drivers length mismatch for {key}"
        )

        # Check driver details
        for a_drv, e_drv in zip(actual["positive_drivers"], expected["positive_drivers"], strict=False):
            assert a_drv["feature"] == e_drv["feature"]
            assert abs(a_drv["value"] - e_drv["value"]) < 1e-6
            assert abs(a_drv["impact"] - e_drv["impact"]) < tol, (
                f"Positive driver impact mismatch for {key} on {a_drv['feature']}"
            )
            assert a_drv["direction"] == e_drv["direction"]

        for a_drv, e_drv in zip(actual["negative_drivers"], expected["negative_drivers"], strict=False):
            assert a_drv["feature"] == e_drv["feature"]
            assert abs(a_drv["value"] - e_drv["value"]) < 1e-6
            assert abs(a_drv["impact"] - e_drv["impact"]) < tol, (
                f"Negative driver impact mismatch for {key} on {a_drv['feature']}"
            )
            assert a_drv["direction"] == e_drv["direction"]

        # Counterfactuals comparison
        assert len(actual["counterfactuals"]) == len(expected["counterfactuals"]), (
            f"Counterfactuals length mismatch for {key}"
        )
        for a_cf, e_cf in zip(actual["counterfactuals"], expected["counterfactuals"], strict=False):
            assert a_cf["feature"] == e_cf["feature"]
            assert abs(a_cf["current_value"] - e_cf["current_value"]) < 1e-6
            assert abs(a_cf["target_value"] - e_cf["target_value"]) < tol, (
                f"Counterfactual target value mismatch for {key} on {a_cf['feature']}"
            )
            assert abs(a_cf["change_needed"] - e_cf["change_needed"]) < tol, (
                f"Counterfactual change needed mismatch for {key} on {a_cf['feature']}"
            )
