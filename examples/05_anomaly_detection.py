#!/usr/bin/env python3
"""Example demonstrating mlxplain with an Isolation Forest model for anomaly detection.

This example:
1. Generates a synthetic dataset modeling server requests (with CPU, latency, and request rate).
2. Trains a scikit-learn IsolationForest model.
3. Selects a server request that represents a system anomaly/outlier.
4. Generates an anomaly explanation report using the unified `explain()` API.
5. Prints a clear explanation of what features drove the anomaly and what values would make it normal.
6. Saves the generated visual charts (gauge, waterfall, counterfactuals) to the output folder.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import IsolationForest

from mlxplain import explain


def main():
    # 1. Create a synthetic server metric dataset
    print("Generating synthetic server metrics dataset...")
    np.random.seed(42)
    n_samples = 200

    # Normal requests: [CPU Usage (%), Latency (ms), Requests per Second]
    # Normal is centered around CPU=30%, Latency=50ms, RPS=100
    cpu = np.random.normal(30, 5, n_samples - 5).clip(0, 100)
    latency = np.random.normal(50, 10, n_samples - 5).clip(5, 500)
    rps = np.random.normal(100, 15, n_samples - 5).clip(1, 1000)

    # 5 anomalous spike points (high CPU load spike simulation)
    # They have extreme spikes in CPU, but normal Latency and RPS
    cpu_anomaly = np.random.normal(95, 2, 5).clip(0, 100)
    latency_anomaly = np.random.normal(50, 10, 5).clip(5, 500)
    rps_anomaly = np.random.normal(100, 15, 5).clip(1, 1000)

    cpu_all = np.concatenate([cpu, cpu_anomaly])
    latency_all = np.concatenate([latency, latency_anomaly])
    rps_all = np.concatenate([rps, rps_anomaly])

    X = np.column_stack([cpu_all, latency_all, rps_all])
    feature_names = ["CPU Usage (%)", "Latency (ms)", "Requests per Second"]

    # 2. Train an Isolation Forest model
    print("Training scikit-learn IsolationForest model...")
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X)

    # 3. Select one of the anomalous points (index 195 is the first outlier)
    anomaly_idx = 195
    metric_values = X[anomaly_idx]
    print(f"\nSelected anomalous server request at index {anomaly_idx}:")
    for name, val in zip(feature_names, metric_values, strict=False):
        print(f"  {name}: {val:.1f}")

    # 4. Generate anomaly explanation report
    # IsolationForest returns an anomaly score in range [0, 1] via get_probability()
    # A score >= 0.50 classifies the instance as an "Anomaly" (outlier)
    print("\nGenerating anomaly explanation report and charts...")
    report = explain(
        model,
        X,
        idx=anomaly_idx,
        feature_names=feature_names,
        threshold=0.50,
        positive_label="Anomaly (DDoS Spike)",
        negative_label="Normal Behavior",
    )

    # 5. Print a business-friendly summary report
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION REPORT")
    print("=" * 60)
    print(f"Prediction:         {report.prediction}")
    print(f"Anomaly Score:      {report.probability:.3f} (Threshold: {report.threshold:.2f})")

    print("\nTop Contributing Drivers:")
    for d in report.positive_drivers:  # Outlier contributions push toward anomaly
        print(f"  - {d.feature} of {d.value:.1f} was a highly abnormal driver (impact: {d.impact:.3f})")

    print("\nRequired Changes to Restore Normal Behavior (Counterfactuals):")
    for cf in report.counterfactuals[:3]:
        print(
            f"  - Reduce {cf.feature} from {cf.current_value:.1f} to {cf.target_value:.1f} (change: {cf.change_needed:.1f})"
        )
    print("=" * 60 + "\n")

    # 6. Save the generated charts
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save each chart to the examples/output/ folder
    report.figures["gauge"].savefig(os.path.join(output_dir, "anomaly_gauge.png"), dpi=150)
    report.figures["drivers"].savefig(os.path.join(output_dir, "anomaly_drivers.png"), dpi=150)
    report.figures["counterfactuals"].savefig(os.path.join(output_dir, "anomaly_counterfactuals.png"), dpi=150)

    print("Success! Matplotlib figures successfully saved to:")
    print(f"  - {os.path.join(output_dir, 'anomaly_gauge.png')}")
    print(f"  - {os.path.join(output_dir, 'anomaly_drivers.png')}")
    print(f"  - {os.path.join(output_dir, 'anomaly_counterfactuals.png')}")


if __name__ == "__main__":
    main()
