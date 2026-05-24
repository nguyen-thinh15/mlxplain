#!/usr/bin/env python3
"""Example demonstrating mlxplain with a KMeans model for customer segmentation.

Business Scenario: Explaining retail customer behavior segments and designing marketing growth plans to upgrade customers to premium tiers.
mlxplain Capability: Unsupervised KMeans explainability via spatial centroid distances, and closed-form mathematical L2 counterfactual projection roadmaps.
Expected Runtime: < 1 second.
Required Dependencies: numpy, scikit-learn, matplotlib, mlxplain.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans

from mlxplain import explain_cluster


def main():
    # 1. Create a synthetic customer segmentation dataset
    print("Generating synthetic customer segmentation dataset...")
    np.random.seed(42)

    # We will generate three distinct segments:
    # Segment 0: Churn Risk (Low spend, Low frequency)
    # Segment 1: Average Buyer (Moderate spend, Moderate frequency)
    # Segment 2: High-Value Spender (High spend, High frequency)
    # Features: [Annual Spend ($), Annual Purchases]
    c0 = np.random.normal([150, 4], [30, 1.5], (100, 2)).clip(10, 1000)
    c1 = np.random.normal([600, 15], [80, 3], (100, 2)).clip(10, 2000)
    c2 = np.random.normal([1800, 45], [150, 5], (100, 2)).clip(10, 5000)

    X = np.vstack([c0, c1, c2])
    feature_names = ["Annual Spend ($)", "Annual Purchases"]

    # 2. Train KMeans Clustering
    print("Training scikit-learn KMeans model...")
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)

    # Inspect centroids to label them correctly
    centroids = model.cluster_centers_
    # Let's map cluster index to business labels based on average annual spend
    spend_order = np.argsort(centroids[:, 0])
    label_mapping = {
        spend_order[0]: "Low-Value / Churn Risk",
        spend_order[1]: "Average Value",
        spend_order[2]: "High-Value Spender",
    }

    # 3. Select a low-value customer at index 0 (which was drawn from c0)
    customer_idx = 0
    customer_values = X[customer_idx]
    print(f"\nSelected customer at index {customer_idx}:")
    for name, val in zip(feature_names, customer_values, strict=False):
        print(f"  {name}: {val:.1f}")

    # 4. Generate clustering explanation report
    # We set target_cluster to the High-Value Spender cluster (spend_order[2])
    # to find the exact growth path to upgrade this customer's segment!
    target_cluster_idx = int(spend_order[2])
    print(f"\nGenerating cluster explanation vs target cluster '{label_mapping[target_cluster_idx]}'...")
    report = explain_cluster(
        model,
        X,
        idx=customer_idx,
        feature_names=feature_names,
        target_cluster=target_cluster_idx,
    )

    # Retrieve assigned cluster label
    assigned_cluster_idx = report.domain_output["assigned_cluster"]
    assigned_label = label_mapping[assigned_cluster_idx]
    target_label = label_mapping[target_cluster_idx]

    # 5. Print the customer development roadmap
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENT EXPLANATION & GROWTH MEMO")
    print("=" * 60)
    print(f"Current Segment:     {assigned_label} (Cluster {assigned_cluster_idx})")
    print(f"Target Segment:      {target_label} (Cluster {target_cluster_idx})")
    print(f"Proximity Score:     {report.probability:.3f} (1.0 = fully centered, 0.5 = crossover boundary)")

    print("\nSegment Drivers (Why they are in this cluster vs target):")
    for d in report.positive_drivers:
        print(
            f"  - {d.feature}: current value of {d.value:.1f} strongly anchors them to {assigned_label} (impact: {d.impact:.1f})"
        )

    print("\nClosed-Form Customer Upgrade Roadmap (Counterfactuals):")
    for cf in report.counterfactuals:
        print(f"  - Increase {cf.feature} by {cf.change_needed:.1f} (Target: {cf.target_value:.1f})")
    print("=" * 60 + "\n")

    # 6. Save the generated charts
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save each chart to the examples/output/ folder
    report.figures["gauge"].savefig(os.path.join(output_dir, "kmeans_gauge.jpg"), dpi=150)
    report.figures["drivers"].savefig(os.path.join(output_dir, "kmeans_drivers.jpg"), dpi=150)
    report.figures["counterfactuals"].savefig(os.path.join(output_dir, "kmeans_counterfactuals.jpg"), dpi=150)

    print("Success! Matplotlib figures successfully saved to:")
    print(f"  - {os.path.join(output_dir, 'kmeans_gauge.jpg')}")
    print(f"  - {os.path.join(output_dir, 'kmeans_drivers.jpg')}")
    print(f"  - {os.path.join(output_dir, 'kmeans_counterfactuals.jpg')}")


if __name__ == "__main__":
    main()
