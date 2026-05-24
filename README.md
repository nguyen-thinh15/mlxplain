<p align="center">
  <img src="docs/images/hero_banner.png" alt="mlxplain — Explainable AI for Credit Decisions" width="700"/>
</p>

<h1 align="center">mlxplain</h1>

<p align="center">
  <strong>Turn any ML model into a compliance-ready credit memo — in 4 lines of code.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#-supported-models--xai-translators"><img src="https://img.shields.io/badge/models-LogReg%20%7C%20Trees%20%7C%20XGBoost-teal.svg" alt="Supported Models"></a>
</p>

---

## 🔥 What You Get

**One function call. Multi-pillar professional outputs.** No complex configuration needed.

<p align="center">
  <img src="docs/images/workflow_diagram.png" alt="mlxplain workflow: Train → Explain → Results" width="650"/>
</p>

### 💻 Standing Out: Premium HTML Risk Dossier
In addition to standard text memos and charts, `mlxplain` generates a fully standalone, **premium glassmorphic HTML Credit Risk Dossier** with inlined high-resolution vector SVGs. This provides a portable, compliance-ready interactive risk dashboard out-of-the-box!

### 📊 Auto-Generated Diagnostic Charts

Every call to `explain_risk()` produces three publication-ready `matplotlib` figures:

<table>
  <tr>
    <td align="center"><strong>Probability Gauge</strong><br/><em>See the decision at a glance</em></td>
    <td align="center"><strong>Feature Drivers (Waterfall)</strong><br/><em>What pushed the prediction?</em></td>
    <td align="center"><strong>Counterfactual Bars</strong><br/><em>What needs to change?</em></td>
  </tr>
  <tr>
    <td><img src="docs/images/logistic_gauge.png" alt="Probability Gauge" width="350"/></td>
    <td><img src="docs/images/logistic_drivers.png" alt="Feature Drivers" width="350"/></td>
    <td><img src="docs/images/logistic_counterfactuals.png" alt="Counterfactual Changes" width="350"/></td>
  </tr>
</table>

### 📝 Auto-Generated Credit Underwriting Memo

For declined applicants, **mlxplain** generates a compliance-ready text summary with the decision, probability, ranked risk drivers, and actionable cure paths:

```text
============================================================
CREDIT UNDERWRITING MEMO (XGBoost SHAP / RandomForest XAI)
============================================================
CREDIT DECISION: Declined
Default Probability: 65.7% (threshold: 45.0%)

RISK FACTORS:
  - Derogatory Public Records: 2 (impact: 0.1789)
  - Credit Score (FICO): 612.7 (impact: 0.1597)
  - Debt-to-Income (%): 42.54 (impact: 0.1022)
  - Loan-to-Value Ratio (%): 92.13 (impact: 0.08839)
  - Annual Income ($k): 68.41 (impact: 0.05231)
  - Credit Card Utilization (%): 25.97 (impact: 0.03185)
  - Employment Duration (yrs): 5.625 (impact: 0.005195)
  - Open Credit Lines: 6 (impact: 0.001935)
  - Requested Loan Amount ($k): 31.06 (impact: 0.0003766)

MITIGATING FACTORS:
  + Savings Balance ($k): 27.25 (impact: 0.06112)
  + On-Time Payment (%): 99.89 (impact: 0.01372)
  + Years of Credit History: 10.78 (impact: 8.391e-05)

CURE PATHS (changes needed for approval):
  → Derogatory Public Records: decrease from 2 to 0.92
  → Savings Balance ($k): increase from 27.25 to 30.52
  → Credit Card Utilization (%): decrease from 25.97 to 16.62
  → Debt-to-Income (%): decrease from 42.54 to 29.78
  → Loan-to-Value Ratio (%): decrease from 92.13 to 75.54
  → Credit Score (FICO): increase from 612.7 to 686.2
============================================================
```

---

## ⚡ Quick Start (4 Lines)

```python
from sklearn.ensemble import RandomForestClassifier
from mlxplain import explain_risk

# 1. Fit your standard ML model (e.g. 12-feature credit scoring model)
model = RandomForestClassifier().fit(X_train, y_train)

# 2. Define the 12 comprehensive attributes
feature_names = [
    "Annual Income ($k)", "Debt-to-Income (%)", "Credit Score (FICO)",
    "Employment Duration (yrs)", "Savings Balance ($k)", "Requested Loan Amount ($k)",
    "Loan-to-Value Ratio (%)", "Open Credit Lines", "On-Time Payment (%)",
    "Derogatory Public Records", "Credit Card Utilization (%)", "Years of Credit History"
]

# 3. Generate visual & compliance explanation report in 1 call!
report = explain_risk(model, X_train, idx=10, feature_names=feature_names, threshold=0.45)

# 4. Print the professional credit underwriting memo
print(report.summary)

# 5. Save vector SVG diagnostic charts
report.figures["gauge"].savefig("gauge.svg")
report.figures["drivers"].savefig("drivers.svg")
report.figures["counterfactuals"].savefig("counterfactuals.svg")
```

---

## 💻 Supported Models & XAI Translators

**mlxplain** works across the most popular model families with zero configuration changes:

| Model Family | XAI Extraction Method | Counterfactual Strategy |
| :--- | :--- | :--- |
| **Logistic Regression** | Coefficient weights × feature values | **Analytical:** Mathematical exact-solution inversion |
| **Decision Trees & Random Forests** | Split-level class probability differences along decision paths | **Perturbation:** Iterative split-boundary space search |
| **Ensemble Boosting** *(XGBoost & LightGBM)* | SHAP (Shapley Additive exPlanations) values | **Perturbation:** Sample-bounded boundary search |
| **Anomaly Detection** *(Isolation Forest)* | SHAP `TreeExplainer` on isolation trees | **Perturbation:** Iterative score boundary search |
| **Clustering** *(K-Means)* | Spatial distance differences vs target centroid | **Analytical:** Exact L2 half-space mathematical projection |
| **Deep Learning** *(Neural Nets)* | LIME / Integrated Gradients | *(Planned / Deferred)* |

### Charts Across All Model Types

<table>
  <tr>
    <th></th>
    <th align="center">Logistic Regression</th>
    <th align="center">Decision Tree</th>
    <th align="center">XGBoost (SHAP)</th>
  </tr>
  <tr>
    <td><strong>Feature Drivers</strong></td>
    <td><img src="docs/images/logistic_drivers.png" alt="Logistic Drivers" width="280"/></td>
    <td><img src="docs/images/tree_drivers.png" alt="Tree Drivers" width="280"/></td>
    <td><img src="docs/images/ensemble_drivers.png" alt="Ensemble Drivers" width="280"/></td>
  </tr>
</table>

---

## 🎯 Core Capabilities

For any supported binary classification model, **mlxplain** delivers three pillars of explanation:

* **The Decision:** Classifies the prediction into a clear label (e.g., *Approved/Declined*) based on a configurable threshold.
* **The Drivers:** Identifies and ranks which features pushed the prediction in each direction, sorted independently by mathematical impact.
* **The Counterfactuals:** For unfavorable outcomes, calculates the exact minimum feature adjustments required to cross the threshold (e.g., *"Reduce loan amount by $5,000"*).
* **Smart Performance:** Counterfactuals are only computed for unfavorable predictions (probability ≥ threshold) to save CPU cycles on favorable decisions.

---

## 🧩 Pluggable Domain Architecture

The core of **mlxplain** is model-aware but completely business-agnostic. It separates **what the model saw** from **what it means in business terms**:

```
mlxplain/
├── core/              # Data structures, threshold logic, counterfactuals
├── translators/       # Model-specific feature extraction (domain-agnostic)
├── visualizations/    # Standard matplotlib figures
├── domains/           # Pluggable business interpreters
│   └── credit_risk/   # Credit-specific: Approved/Declined, Risk Factors, Cure Paths
└── engine.py          # Unified API entrypoint
```

To support a new domain (e.g., `healthcare` or `fraud_detection`), you simply create a new domain interpreter subclassing `BaseDomain`. **The translators, math engines, and charting code remain 100% untouched.**

### 🏦 Credit Risk Domain Mapping
The credit risk domain interpreter maps mathematical abstractions into commercial banking terminology:

| Mathematical Concept | Credit Risk Business Term |
| :--- | :--- |
| Positive Prediction | **Declined** (High default probability) |
| Negative Prediction | **Approved** (Low default probability) |
| Positive Drivers | **Risk Factors** (Model weaknesses) |
| Negative Drivers | **Mitigating Factors** (Model strengths) |
| Counterfactuals | **Cure Paths** (Required adjustments for approval) |

---

## 🚀 Runnable Examples

The `examples/` directory contains complete, runnable scripts showing **mlxplain** end-to-end. They generate synthetic credit data, train models, produce reports, and save diagnostic plots.

Run them directly using `uv`:

```bash
# 1. Run the Logistic Regression Credit Risk example
uv run python examples/01_logistic_credit_risk.py

# 2. Run the Decision Tree Credit Risk example
uv run python examples/02_decision_tree_credit_risk.py

# 3. Run the XGBoost SHAP-based Credit Risk example
uv run python examples/03_ensemble_credit_risk.py

# 4. Run the Advanced 12-Feature HTML Dossier Credit Risk example
uv run python examples/04_advanced_credit_risk.py

# 5. Run the Unsupervised Anomaly Detection example
uv run python examples/05_anomaly_detection.py

# 6. Run the Unsupervised KMeans Customer Segmentation example
uv run python examples/06_kmeans_clustering.py
```

All examples save their generated plots to `examples/output/`. The advanced example also creates `dossier.html` in that directory — open it in your browser to view the interactive glassmorphic dashboard!

---

## 🌀 Unsupervised Model Explainability

**mlxplain** is the first general-purpose explainability engine to unify supervised classification with **unsupervised learning XAI** (Anomaly Detection & Clustering) under the exact same visual and structured output reporting standards!

### 1. Anomaly Detection (via `IsolationForest`)
* **Standardized Anomaly Scoring**: We normalize scikit-learn's Isolation Forest anomaly scores into $[0, 1]$, treating it exactly like a probability with a standard `0.5` decision boundary. Accessible via the unified `explain()` API.
* **SHAP Drivers**: Uses SHAP's `TreeExplainer` on isolation trees to extract exact features pushing the instance into anomalous vs normal states.
* **Cure Counterfactuals**: Employs sample-bounded perturbation search to identify the exact feature changes needed to restore the anomalous system to normal behavior.

### 2. Clustering (via `KMeans`)
* **Dedicated Endpoint**: Introduces `explain_cluster()` to explain K-Means cluster assignments vs runner-up or target clusters.
* **Distance Drivers**: Measures how much each feature contributes to keeping the instance closer to the assigned centroid $c$ rather than the competitor centroid $t$:
  $$\text{impact}_i = (x_i - t_i)^2 - (x_i - c_i)^2$$
* **Closed-Form Upgrade Roadmap**: Utilizes a highly robust, exact **L2 mathematical projection** to calculate the absolute minimum single/multi-feature changes needed to transition the instance to a target cluster *instantly* with zero loops!
* **Cluster Gauge Score**: Normalizes Euclidean distance vectors into a $[0, 1]$ proximity score ($p = d_t^2 / (d_c^2 + d_t^2)$) to reuse the standard gauge visualization.

---

## 📦 Installation

This project is fully compatible with [uv](https://github.com/astral-sh/uv) for lightning-fast package management.

### For Users

Install `mlxplain` directly into your virtual environment:

```bash
uv pip install mlxplain
# or using standard pip
pip install mlxplain
```

Or install the locked dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
pip install -e .
```

### For Developers

Set up the project locally for development:

1. Clone the repository:
   ```bash
   git clone https://github.com/nguyen-thinh15/mlxplain.git
   cd mlxplain
   ```
2. Create and synchronize the environment:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   uv sync --all-extras
   ```
3. Run the test suite:
   ```bash
   uv run pytest tests/ -v
   ```

**Core Dependencies:** `numpy`, `scikit-learn`, `shap`, `matplotlib`
**Optional Dependencies (for ensembles):** `xgboost`, `lightgbm`
