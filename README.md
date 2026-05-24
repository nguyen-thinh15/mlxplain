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

**One function call. Three professional outputs.** No configuration needed.

<p align="center">
  <img src="docs/images/workflow_diagram.png" alt="mlxplain workflow: Train → Explain → Results" width="650"/>
</p>

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
CREDIT UNDERWRITING MEMO
============================================================
CREDIT DECISION: Declined
Default Probability: 100.0% (threshold: 40.0%)

RISK FACTORS:
  - Debt-to-Income (%): 38.58 (impact: 63.30)

MITIGATING FACTORS:
  + Credit Score: 600.3 (impact: 179.0)
  + Months Employed: 66.17 (impact: 27.96)
  + Income (k$): 79.93 (impact: 19.17)

CURE PATHS (changes needed for approval):
  → Debt-to-Income (%): decrease from 38.58 to 8.41
  → Months Employed: increase from 66.17 to 183.30
  → Credit Score: increase from 600.3 to 766.30
  → Income (k$): increase from 79.93 to 286.30
============================================================
```

---

## ⚡ Quick Start (4 Lines)

```python
from sklearn.linear_model import LogisticRegression
from mlxplain import explain_risk

# 1. Fit your standard ML model
model = LogisticRegression().fit(X_train, y_train)

# 2. Generate a comprehensive credit underwriting explanation in 1 call!
report = explain_risk(model, X_train, idx=0, feature_names=["Income", "DTI", "Credit Score", "Tenure"])

# 3. Print a fully-formatted Credit Underwriting Memo
print(report.summary)

# 4. Save professional diagnostic charts
report.figures["gauge"].savefig("gauge.png")
report.figures["drivers"].savefig("drivers.png")
report.figures["counterfactuals"].savefig("counterfactuals.png")
```

---

## 💻 Supported Models & XAI Translators

**mlxplain** works across the most popular model families with zero configuration changes:

| Model Family | XAI Extraction Method | Counterfactual Strategy |
| :--- | :--- | :--- |
| **Logistic Regression** | Coefficient weights × feature values | **Analytical:** Mathematical exact-solution inversion |
| **Decision Trees & Random Forests** | Split-level class probability differences along decision paths | **Perturbation:** Iterative split-boundary space search |
| **Ensemble Boosting** *(XGBoost & LightGBM)* | SHAP (Shapley Additive exPlanations) values | **Perturbation:** Sample-bounded boundary search |
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
```

All examples save their generated plots to `examples/output/`.

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
