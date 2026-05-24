# mlxplain: Internals & Correctness Invariants

This document explains key mathematical internals and invariants required to maintain correctness across different estimators and explainers in `mlxplain`.

---

## 🌲 Isolation Forest Sign-Flip Invariant

### The Problem
By definition, a feature is a **positive driver** of an anomaly when its presence or value pushes the instance closer to being classified as an anomaly (increases the anomaly score).

However, scikit-learn's `IsolationForest` and SHAP's `TreeExplainer` use different directional representations of anomaly:
1. **Isolation Forest Anomaly Score:** A higher score indicates a highly anomalous instance (shorter path lengths). In scikit-learn, `score_samples(x)` returns negative values in range $[-1, 0]$, where values closer to $0$ are anomalous, and values closer to $-1.0$ are normal. We normalize this score to $[0, 1]$ using $s = -\text{score\_samples}(x)$ so that higher scores correspond directly to high anomaly.
2. **SHAP TreeExplainer Attributions:** When run on an `IsolationForest`, the explainer explains the raw leaf values representing average tree path lengths $h(x)$. Since **shorter path lengths** define anomalies, the leaf node path length values *decrease* as the instance becomes more anomalous. As a result, standard SHAP attributions will assign *negative values* to features that decrease path length.

### The Correctness Invariant
If left unadjusted, the feature that pushed the instance to be highly anomalous would receive a negative SHAP value, causing `mlxplain` to misclassify it as a "negative driver" (mitigating factor).

To align the explainer with the anomaly score, we assert the following sign-flip invariant:

$$\phi_{\text{aligned}} = \phi_{\text{TreeExplainer}} \times -1.0$$

#### Proof of Equivalence:
Let $s(x)$ be the anomaly score:

$$s(x) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where:
- $h(x)$ is the path length of instance $x$.
- $E(h(x))$ is the expectation of path lengths across the forest.
- $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree (BST).

Since the base of the exponent is greater than 1 ($2 > 1$) and the exponent is negative, the anomaly score $s(x)$ is strictly **monotonically decreasing** with respect to the expected path length $E(h(x))$:

$$\frac{\partial s}{\partial E(h)} < 0$$

Therefore, to explain contributions toward the anomaly score $s(x)$ (higher is more anomalous) using a linear additive explanation model on tree path lengths, we must negate the path length attributions:

$$\phi_i(s) \propto -\phi_i(E(h))$$

Multiplying by $-1.0$ ensures that:
- $\phi_i \geq 0$ means feature $i$ **decreased** path length, thereby **increasing** the anomaly score (Positive Driver of anomaly).
- $\phi_i < 0$ means feature $i$ **increased** path length, thereby **decreasing** the anomaly score (Negative Driver/Mitigating factor).

This maintains perfect sign consistency with the model-agnostic `shap.Explainer` (which explains $-1.0 \times \text{score\_samples}(x)$ directly) and guarantees correct visualization waterfall layouts.

---

## 🎛️ Reference Locations
- **Anomaly Translator:** [anomaly.py](file:///Users/thinhnguyen/Documents/GitHub/mlxplain/mlxplain/translators/anomaly.py#L35-L45)
- **Unit Tests:** `tests/test_unsupervised.py` asserts that isolation forest explanations correctly flag anomalous features as positive drivers.

---

## 🧮 Multi-Class Logistic regression Analytical Counterfactuals

### Multinomial Logistic Decision Boundary
In multi-class multinomial logistic regression, the model computes linear scores for each class $c \in \{0, 1, \dots, C-1\}$:

$$z_c = \sum_f W_{c, f} X_f + b_c$$

The predicted class $c^*$ is the one with the maximum score:

$$c^* = \operatorname{argmax}_c z_c$$

To transition the prediction from the predicted class $c^*$ to a target class $t$, the score of class $t$ must exceed the score of class $c^*$ (and ideally all other classes). The decision boundary between class $c^*$ and class $t$ is exactly the hyperplane where their linear scores are equal:

$$z_t - z_{c^*} = 0$$

### Mathematical Derivation
Let $\Delta X_f$ be the change in feature $f$ needed to reach the boundary while keeping all other features constant:

$$\sum_{j \neq f} (W_{t, j} - W_{c^*, j}) X_j + (W_{t, f} - W_{c^*, f}) (X_f + \Delta X_f) + (b_t - b_{c^*}) = 0$$

Using the identity $z_c = \sum_j W_{c, j} X_j + b_c$, we can write:

$$(z_t - z_{c^*}) + (W_{t, f} - W_{c^*, f}) \Delta X_f = 0$$

Solving for $\Delta X_f$:

$$\Delta X_f = \frac{z_{c^*} - z_t}{W_{t, f} - W_{c^*, f}}$$

This elegant, closed-form formula provides the exact single-feature counterfactual flip between any two classes in a multinomial logistic regression model.

---

## 🌳 Multi-Class Tree Split Probability Normalization

In multi-class Decision Trees and Random Forests, leaf nodes (and internal nodes) contain the sample count vectors representing class distributions:

$$C_i = [C_{i, 0}, C_{i, 1}, \dots, C_{i, K-1}]$$

Where $K$ is the number of classes. To compute local path-based split attributions, we must normalize the sample counts at each node to obtain the class probability vector:

$$P_i = \frac{C_i}{\sum_{k=0}^{K-1} C_{i, k}}$$

The contribution vector $\Delta P$ of a feature split transitioning from parent node $p$ to child node $c$ is defined as the change in the probability vectors:

$$\Delta P = P_c - P_p = \left[ \frac{C_{c, 0}}{\sum C_{c, k}} - \frac{C_{p, 0}}{\sum C_{p, k}}, \dots, \frac{C_{c, K-1}}{\sum C_{c, k}} - \frac{C_{p, K-1}}{\sum C_{p, k}} \right]$$

These delta probability vectors are accumulated for each feature along the decision paths and averaged across estimators (trees) to produce mathematically sound, additive multi-class path attributions.
