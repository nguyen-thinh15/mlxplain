# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-05-24

### Added
- **Multi-Class Support (Grades & Tiers)**:
  - Extended core data structures (`ExplanationReport.probabilities` and `FeatureDriver.per_class_impacts`) for multi-class predictions.
  - Implemented `interpret_multiclass()` in the base domain interface with a default fallback.
  - Updated all three translators (`LogisticTranslator`, `TreeTranslator`, and `EnsembleTranslator`) to extract class probability vectors and per-class feature impacts.
  - Implemented analytical multinomial logistic regression counterfactual calculation $\Delta X_f = \frac{z_{c^*} - z_t}{W_{t, f} - W_{c^*, f}}$ and generalized perturbation-based searches.
  - Added new premium visualizations: **Class Probabilities Bar Chart** and **Per-Class Feature Impact Heatmap**.
  - Enabled multi-class grading in the credit risk domain localized as **"Hạng A/B/C"** in Vietnamese and **"Grade A/B/C"** in English.
  - Created executable example script `examples/07_multiclass_credit_grading.py` and comprehensive test coverage.
- **Unsupervised Model Explainability**:
  - Native support for `IsolationForest` anomaly explanation (attributions sign-flip invariant) via SHAP and perturbation-based counterfactual search.
  - Native support for `KMeans` clustering assignment explanation with analytical closed-form $L_2$ projections via a dedicated `explain_cluster()` endpoint.
  - Complete Vietnamese translation and localized example scripts for both clustering and anomaly detection.
- **Vietnamese Translation & Localization**:
  - Built native Vietnamese translations for decision labels, feature direction strings (`tích cực` / `tiêu cực`), and generated credit memos using a standard `language="vi"` parameter across the entire package.
- **Trust & Correctness Foundations**:
  - Added optional `immutable_features` and `feature_bounds` parameters to the core `explain` and `explain_risk` interfaces.
  - Implemented a centralized `ConstraintHelper` to enforce user bounds during analytical and perturbation-based counterfactual searches.
  - Created mathematical proof documentation for `IsolationForest` path length sign attribution invariants and multinomial logit boundaries (`docs/internals.md`).
- **Developer Governance**:
  - Added this comprehensive `CHANGELOG.md` file.
  - Added `CONTRIBUTING.md` outlining developer onboarding and quality gates.
  - Configured optimized GitHub Actions CI pipeline (`.github/workflows/ci.yml`).

### Changed
- **BREAKING:** `shap` is now an optional dependency! To reduce cold-start import overhead and bundle size, `shap` has been moved to `[project.optional-dependencies]`.
  - If you use ensemble-based explainers (`EnsembleTranslator` for XGBoost / LightGBM) or anomaly detection (`IsolationForestTranslator`), please make sure to install using **`pip install mlxplain-xai[shap]`**.
  - Logistic regression and standard decision tree/random forest explainability continue to work fully out-of-the-box without requiring `shap`.
- **Memo Normalization**: Normalized raw driver impact values into intuitive percentages of total absolute feature impact calculated separately per group (`% of risk impact` and `% of mitigating impact`).
- **Vietnamese Memo Phrasing**: Polished Vietnamese credit memo translations with natural phrasing (`chiếm X.X% tổng mức rủi ro` and `chiếm X.X% tổng mức giảm thiểu`).

### Fixed
- Fixed global matplotlib state memory leaks by switching from `plt.subplots()` to an Object-Oriented `Figure` constructor.
- Optimized performance in `TreeTranslator` by replacing $O(N)$ list index lookups with an $O(1)$ enum dictionary cache.

---

## [0.1.2] - 2026-05-24

### Added
- Released Phase 1 core translation engines (Logistic, Decision Trees, Ensemble SHAP).
- Created pluggable business domains with `CreditRiskDomain` as the first implementation.
- Added domain-agnostic matplotlib visualizations (Gauge, Waterfall, Counterfactual bars).
- Created advanced credit risk dossier example producing premium glassmorphic HTML reports.
