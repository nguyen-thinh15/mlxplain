# Contributing to mlxplain

Thank you for your interest in contributing to **mlxplain**! We welcome contributions to expand XAI translators, add new business domains, improve visualizations, or optimize performance.

---

## 🛠️ Development Setup

We use [uv](https://github.com/astral-sh/uv) for fast, robust package and environment management.

1. **Clone and Enter Repository**:
   ```bash
   git clone https://github.com/nguyen-thinh15/mlxplain.git
   cd mlxplain
   ```

2. **Initialize Environment**:
   Create a virtual environment and synchronize dependencies (including optional extras):
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   uv sync --all-extras
   ```

---

## 📐 Coding Standards & Guidelines

We follow strict quality standards to keep the library maintainable and robust:

### 1. Separation of Concerns
- **Core translators** (`mlxplain/translators/`) must be entirely **domain-agnostic**. They calculate mathematical contributions, attributions (SHAP), and model probabilities. They do not know about "credit risk", "claims", or other business terms.
- **Domain interpreters** (`mlxplain/domains/`) map core explanation structures to specific commercial terms.
- Keep visualizations completely reusable and generic.

### 2. Optional Dependency Guardrails
- If a translator requires an optional library (e.g. `shap` for ensembles or anomaly detection, or future deep learning backends), do **not** import it at the top-level of any core file.
- Perform a fast-failing import verification in the translator's `__init__` constructor and raise a clean `ImportError` pointing to the PyPI extra:
  ```python
  try:
      import optional_pkg
  except ImportError as e:
      raise ImportError(
          "This translator requires optional_pkg. Install with: pip install mlxplain-xai[extra]"
      ) from e
  ```
- Import the library locally within active worker methods like `extract_drivers`.

### 3. Matplotlib Figure Rules
- Do **not** use `plt.subplots()` or import `pyplot` globally at the module level.
- Always use the Object-Oriented `Figure` constructor (`from matplotlib.figure import Figure`) to prevent global figure registration memory leaks inside long-running servers.
- Decorate test classes or functions that output plots to ensure proper figure cleanup.

---

## 🧪 Testing

We require high test coverage and zero warnings.

Run the test suite using pytest:
```bash
uv run pytest tests/ -v
```

### Writing New Tests
- Keep all mock data generated dynamically via standard NumPy or scikit-learn fixtures. Do **not** commit large CSV or PKL assets to the repository.
- Use the `@pytest.mark.skipif` decorator on test functions that require optional backends (like SHAP) so they skip gracefully on base environments.

---

## 🚀 Release Checklist

Before releasing a new version:
1. Increment the version number inside `mlxplain/_version.py` (e.g., `__version__ = "0.2.0"`).
2. Update the `CHANGELOG.md` with release notes following the *Keep a Changelog* format.
3. Verify that the dynamic version propagates correctly:
   ```bash
   python -c "import mlxplain; print(mlxplain.__version__)"
   ```
4. Run all local tests and ensure the CI pipeline is entirely green.
