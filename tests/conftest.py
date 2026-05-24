"""Shared test fixtures for mlxplain test suite."""

import pytest


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Close all matplotlib figures after each test to free memory."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
