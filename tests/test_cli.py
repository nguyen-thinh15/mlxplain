"""Automated tests for the command-line interface."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlxplain.cli import load_csv, main, parse_bounds, parse_immutable, serialize_report_to_json
from mlxplain.core.report import ExplanationReport, FeatureDriver


@pytest.fixture
def sample_csv_with_header():
    content = "FICO,Income,DebtRatio\n720,50000,0.35\n680,45000,0.42\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_csv_no_header():
    content = "720,50000,0.35\n680,45000,0.42\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_load_csv_with_header(sample_csv_with_header):
    X, feature_names = load_csv(sample_csv_with_header)
    assert feature_names == ["FICO", "Income", "DebtRatio"]
    assert X.shape == (2, 3)
    assert np.allclose(X[0], [720.0, 50000.0, 0.35])


def test_load_csv_no_header(sample_csv_no_header):
    X, feature_names = load_csv(sample_csv_no_header)
    assert feature_names is None
    assert X.shape == (2, 3)
    assert np.allclose(X[0], [720.0, 50000.0, 0.35])


def test_parse_bounds_json_string():
    bounds = parse_bounds('{"FICO": [600, 850], "Income": [10000, 200000]}')
    assert bounds == {"FICO": (600.0, 850.0), "Income": (10000.0, 200000.0)}


def test_parse_bounds_json_file():
    data = {"FICO": [600, 850]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name
    try:
        bounds = parse_bounds(path)
        assert bounds == {"FICO": (600.0, 850.0)}
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_parse_immutable():
    assert parse_immutable("FICO, Income") == ["FICO", "Income"]
    assert parse_immutable("") is None


def test_serialize_report_to_json():
    report = ExplanationReport(
        prediction="Approved",
        probability=0.25,
        threshold=0.5,
        positive_drivers=[FeatureDriver(feature="FICO", value=720.0, impact=0.2, direction="positive")],
        negative_drivers=[],
        counterfactuals=[],
    )
    json_str = serialize_report_to_json(report)
    data = json.loads(json_str)
    assert data["prediction"] == "Approved"
    assert data["probability"] == 0.25
    assert len(data["positive_drivers"]) == 1


@patch("mlxplain.cli.load_model")
@patch("mlxplain.cli.load_csv")
@patch("mlxplain.cli.explain")
def test_cli_main_explain_text(mock_explain, mock_load_csv, mock_load_model):
    mock_load_model.return_value = MagicMock()
    mock_load_csv.return_value = (np.array([[720.0, 50000.0]]), ["FICO", "Income"])

    report = ExplanationReport(
        prediction="Approved",
        probability=0.25,
        threshold=0.5,
        positive_drivers=[FeatureDriver(feature="FICO", value=720.0, impact=0.2, direction="positive")],
        negative_drivers=[],
        counterfactuals=[],
    )
    mock_explain.return_value = report

    test_args = [
        "mlxplain",
        "explain",
        "--model",
        "dummy_model.pkl",
        "--data",
        "dummy_data.csv",
        "--idx",
        "0",
        "--format",
        "text",
    ]

    with patch("sys.argv", test_args), patch("sys.stdout"):
        main()
        # Verify clean orchestration call was made
        mock_explain.assert_called_once()
