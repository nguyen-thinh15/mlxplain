"""Automated tests for HTML exporter."""

import os
import tempfile

import matplotlib.figure as mfig
import pytest

from mlxplain.core.exporter import export_html
from mlxplain.core.report import Counterfactual, ExplanationReport, FeatureDriver


@pytest.fixture
def dummy_report():
    fig = mfig.Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    pos_drivers = [FeatureDriver(feature="FICO", value=720, impact=0.25, direction="positive")]
    neg_drivers = [FeatureDriver(feature="Income", value=50000, impact=0.15, direction="negative")]
    cfs = [Counterfactual(feature="Income", current_value=50000, target_value=60000, change_needed=10000)]

    return ExplanationReport(
        prediction="Declined",
        probability=0.75,
        threshold=0.5,
        positive_drivers=pos_drivers,
        negative_drivers=neg_drivers,
        counterfactuals=cfs,
        figures={"gauge": fig, "drivers": fig, "counterfactuals": fig},
        summary="Localized credit memo paragraph content.",
        domain_output={"language": "en", "version": "0.2.0"},
    )


def test_html_export_creates_file(dummy_report):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.html")
        export_html(dummy_report, out_path, include_plotly=False)

        assert os.path.exists(out_path)
        with open(out_path, encoding="utf-8") as f:
            html = f.read()

        # Check standard layout elements
        assert "<!DOCTYPE html>" in html
        assert "mlxplain Decision Dossier" in html
        assert "FICO" in html
        assert "Income" in html
        assert "Declined" in html
        assert "data:image/png;base64," in html  # Base64 embedded figures


def test_html_export_vietnamese_localization(dummy_report):
    dummy_report.domain_output["language"] = "vi"
    dummy_report.domain_output["positive_label"] = "Cần Điều chỉnh"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report_vi.html")
        export_html(dummy_report, out_path, include_plotly=False)

        assert os.path.exists(out_path)
        with open(out_path, encoding="utf-8") as f:
            html = f.read()

        # Check Vietnamese localized translations
        assert "Hồ sơ Quyết định mlxplain" in html
        assert "Ngưỡng Quyết định" in html
        assert "Nhân tố Tác động" in html
