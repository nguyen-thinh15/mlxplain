"""Elegantly-formatted, standard-library-based CLI interface for mlxplain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from mlxplain.core.exporter import export_html
from mlxplain.engine import explain, explain_risk


def load_model(model_path: str) -> Any:
    """Load a trained model safely using joblib or pickle."""
    # Print the security warning requested by the user's recommendations
    print(
        "\033[93mWARNING: Loading models from disk executes arbitrary code. Only load trusted model files.\033[0m",
        file=sys.stderr,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import joblib

            model = joblib.load(model_path)
        except (ImportError, Exception):
            import pickle  # nosec B403

            with open(model_path, "rb") as f:
                model = pickle.load(f)  # nosec B301  # noqa: S301

    return model


def load_csv(data_path: str) -> tuple[np.ndarray, list[str] | None]:
    """Parse CSV feature matrix, auto-detecting headers in a robust way."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    import csv

    with open(data_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError("Provided CSV file is empty.")

    first_row = reader[0]
    has_header = False
    for val in first_row:
        try:
            float(val)
        except ValueError:
            has_header = True
            break

    if has_header:
        feature_names = [name.strip() for name in first_row]
        data_rows = reader[1:]
    else:
        feature_names = None
        data_rows = reader

    X = np.array([[float(val) for val in row] for row in data_rows], dtype=float)
    return X, feature_names


def parse_bounds(bounds_arg: str | None) -> dict[str, tuple[float, float]] | None:
    """Parse bounds mapping from JSON string or a JSON file path."""
    if not bounds_arg:
        return None

    import json

    if os.path.exists(bounds_arg):
        with open(bounds_arg, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(bounds_arg)

    # Convert list bounds to floats
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}


def parse_immutable(immutable_arg: str | None) -> list[str] | None:
    """Parse comma-separated immutable feature names."""
    if not immutable_arg:
        return None
    return [name.strip() for name in immutable_arg.split(",")]


def render_terminal_dashboard(report: Any, language: str = "en") -> None:
    """Print a gorgeous, premium, ANSI-colored terminal report dashboard."""
    is_vi = language == "vi"

    # Color codes
    C_GREEN = "\033[92m"
    C_RED = "\033[91m"
    C_BLUE = "\033[94m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"
    C_RESET = "\033[0m"

    title = "MLXPLAIN EXPLANATION DOSSIER" if not is_vi else "HỒ SƠ GIẢI THÍCH MLXPLAIN"
    pred_lbl = "Prediction Decision:" if not is_vi else "Quyết định Dự báo:"
    prob_lbl = "Probability:" if not is_vi else "Xác suất:"
    thresh_lbl = "Threshold:" if not is_vi else "Ngưỡng:"
    drivers_lbl = "RANKED FEATURE DRIVERS" if not is_vi else "NHÂN TỐ TÁC ĐỘNG"
    cfs_lbl = "COUNTERFACTUAL ADJUSTMENT CHECKLIST" if not is_vi else "ĐƯỜNG KHẮC PHỤC TỐI THIỂU"

    is_favorable = report.probability < report.threshold
    decision_color = C_GREEN if is_favorable else C_RED
    decision_box = f"{decision_color}{C_BOLD}[ {report.prediction.upper()} ]{C_RESET}"

    # Progress bar representation
    bar_width = 30
    filled = int(report.probability * bar_width)
    empty = bar_width - filled
    progress_bar = f"{C_BLUE}{'█' * filled}{C_RESET}{'░' * empty}"

    print(f"\n{C_BOLD}{C_BLUE}{'=' * 60}{C_RESET}")
    print(f" {C_BOLD}{title}{C_RESET}")
    print(f"{C_BOLD}{C_BLUE}{'=' * 60}{C_RESET}\n")

    # Decision Block Card
    print(f"  {C_BOLD}{pred_lbl:<22}{C_RESET} {decision_box}")
    print(f"  {C_BOLD}{prob_lbl:<22}{C_RESET} {report.probability:.4f}  {progress_bar}")
    print(f"  {C_BOLD}{thresh_lbl:<22}{C_RESET} {report.threshold:.2f}\n")

    # Multi-class classes list if available
    if report.probabilities is not None:
        probs_title = "ALL CLASS PROBABILITIES" if not is_vi else "XÁC SUẤT CÁC LỚP"
        print(f"  {C_BOLD}{C_BLUE}--- {probs_title} ---{C_RESET}")
        for cls, p in sorted(report.probabilities.items(), key=lambda x: x[1], reverse=True):
            cls_bold = C_BOLD if p == max(report.probabilities.values()) else ""
            print(f"   * {cls_bold}{cls:<15}{C_RESET}: {p:.1%}")
        print()

    # Drivers list
    print(f"  {C_BOLD}{C_BLUE}--- {drivers_lbl} ---{C_RESET}")
    all_drivers = sorted(
        report.positive_drivers + report.negative_drivers,
        key=lambda d: d.impact,
        reverse=True,
    )
    if all_drivers:
        max_name_len = max(len(d.feature) for d in all_drivers)
        for d in all_drivers:
            is_pos = d.direction in ("positive", "tích cực")
            sign = "+" if is_pos else "-"
            color = C_RED if is_pos else C_GREEN

            # Create small visual bar for driver impact
            bar_len = min(15, int(d.impact * 20))
            driver_bar = f"{color}{'█' * bar_len}{C_RESET}"

            print(
                f"   {d.feature:<{max_name_len}} "
                f"(Val: {d.value:>6.2f}) : "
                f"{color}{sign}{d.impact:.4f}{C_RESET} "
                f"{driver_bar}"
            )
    else:
        print("   No feature drivers extracted.")
    print()

    # Counterfactuals list
    print(f"  {C_BOLD}{C_BLUE}--- {cfs_lbl} ---{C_RESET}")
    if report.counterfactuals:
        max_cf_len = max(len(c.feature) for c in report.counterfactuals)
        for c in report.counterfactuals:
            change_sign = "+" if c.change_needed > 0 else ""
            print(
                f"   {C_YELLOW}[ ]{C_RESET} {c.feature:<{max_cf_len}} : "
                f"{c.current_value:.3f} -> {C_BOLD}{c.target_value:.3f}{C_RESET} "
                f"(Change: {C_GREEN}{change_sign}{c.change_needed:.3f}{C_RESET})"
            )
    else:
        txt = (
            "No counterfactual changes required or feasible."
            if not is_vi
            else "Không cần hoặc không thể thực hiện điều chỉnh khắc phục."
        )
        print(f"   {C_GREEN}{txt}{C_RESET}")

    print(f"\n{C_BOLD}{C_BLUE}{'=' * 60}{C_RESET}\n")


def serialize_report_to_json(report: Any) -> str:
    """Serialize the key fields of ExplanationReport to structured JSON."""
    report_dict = {
        "prediction": report.prediction,
        "probability": float(report.probability),
        "threshold": float(report.threshold),
        "positive_drivers": [
            {
                "feature": d.feature,
                "value": float(d.value),
                "impact": float(d.impact),
                "direction": d.direction,
            }
            for d in report.positive_drivers
        ],
        "negative_drivers": [
            {
                "feature": d.feature,
                "value": float(d.value),
                "impact": float(d.impact),
                "direction": d.direction,
            }
            for d in report.negative_drivers
        ],
        "counterfactuals": [
            {
                "feature": c.feature,
                "current_value": float(c.current_value),
                "target_value": float(c.target_value),
                "change_needed": float(c.change_needed),
            }
            for c in report.counterfactuals
        ],
        "probabilities": (
            {k: float(v) for k, v in report.probabilities.items()} if report.probabilities is not None else None
        ),
        "summary": report.summary,
    }
    return json.dumps(report_dict, indent=2)


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="Translate ML model predictions into human-readable business explanations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # explain sub-command
    explain_parser = subparsers.add_parser("explain", help="Generate explanation report for a data row.")
    explain_parser.add_argument("--model", required=True, help="Path to pickled or joblib model file.")
    explain_parser.add_argument("--data", required=True, help="Path to CSV features dataset file.")
    explain_parser.add_argument("--idx", type=int, default=0, help="Zero-based row index to explain.")
    explain_parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold.")
    explain_parser.add_argument("--output", help="Optional file path to export HTML/JSON report.")
    explain_parser.add_argument(
        "--format",
        choices=["text", "json", "memo", "html"],
        help="Output format (default: 'text' to stdout, or 'html' if --output ends in .html).",
    )
    explain_parser.add_argument("--domain", choices=["credit_risk"], help="Optional domain routing logic.")
    explain_parser.add_argument("--language", choices=["en", "vi"], default="en", help="Output language translation.")
    explain_parser.add_argument("--top-k", type=int, help="Limit feature drivers displayed.")
    explain_parser.add_argument("--immutable", help="Comma-separated list of immutable feature names.")
    explain_parser.add_argument(
        "--bounds", help="JSON string or JSON file path of feature bounds e.g. '{\"Age\": [18, 80]}'."
    )

    args = parser.parse_args()

    try:
        # Load resources
        model = load_model(args.model)
        X, feature_names = load_csv(args.data)

        # Parse constraints
        immutable_features = parse_immutable(args.immutable)
        feature_bounds = parse_bounds(args.bounds)

        # Orchestrate explanation
        if args.domain == "credit_risk":
            report = explain_risk(
                model=model,
                X=X,
                idx=args.idx,
                feature_names=feature_names,
                threshold=args.threshold,
                top_k=args.top_k,
                language=args.language,
                immutable_features=immutable_features,
                feature_bounds=feature_bounds,
            )
        else:
            report = explain(
                model=model,
                X=X,
                idx=args.idx,
                feature_names=feature_names,
                threshold=args.threshold,
                top_k=args.top_k,
                language=args.language,
                immutable_features=immutable_features,
                feature_bounds=feature_bounds,
            )

        # Handle interactive plotly figures if requested or plotly is installed
        try:
            from mlxplain.visualizations.plotly_charts import plot_report_plotly

            report.plotly_figures = plot_report_plotly(report)
        except ImportError:
            pass

        # Determine output format
        out_format = args.format
        if not out_format:
            if args.output and args.output.lower().endswith(".html"):
                out_format = "html"
            elif args.output and args.output.lower().endswith(".json"):
                out_format = "json"
            else:
                out_format = "text"

        # Generate output content
        if out_format == "text":
            render_terminal_dashboard(report, language=args.language)
        elif out_format == "json":
            json_str = serialize_report_to_json(report)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(json_str)
                print(f"Report saved to JSON: {args.output}")
            else:
                print(json_str)
        elif out_format == "memo":
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(report.summary)
                print(f"Business memo saved: {args.output}")
            else:
                print(report.summary)
        elif out_format == "html":
            # Generate premium single-file HTML report
            if not args.output:
                raise ValueError("An output file path (--output) is required for HTML export.")
            export_html(report, args.output, include_plotly=True)
            print(f"Premium decision dossier exported to HTML: {args.output}")

    except Exception as e:
        print(f"\033[91mError: {e}\033[0m", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
