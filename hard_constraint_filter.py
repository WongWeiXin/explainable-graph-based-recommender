
from __future__ import annotations

import argparse
from html import parser
import json
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "dataset"
    / "Malaysia_Property_Cleaned_Raw.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hard_constraint_filter"

# Columns that must be numeric when used as constraints.
NUMERIC_COLUMNS = {
    "price",
    "bedroom",
    "bathroom",
    "built_up_area",
    "land_area",
    "car_parks",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply hard constraints to the candidate property pool."
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-bedroom", type=float, default=None)
    parser.add_argument("--min-bathroom", type=float, default=None)
    parser.add_argument("--min-built-up-area", type=float, default=None)
    parser.add_argument("--min-car-parks", type=float, default=None)
    parser.add_argument("--property-type", type=str, default=None)
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--tenure", type=str, default=None)
    parser.add_argument("--max-built-up-area", type=float, default=None)
    parser.add_argument("--max-land-area", type=float, default=None)
    parser.add_argument("--area", type=str, default=None)
    parser.add_argument("--district", type=str, default=None)

    return parser.parse_args()


def main():
    started = time.perf_counter()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Candidate pool not found: {args.input}\n"
            "Run candidate generation first, or provide --input."
        )

    df = pd.read_csv(args.input)
    initial_count = len(df)

    constraints = {
        "max_price": ("price", args.max_price, "max"),
        "min_bedroom": ("bedroom", args.min_bedroom, "min"),
        "min_bathroom": ("bathroom", args.min_bathroom, "min"),
        "min_built_up_area": (
            "built_up_area", args.min_built_up_area, "min"
        ),
        "max_built_up_area": (
            "built_up_area", args.max_built_up_area, "max"
        ),
        "min_car_parks": ("car_parks", args.min_car_parks, "min"),
    }

    # Check all requested columns before filtering.
    requested = [
        (name, column, value, operation)
        for name, (column, value, operation) in constraints.items()
        if value is not None
    ]

    categorical = {
        "property_type": args.property_type,
        "state": args.state,
        "tenure": args.tenure,
        "area": args.area,
        "district": args.district,
    }
    requested_categorical = {
        name: value
        for name, value in categorical.items()
        if value is not None
    }

    required_columns = (
        [column for _, column, _, _ in requested]
        + list(requested_categorical.keys())
    )

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Required constraint columns are missing: "
            + ", ".join(missing_columns)
        )

    # Convert numeric constraint columns safely.
    for _, column, _, _ in requested:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Safety check: don't compare raw user requirements against normalized
    # 0-to-1 values. The script cannot infer the original scale reliably.
    scale_warnings = []
    for name, column, value, operation in requested:
        observed = df[column].dropna()

        if (
            not observed.empty
            and observed.between(0, 1).all()
            and abs(value) > 1
        ):
            scale_warnings.append(
                f"{name}: requested value {value} but '{column}' "
                "contains only values between 0 and 1. "
                "Confirm the preprocessing/scaler and convert the "
                "query to the dataset's scale before filtering."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if scale_warnings:
        report = {
            "status": "scale_mismatch_not_filtered",
            "input_file": str(args.input),
            "candidate_count_before_filtering": initial_count,
            "eligible_candidates": None,
            "constraints_applied": {},
            "scale_warnings": scale_warnings,
            "elapsed_seconds": round(
                time.perf_counter() - started, 4
            ),
        }

        report_path = OUTPUT_DIR / "filter_report.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("Filtering stopped to prevent incorrect results.")
        for warning in scale_warnings:
            print(f"WARNING: {warning}")
        print(f"Report saved to: {report_path}")
        return

    # Apply constraints in order and record the number of candidates
    # before each constraint is applied.
    eligible = df.copy()
    audit = []

    for name, column, value, operation in requested:
        before = len(eligible)

        if operation == "max":
            mask = eligible[column].notna() & (eligible[column] <= value)
        else:
            mask = eligible[column].notna() & (eligible[column] >= value)

        eligible = eligible.loc[mask].copy()

        audit.append({
            "constraint": name,
            "column": column,
            "operator": "<=" if operation == "max" else ">=",
            "value": value,
            "candidates_before": before,
            "excluded_by_constraint": before - len(eligible),
            "candidates_after": len(eligible),
        })

    for column, value in requested_categorical.items():
        before = len(eligible)
        mask = (
            eligible[column].astype("string").str.strip().str.casefold()
            == value.strip().casefold()
        )
        eligible = eligible.loc[mask.fillna(False)].copy()

        audit.append({
            "constraint": column,
            "column": column,
            "operator": "equals",
            "value": value,
            "candidates_before": before,
            "excluded_by_constraint": before - len(eligible),
            "candidates_after": len(eligible),
        })

    output_path = OUTPUT_DIR / "eligible_properties.csv"
    eligible.to_csv(output_path, index=False)

    status = (
        "success"
        if len(eligible) > 0
        else "empty_result_constraints_not_relaxed"
    )

    report = {
        "status": status,
        "input_file": str(args.input),
        "output_file": str(output_path),
        "candidate_count_before_filtering": initial_count,
        "eligible_candidates": len(eligible),
        "constraints_applied": {
            name: value
            for name, (_, value, _) in constraints.items()
            if value is not None
        } | requested_categorical,
        "constraint_audit": audit,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }

    report_path = OUTPUT_DIR / "filter_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Candidates before filtering: {initial_count}")
    print(f"Eligible properties: {len(eligible)}")
    print(f"Status: {status}")
    print(f"Filtered data: {output_path}")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
