
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
FILTER_SCRIPT = ROOT / "hard_constraint_filter.py"
BASELINE_SCRIPT = ROOT / "baseline_models" / "direct_preference_baseline.py"

FILTER_DIR = ROOT / "outputs" / "hard_constraint_filter"
FILTERED_CSV = FILTER_DIR / "eligible_properties.csv"
FILTER_REPORT = FILTER_DIR / "filter_report.json"

QUERY_PATH = ROOT / "outputs" / "user_query.json"


def ask_text(prompt):
    value = input(prompt).strip()
    return value or None


def ask_number(prompt):
    value = input(prompt).strip()

    if not value:
        return None

    try:
        result = float(value)
        if result < 0:
            raise ValueError
        return result
    except ValueError:
        print("Enter a non-negative number, or leave blank.")
        return ask_number(prompt)


def main():
    for path in (DATA_PATH, FILTER_SCRIPT, BASELINE_SCRIPT):
        if not path.exists():
            print(f"Required file not found: {path}")
            return

    print("\n=== Residential Property Recommender ===")
    print("Leave optional fields blank to skip them.\n")

    # 1. Mandatory constraints
    max_price = ask_number("Maximum budget (RM): ")
    min_bedroom = ask_number("Minimum bedrooms: ")
    min_bathroom = ask_number("Minimum bathrooms: ")
    min_size = ask_number("Minimum built-up area (sq ft): ")
    max_size = ask_number("Maximum built-up area (sq ft): ")
    min_parking = ask_number("Minimum car parks: ")

    property_type = ask_text("Required property type: ")
    state = ask_text("Required state: ")
    tenure = ask_text("Required tenure: ")
    area_constraint = ask_text("Required area: ")
    district_constraint = ask_text("Required district: ")

    # 2. Ranking preferences
    target_price = ask_number("Preferred target price (RM): ")
    target_bedroom = ask_number("Preferred bedrooms: ")
    target_bathroom = ask_number("Preferred bathrooms: ")
    target_size = ask_number("Preferred built-up area (sq ft): ")

    preferred_area = ask_text("Preferred area for ranking: ")
    preferred_district = ask_text("Preferred district for ranking: ")
    preferred_state = ask_text("Preferred state for ranking: ")
    preferred_tenure = ask_text("Preferred tenure for ranking: ")
    preferred_furnishing = ask_text("Preferred furnishing: ")

    facilities_text = ask_text(
        "Facility columns, comma-separated "
        "(e.g. train_station_count_3km,park_count_1km): "
    )
    facilities = (
        [x.strip() for x in facilities_text.split(",") if x.strip()]
        if facilities_text else []
    )

    query = {
        "query_id": "USER_QUERY",
        "hard_constraints": {
            "max_price": max_price,
            "min_bedroom": min_bedroom,
            "min_bathroom": min_bathroom,
            "min_built_up_area": min_size,
            "max_built_up_area": max_size,
            "min_car_parks": min_parking,
            "property_type": property_type,
            "state": state,
            "tenure": tenure,
            "area": area_constraint,
            "district": district_constraint,
        },
        "preferences": {
            "target_price": target_price,
            "preferred_property_type": property_type,
            "target_bedroom": target_bedroom,
            "target_bathroom": target_bathroom,
            "target_built_up_area": target_size,
            "preferred_area": preferred_area,
            "preferred_district": preferred_district,
            "preferred_state": preferred_state,
            "preferred_tenure": preferred_tenure,
            "preferred_furnishing": preferred_furnishing,
            "preferred_facilities": facilities,
        },
        "weights": {
            "budget_fit": 0.25,
            "property_type_fit": 0.10,
            "bedroom_fit": 0.10,
            "bathroom_fit": 0.05,
            "size_fit": 0.20,
            "location_fit": 0.15,
            "tenure_fit": 0.05,
            "furnishing_fit": 0.03,
            "facility_fit": 0.07,
        },
    }

    QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_PATH.write_text(
        json.dumps(query, indent=2),
        encoding="utf-8",
    )

    # 3. Run hard-constraint filtering
    filter_cmd = [
        sys.executable,
        str(FILTER_SCRIPT),
        "--input", str(DATA_PATH),
    ]

    flags = {
        "--max-price": max_price,
        "--min-bedroom": min_bedroom,
        "--min-bathroom": min_bathroom,
        "--min-built-up-area": min_size,
        "--max-built-up-area": max_size,
        "--min-car-parks": min_parking,
        "--property-type": property_type,
        "--state": state,
        "--tenure": tenure,
        "--area": area_constraint,
        "--district": district_constraint,
    }

    for flag, value in flags.items():
        if value is not None:
            filter_cmd.extend([flag, str(value)])

    print("\nStep 1/2: Applying hard constraints...")
    result = subprocess.run(filter_cmd, cwd=ROOT)

    if result.returncode != 0:
        print("Filtering failed. Ranking was not started.")
        return

    if not FILTER_REPORT.exists():
        print("Filter report was not generated.")
        return

    report = json.loads(
        FILTER_REPORT.read_text(encoding="utf-8")
    )

    if report.get("status") != "success":
        print("\nNo eligible properties were returned.")
        print("Hard constraints have NOT been relaxed.")
        print(f"See report: {FILTER_REPORT}")
        return

    if not FILTERED_CSV.exists():
        print(f"Filtered CSV not found: {FILTERED_CSV}")
        return

    # 4. Run direct-preference scoring on exactly that filtered pool
    baseline_cmd = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--input", str(FILTERED_CSV),
        "--query-json", str(QUERY_PATH),
        "--skip-hard-constraints",
    ]

    print("\nStep 2/2: Scoring and ranking eligible properties...")
    result = subprocess.run(baseline_cmd, cwd=ROOT)

    if result.returncode != 0:
        print("Direct-preference baseline failed.")
        return

    print("\nRecommendation workflow completed.")
    print(
        "Rankings:",
        ROOT / "outputs" / "direct_preference_baseline"
        / "direct_preference_rankings.csv",
    )
    print(
        "Top 10:",
        ROOT / "outputs" / "direct_preference_baseline"
        / "top10_direct_preference.csv",
    )
    print("Filter audit:", FILTER_REPORT)
    print("Query configuration:", QUERY_PATH)


if __name__ == "__main__":
    main()
