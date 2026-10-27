"""Direct-preference baseline for the CPG residential-property project.

Run from project root:
    python .\\recommendation\\direct_preference_baseline.py

Edit QUERY to match a validated benchmark profile before reporting results.
This baseline does not use graph support or market-context scores.
"""
from pathlib import Path
import json, math, time
import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "direct_preference_baseline"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the direct-preference property baseline."
    )
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--query-json", type=Path, default=None)
    parser.add_argument(
        "--skip-hard-constraints",
        action="store_true",
        help="Use only when input is already hard-filtered.",
    )
    return parser.parse_args()

# Example profile only. Change values/category labels to match your cleaned CSV.
QUERY = {
    "query_id": "Q001",
    "hard_constraints": {
        "max_price": 800000,
        "property_type": None,
        "min_bedroom": 3,
        "min_bathroom": None,
        "min_built_up_area": None,
        "max_built_up_area": None,
        "tenure": None,
        "area": None,
        "district": None,
        "state": None,
    },
    "preferences": {
        "target_price": 700000,
        "preferred_property_type": None,
        "target_bedroom": 3,
        "target_bathroom": 2,
        "target_built_up_area": 1000,
        "preferred_area": None,
        "preferred_district": None,
        "preferred_state": None,
        "preferred_tenure": None,
        "preferred_furnishing": None,
        "preferred_facilities": [],  # e.g. ["train_station_count_3km", "park_count_1km"]
    },
    "weights": {
        "budget_fit": 0.25, "property_type_fit": 0.10,
        "bedroom_fit": 0.10, "bathroom_fit": 0.05,
        "size_fit": 0.20, "location_fit": 0.15,
        "tenure_fit": 0.05, "furnishing_fit": 0.03,
        "facility_fit": 0.07,
    },
}

ALIASES = {
    "id": ["Property_ID", "property_id", "listing_id", "id"],
    "price": ["price", "Price", "selling_price"],
    "property_type": ["property_type", "Property_Type"],
    "bedroom": ["bedroom", "bedrooms", "Bedroom"],
    "bathroom": ["bathroom", "bathrooms", "Bathroom"],
    "built_up_area": ["built_up_area", "builtup_area", "Built_up_area"],
    "tenure": ["tenure", "Tenure"], "area": ["area", "Area"],
    "district": ["district", "District"], "state": ["state", "State"],
    "furnishing": ["furnishing", "Furnishing"],
}


def col_for(df, logical):
    return next((c for c in ALIASES[logical] if c in df.columns), None)


def textval(v):
    if pd.isna(v): return None
    s = str(v).strip()
    return None if not s or s.casefold() in {"nan", "none", "unknown", "-1", "-1.0"} else s.casefold()


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) and x >= 0 else None
    except (TypeError, ValueError):
        return None


def cat_score(value, target):
    a, b = textval(value), textval(target)
    return None if a is None or b is None else float(a == b)


def numeric_score(value, target, tolerance):
    a, b = num(value), num(target)
    if a is None or b is None: return None
    if tolerance <= 0: return float(a == b)
    return max(0.0, 1.0 - abs(a - b) / tolerance)


def apply_hard_constraints(df, cols, constraints):
    mask = pd.Series(True, index=df.index)
    checks = []
    def apply(condition, name):
        nonlocal mask
        condition = condition.fillna(False)
        checks.append({"constraint": name, "passed_before_this_check": int(mask.sum()),
                       "excluded_by_this_check": int((mask & ~condition).sum())})
        mask &= condition
    if constraints.get("max_price") is not None:
        if not cols["price"]: raise ValueError("max_price supplied but price column was not found")
        apply(pd.to_numeric(df[cols["price"]], errors="coerce") <= float(constraints["max_price"]), "max_price")
    for key, logical, op in [
        ("min_bedroom", "bedroom", "min"), ("min_bathroom", "bathroom", "min"),
        ("min_built_up_area", "built_up_area", "min"), ("max_built_up_area", "built_up_area", "max")]:
        v = constraints.get(key)
        if v is not None:
            c = cols[logical]
            if not c: raise ValueError(f"{key} supplied but {logical} column was not found")
            values = pd.to_numeric(df[c], errors="coerce")
            apply(values >= float(v) if op == "min" else values <= float(v), key)
    for key in ["property_type", "tenure", "area", "district", "state"]:
        target = constraints.get(key)
        if target is not None:
            c = cols[key]
            if not c: raise ValueError(f"{key} constraint supplied but column was not found")
            apply(df[c].map(textval).eq(textval(target)), key)
    return df.loc[mask].copy(), checks


def score_row(row, cols, query):
    p, w = query["preferences"], query["weights"]
    scores, evidence = {}, {}
    def add(name, logical, target, categorical=False, tolerance=None):
        c = cols.get(logical)
        if target is None or c is None:
            scores[name] = None
            evidence["evidence_" + name] = "inactive: preference or dataset column unavailable"
            return
        value = row[c]
        score = cat_score(value, target) if categorical else numeric_score(value, target, tolerance)
        scores[name] = score
        evidence["evidence_" + name] = json.dumps({"column": c, "value": None if pd.isna(value) else str(value),
            "target": target, "score": score}, ensure_ascii=False)
    price_target = p.get("target_price")
    add("budget_fit", "price", price_target, tolerance=max(10000, abs(float(price_target)) * .30) if price_target is not None else None)
    add("property_type_fit", "property_type", p.get("preferred_property_type"), True)
    add("bedroom_fit", "bedroom", p.get("target_bedroom"), tolerance=2)
    add("bathroom_fit", "bathroom", p.get("target_bathroom"), tolerance=2)
    area_target = p.get("target_built_up_area")
    add("size_fit", "built_up_area", area_target, tolerance=max(100, abs(float(area_target)) * .40) if area_target is not None else None)
    # Location is exact category match here, not geographic distance.
    loc_scores = []
    for logical, prefkey in [("area", "preferred_area"), ("district", "preferred_district"), ("state", "preferred_state")]:
        if p.get(prefkey) is not None and cols.get(logical):
            s = cat_score(row[cols[logical]], p[prefkey])
            if s is not None: loc_scores.append(s)
    scores["location_fit"] = sum(loc_scores) / len(loc_scores) if loc_scores else None
    evidence["evidence_location_fit"] = json.dumps({"matching_fields_used": len(loc_scores), "scores": loc_scores})
    add("tenure_fit", "tenure", p.get("preferred_tenure"), True)
    add("furnishing_fit", "furnishing", p.get("preferred_furnishing"), True)
    fac_scores, fac_notes = [], []
    for c in p.get("preferred_facilities", []):
        if c not in row.index:
            fac_notes.append(f"{c}: column unavailable"); continue
        v = num(row[c])
        if v is None: fac_notes.append(f"{c}: missing value")
        else: fac_scores.append(float(v > 0)); fac_notes.append(f"{c}: value={v}")
    scores["facility_fit"] = sum(fac_scores) / len(fac_scores) if fac_scores else None
    evidence["evidence_facility_fit"] = "; ".join(fac_notes) if fac_notes else "inactive: no usable facility preferences"
    active = {k: v for k, v in scores.items() if v is not None and w.get(k, 0) > 0}
    denom = sum(w[k] for k in active)
    total = sum(v * w[k] for k, v in active.items()) / denom if denom else None
    out = {"score_" + k: v for k, v in scores.items()}
    out.update(evidence)
    out["active_components"] = ",".join(active)
    out["active_weights"] = json.dumps({k: w[k] for k in active}, sort_keys=True)
    out["direct_preference_score"] = total
    out["score_status"] = "scored" if total is not None else "no_active_preferences"
    return out


def main():
    start = time.perf_counter()
    args = parse_args()

    query = QUERY
    if args.query_json is not None:
        if not args.query_json.exists():
            raise FileNotFoundError(
                f"Query JSON not found: {args.query_json}"
            )
        query = json.loads(
            args.query_json.read_text(encoding="utf-8")
        )

    data_path = args.input
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset/input CSV not found: {data_path}"
        )

    df = pd.read_csv(data_path)
    cols = {k: col_for(df, k) for k in ALIASES if k != "id"}
    id_col = col_for(df, "id")
    if not cols["price"]: raise ValueError(f"Price column not found. Available columns: {list(df.columns)}")
    if not id_col:
        df["generated_property_id"] = [f"ROW_{i}" for i in range(len(df))]
        id_col = "generated_property_id"
    if args.skip_hard_constraints:
        eligible = df.copy()
        constraint_audit = []
    else:
        eligible, constraint_audit = apply_hard_constraints(
            df, cols, query["hard_constraints"]
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if eligible.empty:
        summary = {"query_id": query["query_id"], "dataset_rows": len(df), "eligible_candidates": 0,
                   "status": "empty_result_constraints_not_relaxed", "constraint_audit": constraint_audit,
                   "elapsed_seconds": round(time.perf_counter()-start, 4)}
        (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("No properties satisfy all hard constraints. Constraints were not relaxed.")
        return
    score_df = pd.DataFrame([score_row(row, cols, query) for _, row in eligible.iterrows()], index=eligible.index)
    result = pd.concat([eligible, score_df], axis=1)
    result["query_id"] = query["query_id"]
    result = result.sort_values(["direct_preference_score", id_col], ascending=[False, True], na_position="last", kind="mergesort")
    result.insert(0, "rank", range(1, len(result)+1))
    result.to_csv(OUTPUT_DIR / "direct_preference_rankings.csv", index=False)
    result.head(10).to_csv(OUTPUT_DIR / "top10_direct_preference.csv", index=False)
    summary = {"query_id": query["query_id"], "dataset_path": str(data_path), "dataset_rows": len(df),
        "eligible_candidates": len(eligible), "scored_candidates": int(result["direct_preference_score"].notna().sum()),
        "hard_constraint_violation_rate": 0.0, "columns_used": cols, "weights": query["weights"],
        "constraint_audit": constraint_audit, "elapsed_seconds": round(time.perf_counter()-start, 4),
        "notes": ["Example query must be replaced with validated benchmark profiles.",
          "Verify numeric units and categorical labels in the cleaned CSV.",
          "Missing component evidence is excluded from the active-weight denominator.",
          "Location preference currently uses exact area/district/state matches, not geographic distance."]}
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Direct-preference baseline completed.")
    print(f"Dataset rows: {len(df):,}; eligible: {len(eligible):,}; scored: {summary['scored_candidates']:,}")
    print(f"Rankings: {OUTPUT_DIR / 'direct_preference_rankings.csv'}")
    print(f"Top 10: {OUTPUT_DIR / 'top10_direct_preference.csv'}")
    print(f"Summary: {OUTPUT_DIR / 'run_summary.json'}")

if __name__ == "__main__":
    main()
