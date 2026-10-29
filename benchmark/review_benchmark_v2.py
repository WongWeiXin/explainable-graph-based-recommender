"""Automated pre-review of benchmark v2 (frozen_queries/frozen_queries_set_v2.csv).

Run from the project root:
    python benchmark/review_benchmark_v2.py

This does NOT replace the independent human review required by Section 3.7. It checks every
profile against fixed rules and writes a review sheet so the reviewer can focus on flagged
profiles. Nothing here looks at model scores or rankings.

Checks (severity):
  V  validity     ERROR    values outside the allowed vocabulary / types
  C  consistency  ERROR    contradictions inside a profile
  D  duplicates   WARNING  identical profiles, or identical hard constraints
  E  edge labels  WARNING  edge-case label not true for the profile's values
  R  realism      WARNING  values outside what the catalogue contains for that type
  S  soft prefs   INFO     soft preference that cannot differentiate the eligible set
  B  eligibility  INFO     benchmark role from eligibility_manifest_v2.csv

Outputs:
    frozen_queries/review_v2.csv        one row per profile: checks, flags, blank decision
    benchmark/review_v2_summary.json    counts per check and global notes
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recommender_service import FACILITY_COLUMNS, apply_hard_constraints  # noqa: E402

QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
CATALOGUE = ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
MANIFEST = ROOT / "benchmark" / "eligibility_manifest_v2.csv"
OUT_SHEET = ROOT / "frozen_queries" / "review_v2.csv"
OUT_SUMMARY = ROOT / "benchmark" / "review_v2_summary.json"

PRIORITIES = {"balanced", "price", "size", "facilities"}
TENURES = {"Freehold", "Leasehold", "No preference"}
# v1 labelled Cluster, Link Bungalow / Semi-Detached, Town House, House (Others) and
# Low-Cost House as "less-common"; the check follows that definition.
COMMON_TYPES = {"Terrace House", "Condominium", "Bungalow / Detached House", "Flat"}
EDGE_RULES = {
    "Low budget (≤RM350,000)": lambda r: r.max_budget_rm <= 350_000,
    "Very high budget (≥RM2.5m)": lambda r: r.max_budget_rm >= 2_500_000,
    "High bedroom requirement (≥5)": lambda r: r.min_bedrooms >= 5,
    "High bathroom requirement (≥4)": lambda r: r.min_bathrooms >= 4,
    "Parking and furnished both mandatory": lambda r: r.parking_required == 1 and r.furnished_only == 1,
    "Large preferred size (≥4,500 sqft)": lambda r: r.preferred_built_up_sqft >= 4_500,
    "Less-common property type": lambda r: r.property_type not in COMMON_TYPES,
    "Combination stress test (multiple preferences)": lambda r: True,
}


def facilities_of(v) -> list[str]:
    return [x.strip() for x in str(v).split(";") if x.strip() and str(v) != "nan"]


def main() -> None:
    q = pd.read_csv(QUERIES)
    cat = pd.read_csv(CATALOGUE)
    roles = pd.read_csv(MANIFEST).set_index("query_id")
    cat_d = set(cat["district"].str.strip().str.lower())
    cat_t = set(cat["property_type"].str.strip())
    type_stats = cat.groupby("property_type").agg(
        size_p01=("built_up_area", lambda s: s.quantile(0.01)),
        size_p99=("built_up_area", lambda s: s.quantile(0.99)),
        bed_max=("bedroom", "max"), bath_max=("bathroom", "max"))
    dt_min_price = cat.assign(_d=cat["district"].str.lower()).groupby(["_d", "property_type"])["price"].min()

    hard_cols = ["preferred_area", "max_budget_rm", "property_type", "min_bedrooms",
                 "min_bathrooms", "parking_required", "furnished_only", "tenure"]
    soft_cols = ["main_priority", "nearby_facilities", "preferred_built_up_sqft"]
    as_text = q[hard_cols + soft_cols].astype(str).replace({"nan": "", "<NA>": ""})
    hard_key = as_text[hard_cols].agg("|".join, axis=1)
    full_key = as_text.agg("|".join, axis=1)

    rows, counts = [], {}

    def flag(store, code, sev, msg):
        store.append((code, sev, msg))
        counts[(code, sev)] = counts.get((code, sev), 0) + 1

    for i, r in q.iterrows():
        f = []
        fac = facilities_of(r.nearby_facilities)
        # ---- V: validity
        if str(r.preferred_area).strip().lower() not in cat_d:
            flag(f, "V1", "ERROR", f"district '{r.preferred_area}' not in catalogue")
        if r.property_type not in cat_t:
            flag(f, "V2", "ERROR", f"property type '{r.property_type}' not in catalogue")
        if r.tenure not in TENURES:
            flag(f, "V3", "ERROR", f"tenure '{r.tenure}' not allowed")
        if r.main_priority not in PRIORITIES:
            flag(f, "V4", "ERROR", f"main priority '{r.main_priority}' not allowed")
        bad_fac = [x for x in fac if x not in FACILITY_COLUMNS]
        if bad_fac:
            flag(f, "V5", "ERROR", f"unknown facility {bad_fac}")
        for c in ("parking_required", "furnished_only"):
            if r[c] not in (0, 1):
                flag(f, "V6", "ERROR", f"{c} must be 0/1")
        for c in ("max_budget_rm", "preferred_built_up_sqft"):
            if not (pd.notna(r[c]) and r[c] > 0):
                flag(f, "V7", "ERROR", f"{c} must be > 0")
        # ---- C: consistency
        if r.main_priority == "facilities" and not fac:
            flag(f, "C1", "ERROR", "main priority facilities but no facility selected")
        if r.min_bathrooms > r.min_bedrooms + 2:
            flag(f, "C2", "WARNING", f"min bathrooms ({r.min_bathrooms}) far above min bedrooms ({r.min_bedrooms})")
        key = (str(r.preferred_area).lower(), r.property_type)
        if key in dt_min_price and r.max_budget_rm < dt_min_price[key]:
            flag(f, "C3", "WARNING", f"budget below the cheapest {r.property_type} in {r.preferred_area} "
                                      f"(RM{int(dt_min_price[key]):,})")
        elif key not in dt_min_price:
            flag(f, "C4", "WARNING", f"no {r.property_type} listed in {r.preferred_area} at all")
        # ---- D: duplicates
        if (full_key == full_key[i]).sum() > 1:
            twins = q.loc[(full_key == full_key[i]) & (q.index != i), "query_id"].tolist()
            flag(f, "D1", "WARNING", f"identical profile to {twins}")
        elif (hard_key == hard_key[i]).sum() > 1:
            twins = q.loc[(hard_key == hard_key[i]) & (q.index != i), "query_id"].tolist()
            flag(f, "D2", "INFO", f"same hard constraints as {twins} (soft preferences differ)")
        # ---- E: edge-case labels
        label = str(r.edge_case_difference) if pd.notna(r.edge_case_difference) else ""
        for part in [p.strip() for p in label.split(";") if p.strip()]:
            rule = EDGE_RULES.get(part)
            if rule is None:
                flag(f, "E1", "WARNING", f"unknown edge-case label '{part}'")
            elif not rule(r):
                flag(f, "E2", "WARNING", f"edge-case label not true: '{part}'")
        # ---- R: realism
        if r.property_type in type_stats.index:
            ts = type_stats.loc[r.property_type]
            if not (ts.size_p01 <= r.preferred_built_up_sqft <= ts.size_p99):
                flag(f, "R1", "WARNING", f"preferred size {int(r.preferred_built_up_sqft)} sq ft outside 1–99% "
                                          f"range for {r.property_type} ({int(ts.size_p01)}–{int(ts.size_p99)})")
            if r.min_bedrooms > ts.bed_max or r.min_bathrooms > ts.bath_max:
                flag(f, "R2", "WARNING", f"room minimum above any {r.property_type} in the catalogue")
        # ---- S: soft preferences on the eligible set
        prof = {"preferred_area": r.preferred_area, "max_budget_rm": float(r.max_budget_rm),
                "property_type": r.property_type, "min_bedrooms": float(r.min_bedrooms) or None,
                "min_bathrooms": float(r.min_bathrooms) or None,
                "parking_required": r.parking_required == 1, "furnished_only": r.furnished_only == 1,
                "tenure": None if r.tenure == "No preference" else r.tenure}
        elig, _ = apply_hard_constraints(cat, prof)
        if len(elig) >= 2:
            sizes = elig["built_up_area"]
            if r.preferred_built_up_sqft < sizes.min() or r.preferred_built_up_sqft > sizes.max():
                flag(f, "S1", "INFO", f"preferred size outside eligible range ({int(sizes.min())}–{int(sizes.max())})")
            for g in fac:
                cols = FACILITY_COLUMNS.get(g, [])
                present = (elig[cols].fillna(0).sum(axis=1) > 0) if cols else pd.Series(dtype=bool)
                if len(present) and present.nunique() == 1:
                    flag(f, "S2", "INFO", f"'{g}' is {'present for all' if present.iloc[0] else 'absent for all'} "
                                          "eligible properties (no ranking effect)")
            if elig["price"].nunique() == 1 and elig["built_up_area"].nunique() == 1:
                flag(f, "S3", "INFO", "all eligible properties share price and size")
        # ---- B: eligibility role
        role = roles.loc[r.query_id, "benchmark_role"] if r.query_id in roles.index else "missing"
        n_elig = int(roles.loc[r.query_id, "eligible_count"]) if r.query_id in roles.index else len(elig)
        if role == "empty_result_test" and r.difficulty == "standard":
            flag(f, "B1", "WARNING", "ordinary profile with no eligible property")
        sev_rank = {"ERROR": 3, "WARNING": 2, "INFO": 1}
        worst = max((sev_rank[s] for _, s, _ in f), default=0)
        rows.append({
            "query_id": r.query_id,
            "partition": r.partition,
            "difficulty": r.difficulty,
            "benchmark_role": role,
            "eligible_count": n_elig,
            "auto_status": {3: "ERROR", 2: "REVIEW", 1: "OK (info)", 0: "OK"}[worst],
            "flag_codes": ",".join(c for c, _, _ in f),
            "flags": " | ".join(f"[{s}] {m}" for _, s, m in f),
            "suggested_action": ("fix before freezing" if worst == 3 else
                                 "check and decide" if worst == 2 else "accept"),
            "reviewer_decision": "",       # accept / revise / drop
            "reviewer_comment": "",
            "decided_by": "",
        })

    sheet = pd.DataFrame(rows)
    # Keep decisions already recorded in an existing sheet (rerunning must not wipe them).
    if OUT_SHEET.exists():
        old = pd.read_csv(OUT_SHEET, dtype=str).fillna("")
        keep_cols = [c for c in ("reviewer_decision", "reviewer_comment", "decided_by") if c in old]
        if keep_cols:
            old = old.set_index("query_id")[keep_cols]
            sheet = sheet.set_index("query_id")
            for c in keep_cols:
                sheet[c] = old[c].reindex(sheet.index).fillna(sheet.get(c, ""))
            sheet = sheet.reset_index()
    sheet.to_csv(OUT_SHEET, index=False)
    summary = {
        "queries": int(len(sheet)),
        "auto_status": sheet["auto_status"].value_counts().to_dict(),
        "auto_status_by_role": sheet.groupby(["benchmark_role", "auto_status"]).size()
        .unstack(fill_value=0).to_dict(orient="index"),
        "flags": {f"{c} {s}": n for (c, s), n in sorted(counts.items())},
        "coverage": {
            "districts": int(q["preferred_area"].nunique()),
            "property_types": q["property_type"].value_counts().to_dict(),
            "main_priority": q["main_priority"].value_counts().to_dict(),
            "facility_groups_per_query": q["nearby_facilities"].apply(lambda v: len(facilities_of(v)))
            .value_counts().sort_index().to_dict(),
            "partition": q.groupby(["partition", "difficulty"]).size().unstack(fill_value=0)
            .to_dict(orient="index"),
        },
        "global_notes": [
            "parking_required: car_parks was median-imputed for missing values in preprocessing, so the "
            "requirement currently excludes no property.",
            "tenure: missing tenure was mode-imputed (Freehold), so a Freehold requirement accepts listings "
            "with unknown tenure.",
            "Automated checks do not replace the independent human review (Section 3.7).",
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Reviewed {len(sheet)} profiles: {summary['auto_status']}")
    for k, v in summary["flags"].items():
        print(f"  {k}: {v}")
    print(f"Sheet:   {OUT_SHEET}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
