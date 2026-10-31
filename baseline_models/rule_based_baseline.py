"""Rule-based baseline (RB) for the frozen query set - complete implementation.

Everything this baseline needs is in this file: eligibility, ranking, its Top-5 output, the
shared eligibility summary and the pooled relevance-label file.

    python baseline_models/rule_based_baseline.py

Model - a deterministic filter-then-sort with ONE criterion per query (no weights, no
comparable-property graph, no similarity measure):

    main_priority = price / balanced -> lowest asking price within budget
    main_priority = size             -> built-up area closest to preferred_built_up_sqft
    main_priority = facilities       -> most of the selected facility types present nearby
    ties broken by Property_ID, so the order is fully reproducible.

Outputs
    outputs/baseline/rule_based_top5.csv         Top-5 per query (821 rows: 732 + 89 empty)
    outputs/baseline/eligibility_summary.csv     per query, both models' eligibility check
    frozen_queries/relevance_labels_v2.csv       graded relevance for every pair either model
                                                 returned (rebuilt from whatever Top-5 files
                                                 exist in outputs/baseline/)

The reporting functions at the bottom are shared: content_based_baseline.py imports them, so
the grading rubric is defined exactly once.

GRADING RUBRIC (REL-v1, deterministic, attributes only - no model score, no method identity)
    main_priority=size        gap = |area - preferred| / preferred
                              3 <=5%   2 <=15%   1 <=35%   0 >35%
    main_priority=price       pct = asking price / budget
                              3 <=60%  2 <=85%   1 <=98%   0 >98%
    main_priority=facilities  share of selected facility types present
                              3 =100%  2 >=67%   1 >=50%   0 <50%
    main_priority=balanced    pct and gap together
                              3 pct<=70% & gap<=10%   2 pct<=85% & gap<=20%
                              1 pct<=98% & gap<=35%   0 otherwise
    One-step downgrade when a stated secondary preference is badly missed (area >30% off the
    preferred size, or none of the selected facility types present).
    If no eligible listing has the selected facility it cannot differentiate candidates
    (benchmark review flag S2), so it is ignored and the price/size tier is used instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import recommender_service as svc  # noqa: E402

METHOD = "rule_based"
TOP_K = 5
QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
OUT_DIR = ROOT / "outputs" / "baseline"
OUT = OUT_DIR / "rule_based_top5.csv"
OUT_ELIG = OUT_DIR / "eligibility_summary.csv"
OUT_LABELS = ROOT / "frozen_queries" / "relevance_labels_v2.csv"

EVAL_SPLITS = {"validation", "test"}
RELEVANCE_TEXT = {3: "highly relevant", 2: "relevant", 1: "marginal", 0: "not relevant"}
LABEL_METHOD = "auto rubric REL-v1"

FACILITY_COLUMNS = {
    "Public transport": ["train_station_count_3km"],
    "Schools": ["school_count_3km"],
    "Retail & groceries": ["shopping_mall_count_3km", "supermarket_count_3km"],
    "Parks & recreation": ["park_count_1km"],
}

# Schema shared by both baselines, so the two Top-5 files can be concatenated.
TOP5_COLUMNS = ["method", "query_id", "split", "eligible_count", "rank", "Property_ID",
                "listing", "district", "property_type", "asking_price_rm", "built_up_sqft",
                "bedrooms", "bathrooms", "score", "criterion", "status"]
ELIG_COLUMNS = ["query_id", "split", "eligible_count", "enough_for_top5", "emptied_by_constraint",
                "rule_based_returned", "rule_based_check",
                "content_based_returned", "content_based_check", "note"]
LABEL_COLUMNS = ["query_id", "split", "Property_ID", "listing", "district", "property_type",
                 "asking_price_rm", "budget_rm", "ask_pct_of_budget", "built_up_sqft",
                 "preferred_sqft", "size_gap_pct", "bedrooms", "bathrooms", "main_priority",
                 "facility_match", "relevance", "relevance_text", "label_reason",
                 "label_method", "reviewer_note"]


def _int(value):
    num = pd.to_numeric(value, errors="coerce")
    return int(num) if pd.notna(num) else ""


def _listing_name(row) -> str:
    return str(row.get("original_address") or "").split(",")[0].strip().title()


def _context(eligible: pd.DataFrame, prof: dict) -> dict:
    """Query facts the grading step needs (budget, preferred size, achievable facilities)."""
    selected = [c for f in prof["nearby_facilities"] for c in FACILITY_COLUMNS.get(f, [])]
    useful = [c for c in selected
              if c in eligible.columns
              and (pd.to_numeric(eligible[c], errors="coerce").fillna(0) > 0).any()]
    return {"budget": float(prof["max_budget_rm"]),
            "preferred_sqft": prof["preferred_built_up_sqft"],
            "priority": prof["main_priority"], "facility_cols": useful}


# --------------------------------------------------------------------------- model
def score(eligible: pd.DataFrame, prof: dict) -> tuple[pd.Series, pd.Series, str]:
    """(sort key ascending, 0-1 rule score, criterion text) - one criterion, no weights."""
    price = pd.to_numeric(eligible["price"], errors="coerce")
    budget = float(prof["max_budget_rm"])
    preferred = prof["preferred_built_up_sqft"]
    facility_cols = [c for f in prof["nearby_facilities"]
                     for c in FACILITY_COLUMNS.get(f, [])]

    if prof["main_priority"] == "size" and preferred:
        size = pd.to_numeric(eligible["built_up_area"], errors="coerce")
        gap = (size - float(preferred)).abs()
        return (gap.fillna(np.inf), (1 - gap / float(preferred)).clip(0, 1),
                "closest built-up area to the preferred size")
    if prof["main_priority"] == "facilities" and facility_cols:
        hits = None
        for col in facility_cols:
            has = (pd.to_numeric(eligible[col], errors="coerce").fillna(0) > 0).astype(int)
            hits = has if hits is None else hits + has
        return (-hits, (hits / len(facility_cols)).clip(0, 1),
                "most of the selected facility types present nearby")
    return (price.fillna(np.inf), (1 - price / budget).clip(0, 1),
            "lowest asking price within budget")


def rank(eligible: pd.DataFrame, prof: dict, k: int = TOP_K) -> tuple[pd.DataFrame, str]:
    """Top-k by the query's single rule, ties broken by Property_ID."""
    key, rule_score, criterion = score(eligible, prof)
    ranked = (eligible.assign(_key=key, top_score=rule_score)
              .sort_values(["_key", "Property_ID"], kind="mergesort")
              .drop(columns=["_key"]).reset_index(drop=True))
    return ranked.head(k), criterion


# --------------------------------------------------------------------------- run the query set
def run(queries: pd.DataFrame, catalogue: pd.DataFrame,
        k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    """Eligibility + Top-k for every query. Returns (rows, per-query eligibility records)."""
    info = catalogue.set_index("Property_ID")
    rows, eligibility = [], {}

    for _, q in queries.iterrows():
        qid, split = q["query_id"], q.get("split", q.get("partition", ""))
        try:
            prof = svc.parse_profile(q.to_dict())
        except svc.QueryError as exc:
            rows.append({"method": METHOD, "query_id": qid, "split": split,
                         "status": f"invalid_query: {exc}"})
            continue

        eligible, audit = svc.apply_hard_constraints(catalogue, prof)
        emptied_by = next((a["constraint"] for a in audit if a["candidates_after"] == 0), "")
        eligibility[qid] = {"split": split, "eligible_count": int(len(eligible)),
                            "emptied_by": emptied_by, "context": _context(eligible, prof)}

        if eligible.empty:
            rows.append({"method": METHOD, "query_id": qid, "split": split,
                         "eligible_count": 0, "status": "no_eligible_property"})
            continue

        top, criterion = rank(eligible, prof, k)
        status = "ok" if len(eligible) >= k else "fewer_than_top_n_eligible"
        for position, (_, r) in enumerate(top.iterrows(), start=1):
            pid = int(r["Property_ID"])
            listing = info.loc[pid]
            rows.append({
                "method": METHOD, "query_id": qid, "split": split,
                "eligible_count": int(len(eligible)), "rank": position, "Property_ID": pid,
                "listing": _listing_name(listing),
                "district": str(listing["district"]).title(),
                "property_type": str(listing["property_type"]),
                "asking_price_rm": _int(listing["price"]),
                "built_up_sqft": _int(listing["built_up_area"]),
                "bedrooms": _int(listing["bedroom"]),
                "bathrooms": _int(listing["bathroom"]),
                "score": round(float(r["top_score"]), 4),
                "criterion": criterion, "status": status,
            })

    return pd.DataFrame(rows, columns=TOP5_COLUMNS), eligibility


# --------------------------------------------------------------------------- shared reporting
def write_eligibility_summary(eligibility: dict, path: Path = OUT_ELIG) -> pd.DataFrame:
    """One row per query for BOTH models (they share the same eligible set by construction).

    Both baselines rank the entire eligible set and return its Top-5, so each returns
    min(5, eligible_count) rows: the two 'returned' columns are identical.
    """
    rows = []
    for qid in sorted(eligibility):
        e = eligibility[qid]
        n = e["eligible_count"]
        check = "ok" if n >= TOP_K else ("empty_eligible_set" if n == 0 else "fewer_than_top5")
        rows.append({
            "query_id": qid, "split": e["split"], "eligible_count": n,
            "enough_for_top5": "yes" if n >= TOP_K else "no",
            "emptied_by_constraint": e["emptied_by"],
            "rule_based_returned": min(TOP_K, n), "rule_based_check": check,
            "content_based_returned": min(TOP_K, n), "content_based_check": check,
            "note": (f"no listing passes every constraint (emptied by {e['emptied_by']})"
                     if n == 0 else "" if n >= TOP_K else f"only {n} eligible (Top-5 incomplete)"),
        })
    frame = pd.DataFrame(rows, columns=ELIG_COLUMNS)
    numeric = pd.to_numeric(frame["eligible_count"], errors="coerce")
    total = {
        "query_id": "TOTAL", "split": "all", "eligible_count": int(numeric.sum()),
        "enough_for_top5": f"{int((numeric >= TOP_K).sum())} of {len(frame)}",
        "emptied_by_constraint": "",
        "rule_based_returned": int(frame["rule_based_returned"].sum()),
        "rule_based_check": f"{int((frame['rule_based_check'] == 'ok').sum())} ok / "
                            f"{int((frame['rule_based_check'] == 'fewer_than_top5').sum())} short / "
                            f"{int((frame['rule_based_check'] == 'empty_eligible_set').sum())} empty",
        "content_based_returned": int(frame["content_based_returned"].sum()),
        "content_based_check": f"{int((frame['content_based_check'] == 'ok').sum())} ok / "
                               f"{int((frame['content_based_check'] == 'fewer_than_top5').sum())} short / "
                               f"{int((frame['content_based_check'] == 'empty_eligible_set').sum())} empty",
        "note": ("eligible_count is computed once with recommender_service.apply_hard_constraints "
                 "and used by both models, so the counts are identical by construction; "
                 "'ok' needs at least 5 candidates to fill a Top-5"),
    }
    out = pd.concat([frame, pd.DataFrame([total], columns=ELIG_COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _tier(priority: str, pct, gap, fac) -> int:
    if priority == "size" and gap is not None:
        return 3 if gap <= 0.05 else 2 if gap <= 0.15 else 1 if gap <= 0.35 else 0
    if priority == "price" and pct is not None:
        return 3 if pct <= 0.60 else 2 if pct <= 0.85 else 1 if pct <= 0.98 else 0
    if priority == "facilities" and fac is not None:
        return 3 if fac >= 0.999 else 2 if fac >= 0.67 else 1 if fac >= 0.50 else 0
    strong = pct is not None and pct <= 0.70 and (gap is None or gap <= 0.10)
    good = pct is not None and pct <= 0.85 and (gap is None or gap <= 0.20)
    marginal = pct is not None and pct <= 0.98 and (gap is None or gap <= 0.35)
    return 3 if strong else 2 if good else 1 if marginal else 0


def grade_pair(context: dict, listing) -> tuple[int, str]:
    """(relevance 0-3, one-line reason) from query and listing attributes only."""
    priority = context["priority"]
    price = pd.to_numeric(listing["price"], errors="coerce")
    size = pd.to_numeric(listing["built_up_area"], errors="coerce")
    pct = float(price) / context["budget"] if pd.notna(price) else None
    pref = context["preferred_sqft"]
    gap = abs(float(size) - pref) / pref if (pd.notna(size) and pref) else None
    cols = context["facility_cols"]
    hits = sum(1 for c in cols if pd.to_numeric(listing[c], errors="coerce") > 0)
    fac = hits / len(cols) if cols else None

    effective = priority if (priority != "facilities" or fac is not None) else "balanced"
    grade = _tier(effective, pct, gap, fac)

    bits = []
    if pct is not None:
        bits.append(f"asks {pct:.1%} of the budget")
    if gap is not None:
        bits.append(f"area {gap:.1%} off the preferred {pref:,.0f} sq ft")
    if cols:
        bits.append(f"{hits}/{len(cols)} selected facilities present")
    reason = f"{effective} priority: " + "; ".join(bits) if bits else effective
    if priority == "facilities" and fac is None:
        reason = ("facilities priority ignored (no eligible listing has the selected facility); "
                  "graded on price/size: " + "; ".join(bits))

    downgrade = None
    if priority != "size" and gap is not None and gap > 0.30:
        downgrade = "area >30% off a stated preference"
    elif priority != "facilities" and cols and hits == 0:
        downgrade = "none of the selected facilities present"
    if downgrade and grade > 0:
        grade -= 1
        reason += f" (downgraded: {downgrade})"
    return grade, reason


def build_labels(queries: pd.DataFrame, catalogue: pd.DataFrame,
                 path: Path = OUT_LABELS) -> pd.DataFrame:
    """Pool every pair found in outputs/baseline/*_top5.csv and grade it (blind to method).

    The label file must cover BOTH baselines, otherwise the missing model looks artificially
    good on its unjudged items. When only one Top-5 file is present a warning is printed.
    """
    info = catalogue.set_index("Property_ID")
    query_index = queries.set_index("query_id")
    files = sorted(OUT_DIR.glob("*_top5.csv"))
    expected = {"rule_based_top5.csv", "content_based_top5.csv"}
    missing = expected - {f.name for f in files}
    if missing:
        print(f"WARNING: missing {', '.join(sorted(missing))} in outputs/baseline/ - the label "
              f"file is INCOMPLETE. Run that baseline too, then re-run either script to rebuild "
              f"the pooled labels before computing any metric.")

    pairs = set()
    for file in files:
        frame = pd.read_csv(file)
        frame = frame.dropna(subset=["Property_ID"])
        frame = frame[frame["split"].isin(EVAL_SPLITS)]
        pairs |= {(r.query_id, int(r.Property_ID)) for r in frame.itertuples()}

    contexts, rows = {}, []
    for qid, pid in sorted(pairs):
        if qid not in contexts:
            prof = svc.parse_profile(query_index.loc[qid].to_dict())
            eligible, _ = svc.apply_hard_constraints(catalogue, prof)
            contexts[qid] = {"split": str(query_index.loc[qid]["partition"]),
                             **_context(eligible, prof)}
        context = contexts[qid]
        listing = info.loc[pid]
        grade, reason = grade_pair(context, listing)
        price = pd.to_numeric(listing["price"], errors="coerce")
        size = pd.to_numeric(listing["built_up_area"], errors="coerce")
        pref = context["preferred_sqft"]
        cols = context["facility_cols"]
        hits = sum(1 for c in cols if pd.to_numeric(listing[c], errors="coerce") > 0)
        rows.append({
            "query_id": qid, "split": context["split"], "Property_ID": pid,
            "listing": _listing_name(listing),
            "district": str(listing["district"]).title(),
            "property_type": str(listing["property_type"]),
            "asking_price_rm": _int(price), "budget_rm": int(context["budget"]),
            "ask_pct_of_budget": (round(float(price) / context["budget"] * 100)
                                  if pd.notna(price) else ""),
            "built_up_sqft": _int(size),
            "preferred_sqft": int(pref) if pref else "",
            "size_gap_pct": (round(abs(float(size) - pref) / pref * 100)
                             if (pd.notna(size) and pref) else ""),
            "bedrooms": _int(listing["bedroom"]), "bathrooms": _int(listing["bathroom"]),
            "main_priority": context["priority"],
            "facility_match": f"{hits}/{len(cols)}" if cols else "",
            "relevance": grade, "relevance_text": RELEVANCE_TEXT[grade],
            "label_reason": reason, "label_method": LABEL_METHOD, "reviewer_note": "",
        })

    frame = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


# --------------------------------------------------------------------------- entry point
def main() -> None:
    queries = pd.read_csv(QUERIES, dtype=str, keep_default_na=False)
    catalogue = pd.read_csv(svc.DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, eligibility = run(queries, catalogue)
    rows.to_csv(OUT, index=False)
    elig = write_eligibility_summary(eligibility)
    labels = build_labels(queries, catalogue)

    total = elig[elig["query_id"] == "TOTAL"].iloc[0]
    dist = labels["relevance"].value_counts().sort_index().to_dict()
    print(f"rule_based: {int(rows['status'].isin(['ok', 'fewer_than_top_n_eligible']).sum())} "
          f"recommendations over "
          f"{rows.loc[rows['Property_ID'].notna(), 'query_id'].nunique()} queries "
          f"-> {rows['status'].value_counts().to_dict()}")
    print(f"eligibility: {total['enough_for_top5']} queries can fill a Top-5, "
          f"{int((elig['rule_based_check'] == 'empty_eligible_set').sum())} empty")
    print(f"relevance sheet: {len(labels)} pairs over {labels['query_id'].nunique()} queries "
          f"({labels['split'].value_counts().to_dict()}) grades {dist}")
    if len(labels) and len(list(OUT_DIR.glob('*_top5.csv'))) < 2:
        print("NOTE: this file is complete only after both baselines have been run; "
              "re-run either script afterwards to rebuild the pooled labels.")
    print("Saved: outputs/baseline/rule_based_top5.csv, outputs/baseline/eligibility_summary.csv, "
          "frozen_queries/relevance_labels_v2.csv")


if __name__ == "__main__":
    main()
