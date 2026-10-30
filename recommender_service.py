"""In-process recommendation service used by app.py (POST /api/recommend).

Turns the web search profile into the same pipeline the scripts run:
    hard constraints -> direct preference score -> CPG neighbours + graph support -> ranking

The scoring functions are imported from the existing modules, so the web app, the CLI
and the experiments share one implementation:
    direct preference : baseline_models/direct_preference_baseline.py  (score_row)
    graph + ranking   : ranking/cpg_ranker.py                          (rank_candidates)

Search profile (field names follow frozen_queries_set.csv):
    preferred_area (always a hard constraint), max_budget_rm, property_type, min_bedrooms,
    min_bathrooms, parking_required, furnished_only, tenure,
    main_priority, nearby_facilities (list), preferred_built_up_sqft
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for sub in ("baseline_models", "graph", "market_context", "ranking"):
    sys.path.insert(0, str(ROOT / sub))

import direct_preference_baseline as dpb  # noqa: E402
from build_comparable_graph import ComparableGraph  # noqa: E402
from cpg_ranker import G_AVAILABLE, RankingConfig, rank_candidates  # noqa: E402

DATA_PATH = ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
MARKET_PATH = ROOT / "outputs" / "market_context" / "market_context_scores.csv"

# ----------------------------------------------------------------------------- frozen mappings
# Form facility label -> dataset count columns.
FACILITY_COLUMNS = {
    "Public transport": ["train_station_count_3km"],
    "Schools": ["school_count_3km"],
    "Retail & groceries": ["shopping_mall_count_3km", "supermarket_count_3km"],
    "Parks & recreation": ["park_count_1km"],
}
FACILITY_LABELS = {
    "train_station_count_3km": "train stations within 3 km",
    "school_count_3km": "schools within 3 km",
    "shopping_mall_count_3km": "shopping malls within 3 km",
    "supermarket_count_3km": "supermarkets within 3 km",
    "park_count_1km": "parks within 1 km",
}
FURNISHED_VALUES = {"partially furnished", "fully furnished"}

# Direct-preference weights for the web search profile ("balanced").
BASE_WEIGHTS = {
    "price_value_fit": 0.25,   # lower asking price relative to budget
    "size_fit": 0.25,          # closeness to preferred built-up area
    "facility_fit": 0.15,      # selected nearby facilities (active only when any selected)
    # the remaining baseline components stay inactive: those attributes are hard constraints
}
PRIORITY_COMPONENT = {
    "price": "price_value_fit",
    "size": "size_fit",
    "facilities": "facility_fit",
}
PRIORITY_MULTIPLIER = 2.0  # main priority doubles that component's weight
RECOMMENDED_K = 5          # Top-K shown as "Recommended" (= evaluated K)
VIEW_MORE_LIMIT = 20       # at most 20 further eligible properties under "View more"
TOP_N_RETURNED = RECOMMENDED_K + VIEW_MORE_LIMIT   # None = return every eligible property
MAX_NEIGHBOURS_RETURNED = 8


class QueryError(ValueError):
    """Invalid search profile (returned to the browser as HTTP 400)."""


# ----------------------------------------------------------------------------- resources
class _Resources:
    _lock = threading.Lock()
    _loaded = None

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._loaded is None:
                catalogue = pd.read_csv(DATA_PATH)
                cpg = ComparableGraph.load()
                market = pd.read_csv(MARKET_PATH) if MARKET_PATH.exists() else None
                missing = set(catalogue["Property_ID"]) - set(cpg.graph.nodes)
                if missing:
                    raise RuntimeError(
                        f"{len(missing)} catalogue properties are not in the graph. "
                        "Rebuild it: python graph/build_comparable_graph.py --rebuild")
                cls._loaded = (catalogue, cpg, market)
            return cls._loaded


# ----------------------------------------------------------------------------- query parsing
def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(v)


def _number(v, name, minimum=0.0):
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise QueryError(f"{name} must be a number.")
    if not np.isfinite(x) or x < minimum:
        raise QueryError(f"{name} must be at least {minimum:g}.")
    return x


def _facilities(v) -> list[str]:
    if v in (None, "", "No preference", "Any"):
        return []
    # benchmark files store several facilities as "A;B"; the web form sends a list
    items = [x.strip() for x in v.split(";") if x.strip()] if isinstance(v, str) else list(v)
    unknown = [x for x in items if x not in FACILITY_COLUMNS]
    if unknown:
        raise QueryError(f"Unknown nearby facility: {', '.join(map(str, unknown))}")
    return sorted(set(items), key=list(FACILITY_COLUMNS).index)


def parse_profile(raw: dict) -> dict:
    budget = _number(raw.get("max_budget_rm"), "Maximum budget", minimum=1)
    if budget is None:
        raise QueryError("Maximum budget is required.")
    area = str(raw.get("preferred_area") or "").strip()
    if not area or area.casefold() == "any":
        raise QueryError("Preferred area (district) is required.")
    ptype = str(raw.get("property_type") or "Any").strip()
    tenure = str(raw.get("tenure") or "No preference").strip()
    priority = str(raw.get("main_priority") or "balanced").strip().casefold()
    if priority not in {"balanced", *PRIORITY_COMPONENT}:
        raise QueryError(f"Unknown main priority: {priority}")
    facilities = _facilities(raw.get("nearby_facilities"))
    if priority == "facilities" and not facilities:
        raise QueryError("Main priority is facilities but no nearby facility was selected.")
    size = _number(raw.get("preferred_built_up_sqft"), "Preferred built-up area")
    return {
        "preferred_area": area,
        "max_budget_rm": budget,
        "property_type": None if ptype.casefold() == "any" else ptype,
        "min_bedrooms": _number(raw.get("min_bedrooms"), "Minimum bedrooms") or None,
        "min_bathrooms": _number(raw.get("min_bathrooms"), "Minimum bathrooms") or None,
        "parking_required": _bool(raw.get("parking_required", False)),
        "furnished_only": _bool(raw.get("furnished_only", False)),
        "tenure": None if tenure.casefold() in {"no preference", "any"} else tenure,
        "main_priority": priority,
        "nearby_facilities": facilities,
        "preferred_built_up_sqft": size or None,
        # Optional, user-selected: include market context in the final score (full_market variant).
        "include_market_context": _bool(raw.get("include_market_context", False)),
    }


# ----------------------------------------------------------------------------- hard constraints
def apply_hard_constraints(df: pd.DataFrame, prof: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Conjunction of all hard constraints (Eq. 4.13). Missing values never pass."""
    eligible = df
    audit = []

    def gate(name, value, mask):
        nonlocal eligible
        before = len(eligible)
        eligible = eligible.loc[mask.fillna(False).astype(bool)]
        audit.append({"constraint": name, "value": value, "candidates_before": before,
                      "excluded": before - len(eligible), "candidates_after": len(eligible)})

    num = lambda c: pd.to_numeric(eligible[c], errors="coerce")  # noqa: E731
    txt = lambda c: eligible[c].astype("string").str.strip().str.casefold()  # noqa: E731

    gate("max_budget_rm", prof["max_budget_rm"], num("price") <= prof["max_budget_rm"])
    gate("preferred_area (district)", prof["preferred_area"],
         txt("district") == prof["preferred_area"].casefold())
    if prof["property_type"]:
        gate("property_type", prof["property_type"], txt("property_type") == prof["property_type"].casefold())
    if prof["min_bedrooms"]:
        gate("min_bedrooms", prof["min_bedrooms"], num("bedroom") >= prof["min_bedrooms"])
    if prof["min_bathrooms"]:
        gate("min_bathrooms", prof["min_bathrooms"], num("bathroom") >= prof["min_bathrooms"])
    if prof["parking_required"]:
        gate("parking_required", True, num("car_parks") >= 1)
    if prof["furnished_only"]:
        gate("furnished_only", "partially or fully furnished", txt("furnishing").isin(FURNISHED_VALUES))
    if prof["tenure"]:
        gate("tenure", prof["tenure"], txt("tenure") == prof["tenure"].casefold())
    return eligible.copy(), audit


# ----------------------------------------------------------------------------- direct preference
def build_direct_query(prof: dict) -> dict:
    weights = dict(BASE_WEIGHTS)
    comp = PRIORITY_COMPONENT.get(prof["main_priority"])
    if comp:
        weights[comp] *= PRIORITY_MULTIPLIER
    cols = [c for f in prof["nearby_facilities"] for c in FACILITY_COLUMNS[f]]
    return {
        "query_id": "WEB",
        "preferences": {
            "reference_budget": prof["max_budget_rm"],
            "target_price": None,
            "preferred_property_type": None,
            "target_bedroom": None,
            "target_bathroom": None,
            "target_built_up_area": prof["preferred_built_up_sqft"],
            "preferred_area": None,
            # preferred area is a hard constraint -> not scored again as a preference
            "preferred_district": None,
            "preferred_state": None,
            "preferred_tenure": None,
            "preferred_furnishing": None,
            "preferred_facilities": cols,
        },
        "weights": weights,
    }


def score_direct(eligible: pd.DataFrame, query: dict) -> pd.DataFrame:
    cols = {k: dpb.col_for(eligible, k) for k in dpb.ALIASES if k != "id"}
    scores = pd.DataFrame([dpb.score_row(row, cols, query) for _, row in eligible.iterrows()],
                          index=eligible.index)
    out = pd.concat([eligible, scores], axis=1)
    out["query_id"] = query["query_id"]
    return out


# ----------------------------------------------------------------------------- response
def _f(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return round(float(v), nd)


def _name(row) -> str:
    addr = str(row.get("original_address") or "").split(",")[0].strip()
    return addr or str(row.get("area") or "Property").title()


def _facility_list(row) -> list[str]:
    out = []
    for col, label in FACILITY_LABELS.items():
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v) and v > 0:
            out.append(f"{int(v)} {label}")
    return out


DIRECT_LABELS = {
    "price_value_fit": "Asking price relative to budget",
    "size_fit": "Built-up area vs preferred size",
    "location_fit": "Preferred district",
    "facility_fit": "Selected nearby facilities",
}


def _direct_reasons(row, weights) -> list[dict]:
    """Direct-preference evidence ordered by weighted contribution (Section 4.8.2)."""
    active = {k: row.get(f"score_{k}") for k in weights if pd.notna(row.get(f"score_{k}"))}
    denom = sum(weights[k] for k in active)
    reasons = []
    for k, m in active.items():
        ev = row.get(f"evidence_{k}")
        try:
            ev = json.loads(ev) if isinstance(ev, str) and ev.startswith("{") else ev
        except json.JSONDecodeError:
            pass
        reasons.append({"preference": k, "label": DIRECT_LABELS.get(k, k), "match": _f(m),
                        "weight": _f(weights[k] / denom), "contribution": _f(weights[k] * m / denom),
                        "evidence": ev})
    return sorted(reasons, key=lambda r: -(r["contribution"] or 0))


def _notes(reasons, row) -> str:
    parts = []
    for r in reasons[:3]:
        k, m = r["preference"], r["match"]
        if k == "price_value_fit":
            parts.append(f"asking price RM {int(row['price']):,} is {round((1 - m) * 100)}% of the stated budget")
        elif k == "size_fit":
            ev = r["evidence"] if isinstance(r["evidence"], dict) else {}
            parts.append(f"built-up area {float(row['built_up_area']):,.0f} sq ft vs preferred "
                         f"{float(ev.get('target', 0)):,.0f} sq ft (match {round(m * 100)}%)")
        elif k == "location_fit":
            parts.append("in the preferred district" if m >= 1 else "outside the preferred district")
        elif k == "facility_fit":
            parts.append(f"{round(m * 100)}% of the selected facility types are present nearby")
    return ("Direct evidence: " + "; ".join(parts) + ".") if parts else \
        "No soft preference could be scored for this property."


def _market_context(mkt, pid) -> dict:
    """Market-context evidence for display only (not used by the "full" ranking)."""
    if mkt is None or pid not in mkt.index:
        return {"status": "unavailable", "reason": "no_market_context_record"}
    m = mkt.loc[pid]
    if m["market_context_status"] != "available":
        return {"status": "unavailable", "reason": str(m["market_context_unavailable_reason"])}
    return {
        "status": "available",
        "score": _f(m["market_context_score"]),
        "z": _f(m["market_z"], 2),
        "askingPpsf": _f(m["asking_ppsf"], 0),
        "refMean": _f(m["ref_ppsf_mean"], 0),
        "refStd": _f(m["ref_ppsf_std"], 0),
        "refQ1": _f(m["ref_ppsf_q1"], 0),
        "refQ3": _f(m["ref_ppsf_q3"], 0),
        "ratio": _f(m["ppsf_ratio_to_reference_mean"], 2),
        "position": str(m["ppsf_position_vs_reference"]),
        "refLevel": str(m["ref_level"]),
        "configId": str(m["market_context_config_id"]),
    }


def recommend(raw_profile: dict) -> dict:
    started = time.perf_counter()
    prof = parse_profile(raw_profile)
    catalogue, cpg, market = _Resources.get()

    eligible, audit = apply_hard_constraints(catalogue, prof)
    base = {"profile": prof, "constraint_audit": audit, "catalogue_size": int(len(catalogue)),
            "eligible_count": int(len(eligible)), "graph_config_id": cpg.config_id}
    if eligible.empty:
        return {**base, "status": "empty_result_constraints_not_relaxed", "properties": [],
                "elapsed_seconds": round(time.perf_counter() - started, 3)}

    query = build_direct_query(prof)
    direct = score_direct(eligible, query)
    # Default = evaluated CPG ("full"). If the user ticks "Include market context", the separately
    # reported "full_market" variant is used: M joins the final score when valid (Eq. 4.21).
    use_market = prof["include_market_context"] and market is not None
    cfg = RankingConfig(variant="full_market" if use_market else "full", top_k=RECOMMENDED_K)
    ranked, nb_evidence = rank_candidates(direct, cfg, cpg, market)

    nb_by = {pid: g for pid, g in nb_evidence.groupby("Property_ID")} if not nb_evidence.empty else {}
    info = catalogue.set_index("Property_ID")
    mkt = market.set_index("Property_ID") if market is not None else None
    props = []
    shown = ranked if TOP_N_RETURNED is None else ranked.head(TOP_N_RETURNED)
    for _, r in shown.iterrows():
        pid = r["Property_ID"]
        c = info.loc[pid]
        reasons = _direct_reasons(r, query["weights"])
        nbs = []
        if pid in nb_by:
            for _, n in nb_by[pid].head(MAX_NEIGHBOURS_RETURNED).iterrows():
                nc = info.loc[n["neighbour_Property_ID"]]
                nbs.append({
                    "id": str(n["neighbour_Property_ID"]),
                    "name": _name(nc),
                    "district": str(nc["district"]).title(),
                    "price": _f(nc["price"], 0),
                    "similarity": _f(n["edge_similarity"]),
                    "neighbourDirect": _f(n["neighbour_direct_score"]),
                    "supportShare": _f(n["neighbour_support_share"]),
                    "locationResolution": n.get("location_resolution"),
                    "distanceKm": _f(n.get("distance_km"), 2),
                    "components": {k[4:]: _f(n[k]) for k in n.index if k.startswith("sim_")},
                })
        graph_ok = r["graph_support_status"] == G_AVAILABLE
        market_ok = use_market and r.get("market_context_status") == "available"
        resolutions = {n["locationResolution"] for n in nbs}
        props.append({
            "id": str(pid),
            "rank": int(r["cpg_rank"]),
            "recommended": int(r["cpg_rank"]) <= RECOMMENDED_K,
            "name": _name(c),
            "area": str(c["area"]).title(),
            "district": str(c["district"]).title(),
            "state": str(c["state"]),
            "type": str(c["property_type"]),
            "tenure": str(c["tenure"]),
            "furnishing": str(c["furnishing"]),
            "price": _f(c["price"], 0),
            "size": _f(c["built_up_area"], 0),
            "beds": int(c["bedroom"]),
            "baths": int(c["bathroom"]),
            "parking": bool(pd.to_numeric(c["car_parks"], errors="coerce") >= 1),
            "carParks": _f(c["car_parks"], 0),
            "url": str(c.get("original_link") or ""),
            "facilities": _facility_list(c),
            "score": _f(r["final_score"]),
            "direct": _f(r["direct_preference_score"]),
            "graph": _f(r["graph_support"]) if graph_ok else None,
            "displayDirect": _f(r["direct_preference_score"]),
            "displayGraph": _f(r["graph_support"]) if graph_ok else None,
            "graphStatus": r["graph_support_status"],
            "trace": {"contribD": _f(r["contribution_direct"]),
                      "contribG": _f(r["contribution_graph"]) if graph_ok else None,
                      "weightD": _f(r["weight_direct_normalised"]),
                      "weightG": _f(r["weight_graph_normalised"]) if graph_ok else None,
                      "contribM": _f(r["contribution_market"]) if market_ok else None,
                      "weightM": _f(r["weight_market_normalised"]) if market_ok else None},
            "directReasons": reasons,
            "notes": _notes(reasons, c),
            "neighbors": [n["id"] for n in nbs],
            "neighbourDetails": nbs,
            "qualifiedNeighbours": int(r["qualified_neighbours"]),
            "locationEvidence": ("District-level fallback" if "district" in resolutions
                                 else "Coordinate-level"),
            "marketContext": {**_market_context(mkt, pid), "usedInRanking": bool(market_ok)},
        })

    return {**base, "status": "success", "properties": props,
            "graph_support_available": int(ranked["graph_support_status"].eq(G_AVAILABLE).sum()),
            "ranking_config_id": cfg.config_id(),
            "market_context_used_in_ranking": "market" in cfg.components,
            "market_context_requested": prof["include_market_context"],
            "ranking_variant": cfg.variant,
            "recommended_k": RECOMMENDED_K,
            "view_more_limit": VIEW_MORE_LIMIT,
            "direct_weights": query["weights"],
            "elapsed_seconds": round(time.perf_counter() - started, 3)}
