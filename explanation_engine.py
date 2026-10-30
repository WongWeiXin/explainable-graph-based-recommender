"""Stage 8 + Stage 9: faithful explanations and their programmatic checks.

Stage 8 (explanation engine)
    ``build_explanation`` turns one candidate's score/evidence record (built by
    ``ranking/cpg_ranker.build_evidence``) into the explanation record below.
    The record produced by the ranker is the engine's ONLY fact source, so the UI
    can never show a listing feature, distance or comparable that did not take
    part in scoring.

        1 recommendation rank        6 market-context status
        2 final score                7 location-resolution status
        3 direct-score components    8 unavailable evidence
        4 graph-score components     9 fallback used
        5 comparable neighbours     10 generated explanation

    Allowed wording is limited to what the record can support: hard constraints
    satisfied, property type / bedrooms / bathrooms matched as hard constraints,
    direct preference scores, qualified comparable neighbours, market reference
    availability, limited location precision, and "graph support unavailable".

Stage 9 (faithfulness checks)
    Every generated claim carries ``evidence_ref`` pointers into the record.
    ``validate_claim`` resolves them and re-checks the linkage rules; claims that
    do not resolve are withheld and reported, never displayed. ``check_faithfulness``
    reports the six required checks plus the faithfulness metrics
    (unsupported-claim rate, evidence coverage, score-linked reason precision,
    comparable-neighbour validity, fallback-disclosure accuracy,
    location-disclosure accuracy).

Run ``python explanation_engine.py`` to self-test the engine on synthetic records.
"""
from __future__ import annotations

import sys

ENGINE_VERSION = "EXPL-1.0"
DEFAULT_MAX_NEIGHBOURS = 8
SPARSE_NEIGHBOUR_THRESHOLD = 2
DISTANCE_PRECISION_DP = 1
SAME_VALUE = 0.99

DIRECT_LABELS = {
    "budget_fit": "Price vs target price",
    "price_value_fit": "Asking price relative to the stated budget",
    "property_type_fit": "Property type",
    "bedroom_fit": "Bedrooms",
    "bathroom_fit": "Bathrooms",
    "size_fit": "Built-up area vs preferred size",
    "location_fit": "Preferred location match",
    "tenure_fit": "Tenure",
    "furnishing_fit": "Furnishing",
    "facility_fit": "Selected nearby facilities",
}

HARD_CONSTRAINT_LABELS = {
    "max_budget_rm": "maximum budget",
    "preferred_area (district)": "preferred district",
    "property_type": "property type",
    "min_bedrooms": "required bedrooms",
    "min_bathrooms": "required bathrooms",
    "parking_required": "parking requirement",
    "furnished_only": "furnishing requirement",
    "tenure": "tenure requirement",
}

# A preference that reached the scorer only because it is enforced as a hard gate.
HARD_CONSTRAINT_FOR_PREFERENCE = {
    "property_type_fit": "property_type",
    "bedroom_fit": "min_bedrooms",
    "bathroom_fit": "min_bathrooms",
    "tenure_fit": "tenure",
    "furnishing_fit": "furnished_only",
    "location_fit": "preferred_area",
}

GRAPH_STATUS_REASONS = {
    "unavailable_unknown": "the candidate's graph evidence is missing from its score record",
    "unavailable_candidate_not_in_graph":
        "the candidate is not a node in the comparable-property graph",
    "unavailable_no_retained_neighbours":
        "the candidate has no retained comparable edge above the similarity threshold",
    "unavailable_no_neighbour_meets_hard_constraints":
        "no retained neighbour also satisfies the current hard constraints",
    "unavailable_neighbours_have_no_direct_score":
        "retained neighbours carry no direct preference score, so support cannot be aggregated",
    "disabled_by_variant": "graph support is disabled in this ranking configuration",
}

MARKET_POSITION_PHRASES = {
    "below_reference_q1": "below the reference-group lower quartile (Q1)",
    "within_reference_iqr": "inside the reference-group interquartile range",
    "above_reference_q3": "above the reference-group upper quartile (Q3)",
}

MARKET_STATUS_REASONS = {
    "disabled_by_variant": "market context is not part of this ranking configuration",
    "unavailable_not_computed": "market context was not computed for this listing",
    "unavailable": "the listing has no valid market reference group",
}

LIMITATIONS = [
    "Only evidence retained in this candidate's score record is shown; listing "
    "features that were not scored are not described.",
    "Comparable similarity indicates similarity between listings, not a valuation "
    "and not a promise of future price.",
    "Asking prices are listing prices, not market valuations.",
    "Market context, when shown, is a reference-group comparison and not a price guarantee.",
    "The ranking reflects the stated search profile and does not claim the property "
    "is the best available option for the user.",
]

CLAIM_ORDER = {
    "score": 0,
    "constraint": 1,
    "direct_reason": 2,
    "graph_reason": 3,
    "comparable": 4,
    "ranking": 5,
    "market": 6,
    "location": 7,
    "distance": 8,
    "disclosure": 9,
}

_MISSING = object()


# --------------------------------------------------------------------- helpers
def _r(value, nd):
    if value is None:
        return None
    try:
        return round(float(value), nd)
    except (TypeError, ValueError):
        return None


def _pct(value, nd=0):
    return f"{round(float(value) * 100, nd)}%"


def _num(value, nd=2):
    return "n/a" if value is None else f"{float(value):.{nd}f}"


def _human(code) -> str:
    return str(code).replace("_", " ")


def _requirement(value) -> str:
    if isinstance(value, bool):
        return "required" if value else "not required"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


def normalise_property_id(value) -> str:
    """Normalise a Property_ID (int, numpy int or float) to its display string."""
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


# ------------------------------------------------------- Stage 9: claim checks
def resolve_ref(record: dict, ref: str):
    """Resolve a reference such as ``comparable_neighbours_shown[0].id`` in ``record``."""
    current = record
    for part in str(ref).split("."):
        if not part:
            continue
        name, _, index = part.partition("[")
        if name:
            if not isinstance(current, dict) or name not in current:
                return _MISSING
            current = current[name]
        if index:
            try:
                position = int(index.rstrip("]"))
            except ValueError:
                return _MISSING
            if not isinstance(current, list) or not -len(current) <= position < len(current):
                return _MISSING
            current = current[position]
    return current


def _close(a, b, tolerance=1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return False


def _shown_neighbours(record: dict) -> list:
    return record.get("comparable_neighbours_shown") or []


def _ranking_problems(record: dict, claim: dict) -> list:
    check = claim.get("ranking_check") or {}
    kind = check.get("type")
    if kind == "largest_support":
        neighbours = _shown_neighbours(record)
        if not neighbours:
            return ["no comparable neighbours to rank"]
        top = max(neighbours, key=lambda n: n.get("support_share") or 0.0)
        if top.get("id") != check.get("neighbour_id"):
            return ["largest-contribution claim does not match the neighbour support shares"]
    elif kind == "dominant_component":
        contributions = (record.get("score_trace") or {}).get("contributions") or {}
        if not contributions:
            return ["no component contributions retained"]
        weight = {k: v for k, v in contributions.items() if v is not None}
        if not weight:
            return ["no component contributions retained"]
        best = max(weight, key=weight.get)
        if best != check.get("component"):
            return ["claimed dominant component is not the largest contribution"]
    else:
        return [f"unsupported ranking check: {kind}"]
    return []


def _distance_problems(record: dict, claim: dict) -> list:
    status = (record.get("location_resolution_status") or {}).get("status")
    if status != "coordinate_level":
        return ["distance stated without coordinate-level location evidence"]
    distance = claim.get("distance") or {}
    raw, shown = distance.get("raw"), distance.get("shown")
    if raw is None or shown is None:
        return ["distance claim has no retained value"]
    raw = raw if isinstance(raw, list) else [raw]
    shown = shown if isinstance(shown, list) else [shown]
    if len(raw) != len(shown) or not raw:
        return ["distance claim does not match the retained distance evidence"]
    for stored, displayed in zip(raw, shown):
        if abs(round(float(stored), DISTANCE_PRECISION_DP) - float(displayed)) > 1e-9:
            return ["distance is more precise than the retained location resolution"]
    return []


def validate_claim(record: dict, claim: dict) -> list:
    """Return the problems found for a claim; an empty list means it is supported."""
    problems = []
    for ref in claim.get("evidence_ref", []):
        if resolve_ref(record, ref) is _MISSING:
            problems.append(f"evidence not present in the record: {ref}")

    kind = claim.get("kind")
    active = record.get("active_score_components") or []
    if kind == "direct_reason":
        scored = record.get("direct_score_components") or {}
        if "direct" not in active or claim.get("component") not in scored:
            problems.append("reason component did not contribute to the final score")
    elif kind == "graph_reason":
        if "graph" not in active or claim.get("component") != "graph":
            problems.append("graph component did not contribute to the final score")
    elif kind == "comparable":
        shown = {n.get("id"): n for n in _shown_neighbours(record)}
        neighbour = shown.get(claim.get("neighbour_id"))
        if neighbour is None:
            problems.append("displayed property is not in the qualified neighbour set")
        elif claim.get("support_share") is not None and not _close(
            neighbour.get("support_share"), claim["support_share"]
        ):
            problems.append("neighbour support share does not match the evidence record")
    elif kind == "ranking":
        problems.extend(_ranking_problems(record, claim))
    elif kind == "constraint":
        gates = {str(g.get("constraint")): g
                 for g in (record.get("hard_constraint_evidence") or [])}
        if claim.get("constraint") not in gates:
            problems.append("constraint is not in the record's hard-constraint evidence")
    elif kind == "distance":
        problems.extend(_distance_problems(record, claim))
    elif kind == "comparable_summary":
        problems.extend(_summary_problems(record, claim))
    elif kind == "market_context":
        problems.extend(_market_problems(record, claim))
    elif kind == "disclosure":
        if claim.get("discloses") not in (record.get("required_disclosures") or []):
            problems.append("disclosure is not required by the evidence record")
    return problems


def _available_blocks(record: dict) -> set:
    blocks = set()
    if record.get("hard_constraint_evidence"):
        blocks.add("constraints")
    if record.get("direct_score_components"):
        blocks.add("direct")
    if (record.get("graph_score_component") or {}).get("available"):
        blocks.add("graph")
    blocks |= {f"neighbour:{n.get('id')}" for n in _shown_neighbours(record)}
    if (record.get("market_context_status") or {}).get("available"):
        blocks.add("market")
    if (record.get("location_resolution_status") or {}).get("status") not in (
        None, "not_applicable_no_neighbour_evidence"
    ):
        blocks.add("location")
    return blocks


def _ratio(numerator, denominator, empty=1.0) -> float:
    return round(numerator / denominator, 4) if denominator else empty


def check_faithfulness(record: dict, accepted: list, rejected: list,
                       required_disclosures: list) -> dict:
    """Stage 9 checks and metrics for one explanation record."""
    checks = []

    # 1. Evidence presence: every displayed fact resolves into the evidence record.
    unresolved = [c for c in accepted if validate_claim(record, c)]
    checks.append({
        "id": "evidence_presence",
        "label": "Evidence presence",
        "passed": not unresolved,
        "detail": (f"{len(accepted)} displayed claim(s) resolve to retained evidence"
                   + (f"; {len(rejected)} unsupported claim(s) withheld" if rejected else "")),
    })

    # 2. Score linkage: named factors are components that entered the final score.
    direct_claims = [c for c in accepted if c.get("kind") == "direct_reason"]
    graph_claims = [c for c in accepted if c.get("kind") == "graph_reason"]
    active = set(record.get("active_score_components") or [])
    scored_direct = set(record.get("direct_score_components") or {})
    bad_direct = [c for c in direct_claims if c.get("component") not in scored_direct]
    bad_graph = [c for c in graph_claims if "graph" not in active]
    checks.append({
        "id": "score_linkage",
        "label": "Score linkage",
        "passed": not bad_direct and not bad_graph,
        "detail": (f"{len(direct_claims) - len(bad_direct)} direct and "
                   f"{len(graph_claims) - len(bad_graph)} graph reason(s) map to scored components"),
    })

    # 3. Comparable linkage: shown comparables belong to the qualified neighbour set.
    comparable_claims = [c for c in accepted if c.get("kind") == "comparable"]
    shown_ids = {n.get("id") for n in _shown_neighbours(record)}
    valid_comparables = [c for c in comparable_claims if c.get("neighbour_id") in shown_ids]
    checks.append({
        "id": "comparable_linkage",
        "label": "Comparable linkage",
        "passed": len(valid_comparables) == len(comparable_claims),
        "detail": (f"{len(valid_comparables)}/{len(comparable_claims)} displayed comparable(s) "
                   "are retained qualified neighbours"),
    })

    # 4. Ranking linkage: "largest"/"dominant" statements match the stored shares.
    ranking_claims = [c for c in accepted if c.get("kind") == "ranking"]
    bad_ranking = [c for c in ranking_claims if _ranking_problems(record, c)]
    checks.append({
        "id": "ranking_linkage",
        "label": "Ranking linkage",
        "passed": not bad_ranking,
        "detail": (f"{len(ranking_claims) - len(bad_ranking)}/{len(ranking_claims)} contribution "
                   "comparison(s) match the retained component shares"),
    })

    # 5. Component coverage: every active score component is retained in the trace.
    active_components = list(record.get("active_score_components") or [])
    trace = record.get("score_trace") or {}
    trace_weights = trace.get("weights") or {}
    trace_contributions = trace.get("contributions") or {}
    uncovered = [name for name in active_components
                 if trace_weights.get(name) is None or trace_contributions.get(name) is None]
    checks.append({
        "id": "component_coverage",
        "label": "Component coverage",
        "passed": not uncovered,
        "detail": (f"{len(active_components) - len(uncovered)}/{len(active_components)} active "
                   "score component(s) carry a retained weight and contribution"
                   + (f"; missing: {', '.join(uncovered)}" if uncovered else "")),
    })

    # 6. Fallback disclosure: unavailable evidence is disclosed, never silently dropped.
    disclosed = {c.get("discloses") for c in accepted if c.get("discloses")}
    missing = [d for d in required_disclosures if d not in disclosed]
    checks.append({
        "id": "fallback_disclosure",
        "label": "Fallback disclosure",
        "passed": not missing,
        "detail": ("no unavailable component to disclose" if not required_disclosures
                   else f"{len(required_disclosures) - len(missing)}/"
                        f"{len(required_disclosures)} required disclosure(s) present"),
    })

    # 7. Location disclosure: no over-precise distance without coordinate evidence.
    distance_claims = [c for c in accepted
                       if c.get("kind") == "distance"
                       or (c.get("kind") == "comparable_summary" and c.get("distance"))]
    bad_distance = [c for c in distance_claims if _distance_problems(record, c)]
    location_status = (record.get("location_resolution_status") or {}).get("status")
    checks.append({
        "id": "location_disclosure",
        "label": "Location disclosure",
        "passed": not bad_distance,
        "detail": (f"{len(distance_claims) - len(bad_distance)} distance statement(s) respect the "
                   f"{_human(location_status)} resolution"),
    })

    available = _available_blocks(record)
    referenced = {c.get("block") for c in accepted if c.get("block")}
    total_claims = len(accepted) + len(rejected)
    solid_reasons = (len(direct_claims) - len(bad_direct)) + (len(graph_claims) - len(bad_graph))

    metrics = {
        "unsupported_explanation_claim_rate": _ratio(len(rejected), total_claims, empty=0.0),
        "evidence_coverage": _ratio(len(available & referenced), len(available)),
        "score_linked_reason_precision": _ratio(
            solid_reasons, len(direct_claims) + len(graph_claims)
        ),
        "comparable_neighbour_validity": _ratio(
            len(valid_comparables), len(comparable_claims)
        ),
        "fallback_disclosure_accuracy": _ratio(
            len(required_disclosures) - len(missing), len(required_disclosures)
        ),
        "location_disclosure_accuracy": _ratio(
            len(distance_claims) - len(bad_distance), len(distance_claims)
        ),
    }

    passed = sum(1 for c in checks if c["passed"])
    return {
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "passed": passed == len(checks),
        "metrics": metrics,
        "claims_displayed": len(accepted),
        "claims_withheld": len(rejected),
    }


# ------------------------------------------------- Stage 8: explanation record
def _hard_constraint_evidence(audit) -> list:
    return [{
        "constraint": gate.get("constraint"),
        "requirement": gate.get("value"),
        "property_value": gate.get("property_value"),
        "candidates_passing": gate.get("candidates_after"),
    } for gate in (audit or [])]


def _direct_components(evidence: dict, direct_weights: dict | None) -> dict:
    """Direct-preference evidence ordered by contribution to D (weight_share x match).

    Contribution uses the same normalisation as the baseline: the preference's share
    of the active direct weight, times its match. The first entry is therefore the
    strongest scored reason, so "the main reason" can be read straight off the record.
    """
    weights = {k: float(v) for k, v in (direct_weights or {}).items() if v}
    matches = [m for m in (evidence.get("direct_preference_matches") or [])
               if m.get("match") is not None]
    denominator = sum(weights.get(m.get("preference"), 0.0) for m in matches) or 0.0
    components = {}
    for match in matches:
        preference = match.get("preference")
        weight = weights.get(preference, 0.0)
        weight_share = weight / denominator if denominator else 0.0
        entry = {
            "label": DIRECT_LABELS.get(preference, preference),
            "match": _r(match.get("match"), 4),
            "weight_share": _r(weight_share, 4),
            "contribution": _r(weight_share * float(match["match"]), 4),
            "active": weight > 0,
            "evidence": match.get("evidence"),
        }
        entry["plain"] = _plain_reason_text(preference, entry)
        entry["comparison"] = _comparison(preference, match.get("evidence"))
        components[preference] = entry
    return dict(sorted(components.items(), key=lambda kv: -(kv[1]["contribution"] or 0.0)))


def _market_component(component: dict | None, market: dict) -> dict:
    """Market component as retained for scoring, or an explicit empty entry."""
    component = component or {}
    available = bool(market.get("available")) and component.get("value") is not None
    return {
        "available": available,
        "value": _r(component.get("value"), 4) if available else None,
        "lambda": _r(component.get("lambda"), 4),
        "normalised_weight": _r(component.get("normalised_weight"), 4) if available else None,
        "contribution": _r(component.get("contribution"), 4) if available else None,
    }


def _graph_component(component: dict | None, graph_support: dict) -> dict:
    component = component or {}
    available = component.get("value") is not None
    return {
        "status": graph_support.get("status") or "unavailable_unknown",
        "available": available,
        "value": _r(component.get("value"), 4) if available else None,
        "lambda": _r(component.get("lambda"), 4),
        "normalised_weight": _r(component.get("normalised_weight"), 4),
        "contribution": _r(component.get("contribution"), 4),
        "retained_neighbours": graph_support.get("retained_neighbours"),
        "qualified_neighbours": graph_support.get("qualified_neighbours"),
        "neighbours_used": graph_support.get("neighbours_used"),
        "fallback_direct_only": bool(graph_support.get("fallback_direct_only")) or not available,
    }


def _neighbours(raw_neighbours: list, display_map: dict, limit: int) -> list:
    shown = []
    for neighbour in raw_neighbours[:limit]:
        pid = normalise_property_id(neighbour.get("neighbour_Property_ID"))
        display = display_map.get(pid) or {}
        entry = {
            "id": pid,
            "name": display.get("name"),
            "district": display.get("district"),
            "price": display.get("price"),
            "similarity": _r(neighbour.get("edge_similarity"), 4),
            "neighbour_direct_score": _r(neighbour.get("neighbour_direct_score"), 4),
            "support_share": _r(neighbour.get("neighbour_support_share"), 4),
            "location_resolution": neighbour.get("location_resolution"),
            "distance_km": _r(neighbour.get("distance_km"), 2),
            "similarity_components": {
                str(k)[4:]: _r(v, 4) for k, v in neighbour.items()
                if str(k).startswith("sim_") and _r(v, 4) is not None
            },
        }
        entry["plain"] = _plain_comparable_text(entry["name"] or entry["id"], entry)
        shown.append(entry)
    return shown


def _market_status(market: dict) -> dict:
    status = market.get("status") or "unavailable_not_computed"
    asking = _r(market.get("asking_ppsf"), 2)
    reference = _r(market.get("reference_mean"), 2)
    position = market.get("position_vs_reference")
    # A status alone is not evidence: the numbers must be retained too.
    available = status == "available" and None not in (asking, reference, position)
    reason = market.get("unavailable_reason") or MARKET_STATUS_REASONS.get(status, _human(status))
    if status == "available" and not available:
        reason = "the reference-group statistics were not retained for this listing"
    return {
        "enabled": bool(market.get("enabled")),
        "status": status,
        "available": available,
        "reason": reason,
        "asking_ppsf": asking,
        "reference_mean": reference,
        "position_vs_reference": position,
    }


def _location_status(neighbours: list) -> dict:
    resolutions = sorted({n["location_resolution"] for n in neighbours if n.get("location_resolution")})
    if not neighbours:
        return {
            "status": "not_applicable_no_neighbour_evidence",
            "label": "No comparable-neighbour location evidence",
            "detail": ("Graph support is unavailable for this candidate, so no neighbour "
                       "location similarity was used."),
            "resolutions": resolutions,
        }
    if set(resolutions) <= {"coordinates"}:
        return {
            "status": "coordinate_level",
            "label": "Coordinate-level location evidence",
            "detail": ("Neighbour location similarity uses listing coordinates; any distance is "
                       f"approximate and rounded to {DISTANCE_PRECISION_DP} decimal place."),
            "resolutions": resolutions,
        }
    return {
        "status": "district_level_fallback",
        "label": "District-level (administrative area) location evidence",
        "detail": ("At least one comparable relationship was built from administrative-area "
                   "matching instead of coordinates, so no precise distance is claimed."),
        "resolutions": resolutions,
    }


def _unavailable_evidence(evidence: dict, profile: dict, graph: dict, market: dict,
                          raw_count: int, max_neighbours: int) -> list:
    rows = []
    if not graph["available"]:
        rows.append({
            "code": f"graph_support_{graph['status']}",
            "label": "Comparable-neighbour support unavailable",
            "detail": GRAPH_STATUS_REASONS.get(graph["status"], _human(graph["status"])),
        })
    if not market["available"]:
        rows.append({
            "code": "market_context_unavailable",
            "label": "Market-context evidence unavailable",
            "detail": market["reason"],
        })
    inactive = []
    for match in evidence.get("direct_preference_matches") or []:
        if match.get("match") is not None:
            continue
        preference = match.get("preference")
        enforced = HARD_CONSTRAINT_FOR_PREFERENCE.get(preference)
        requirement = profile.get(enforced) if enforced else None
        inactive.append({
            "preference": preference,
            "label": DIRECT_LABELS.get(preference, preference),
            "reason": (f"enforced as a hard constraint instead ({_requirement(requirement)})"
                       if requirement not in (None, False, "", 0)
                       else "not stated in this search profile"),
        })
    if inactive:
        rows.append({
            "code": "inactive_preferences",
            "label": f"{len(inactive)} preference(s) not scored",
            "detail": "; ".join(f"{i['label']}: {i['reason']}" for i in inactive),
            "preferences": inactive,
        })
    used = graph["neighbours_used"] or 0
    if graph["available"] and used < SPARSE_NEIGHBOUR_THRESHOLD:
        rows.append({
            "code": "sparse_comparable_neighbours",
            "label": "Sparse comparable evidence",
            "detail": f"Only {used} qualified neighbour(s) support this candidate's graph component.",
        })
    if raw_count > max_neighbours:
        rows.append({
            "code": "neighbour_display_truncated",
            "label": "Comparable list truncated for display",
            "detail": (f"{max_neighbours} of {raw_count} retained neighbours are shown; "
                       "the score uses all of them."),
        })
    return rows


def _fallback_used(graph: dict, location: dict, unavailable: list) -> dict:
    items = []
    if not graph["available"] or graph["fallback_direct_only"]:
        items.append({
            "code": "direct_only_fallback",
            "label": "Direct-only fallback used",
            "detail": ("Graph support was unavailable, so the final score was renormalised to "
                       "the direct component."),
        })
    if location["status"] == "district_level_fallback":
        items.append({
            "code": "location_district_fallback",
            "label": "District-level location fallback used",
            "detail": location["detail"],
        })
    if any(r["code"] == "sparse_comparable_neighbours" for r in unavailable):
        items.append({
            "code": "sparse_neighbour_support",
            "label": "Sparse neighbour support",
            "detail": "Similarity support rests on very few qualified neighbours.",
        })
    return {"used": bool(items), "codes": [i["code"] for i in items], "items": items}


def _required_disclosures(graph: dict, market: dict, location: dict, fallback: dict) -> list:
    required = []
    if not graph["available"]:
        required.append("graph_support_unavailable")
    if not market["available"]:
        required.append("market_context_unavailable")
    if location["status"] == "district_level_fallback":
        required.append("location_precision_limited")
    if "direct_only_fallback" in fallback["codes"]:
        required.append("direct_only_fallback")
    if "sparse_neighbour_support" in fallback["codes"]:
        required.append("sparse_neighbour_support")
    return required


def _direct_reason_text(preference: str, component: dict) -> str:
    evidence = component.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    match, label = component["match"], component["label"]

    if preference == "price_value_fit" and evidence.get("value") is not None:
        budget = evidence.get("reference_budget")
        text = (f"Asking price RM {float(evidence['value']):,.0f} is about {_pct(1 - match)} "
                "of the stated budget"
                + (f" of RM {float(budget):,.0f}" if budget else "")
                + f" (price-value match {match:.2f}).")
    elif preference == "size_fit" and evidence.get("value") is not None and evidence.get("target"):
        text = (f"Built-up area {float(evidence['value']):,.0f} sq ft against a preferred "
                f"{float(evidence['target']):,.0f} sq ft (size match {match:.2f}).")
    elif preference == "facility_fit":
        checked = str(evidence).count("value=") if isinstance(evidence, str) else 0
        text = (f"{_pct(match)} of the selected nearby facility types are present near the "
                f"listing (facility match {match:.2f}"
                + (f"; {checked} facility column(s) checked" if checked else "") + ").")
    elif evidence.get("value") is not None:
        target = evidence.get("target")
        text = (f"{label}: {evidence.get('value')}"
                + (f" against a requested {target}" if target is not None else "")
                + f" (match {match:.2f}).")
    else:
        text = f"{label} match {match:.2f}."

    if component.get("weight_share"):
        text += f" This preference carries {_pct(component['weight_share'])} of the direct weight."
    return text


def _money(value) -> str | None:
    try:
        return f"RM {float(value):,.0f}"
    except (TypeError, ValueError):
        return None


def _area(value) -> str | None:
    try:
        return f"{float(value):,.0f} sq ft"
    except (TypeError, ValueError):
        return None


def _facility_counts(evidence) -> tuple[int, int]:
    """(facility types asked for, facility types present) parsed from the stored evidence."""
    if not isinstance(evidence, str):
        return (0, 0)
    asked = present = 0
    for part in evidence.split(";"):
        if "value=" not in part:
            continue
        asked += 1
        try:
            if float(part.split("value=")[-1]) > 0:
                present += 1
        except ValueError:
            pass
    return (asked, present)


def _comparison(preference: str, evidence) -> dict | None:
    """What the user asked for next to what this listing offers (for display)."""
    if preference == "facility_fit":
        asked, present = _facility_counts(evidence)
        if not asked:
            return None
        return {"asked": f"{asked} nearby facility type{'s' if asked != 1 else ''}",
                "listing": (f"all {present} present nearby" if present == asked
                            else f"{present} of {asked} present nearby")}
    if not isinstance(evidence, dict):
        return None
    value, target = evidence.get("value"), evidence.get("target")
    if preference == "price_value_fit":
        budget = evidence.get("reference_budget")
        asked, listing = _money(budget) if budget else None, _money(value)
        if asked and listing:
            return {"asked": f"under {asked}", "listing": f"asking {listing}"}
        return None
    if preference == "size_fit" and target and value:
        asked, listing = _area(target), _area(value)
        if asked and listing:
            return {"asked": asked, "listing": listing}
    if value is not None and target is not None:
        return {"asked": str(target), "listing": str(value)}
    return None


def _plain_reason_text(preference: str, component: dict) -> str:
    """Same evidence as _direct_reason_text, phrased for a homebuyer."""
    raw = component.get("evidence")
    evidence = raw if isinstance(raw, dict) else {}
    label = component["label"]

    if preference == "price_value_fit" and evidence.get("value") is not None:
        budget = evidence.get("reference_budget")
        text = f"This listing asks RM {float(evidence['value']):,.0f}"
        if budget:
            text += (f", which is about {_pct(1 - component['match'])} of your "
                     f"RM {float(budget):,.0f} budget")
        return text + "."
    if preference == "size_fit" and evidence.get("value") is not None and evidence.get("target"):
        return (f"You asked for about {float(evidence['target']):,.0f} sq ft; this listing has "
                f"{float(evidence['value']):,.0f} sq ft.")
    if preference == "facility_fit":
        asked, present = _facility_counts(raw)
        if asked:
            noun = "type" if asked == 1 else "types"
            if present == asked:
                return f"You picked {asked} nearby facility {noun}; all of them are close to this listing."
            return f"You picked {asked} nearby facility {noun}; {present} of them are close to this listing."
        return f"{_pct(component['match'])} of the nearby facility types you picked are close to this listing."
    if preference == "location_fit":
        return f"It is located where you asked ({label.lower()})."
    if evidence.get("value") is not None and evidence.get("target") is not None:
        return (f"You asked for {evidence.get('target')}; this listing has "
                f"{evidence.get('value')}.")
    return f"Its {label.lower()} is a good match for your search."


def _plain_comparable_text(name, neighbour: dict) -> str:
    """Why two listings count as comparable, without similarity jargon."""
    comps = neighbour.get("similarity_components") or {}
    bits = []
    ptype = comps.get("property_type")
    if ptype is not None:
        bits.append("same property type" if ptype >= 0.99 else "different property type")
    price = comps.get("price")
    if price is not None:
        bits.append(f"price within {max(1, round((1 - price) * 100))}%")
    size = comps.get("size")
    if size is not None:
        bits.append(f"size within {max(1, round((1 - size) * 100))}%")
    bedroom = comps.get("bedroom")
    if bedroom is not None:
        bits.append("same number of bedrooms" if bedroom >= 0.99 else "a different bedroom count")
    facility = comps.get("facility")
    if facility is not None and facility >= 0.8:
        bits.append("similar nearby amenities")
    distance = neighbour.get("distance_km")
    if distance is not None:
        bits.append(f"about {round(float(distance), DISTANCE_PRECISION_DP)} km away")
    elif neighbour.get("location_resolution") == "district":
        bits.append("in the same district")

    similarity = neighbour.get("similarity")
    lead = (f"{name} is {round(float(similarity) * 100)}% alike" if similarity is not None
            else f"{name} is a similar listing")
    return lead + (": " + ", ".join(bits) + "." if bits else ".")


def _summary_facts(neighbours: list) -> dict:
    """Aggregates over the retained comparable set, computed once for claim + check."""
    if not neighbours:
        return {}
    components = [n.get("similarity_components") or {} for n in neighbours]
    similarities = [n["similarity"] for n in neighbours if n.get("similarity") is not None]

    def gap(key: str):
        values = [c[key] for c in components if c.get(key) is not None]
        return round((1 - max(values)) * 100) if values else None

    types = [c["property_type"] for c in components if c.get("property_type") is not None]
    bedrooms = [c["bedroom"] for c in components if c.get("bedroom") is not None]
    return {
        "count": len(neighbours),
        "min_similarity": min(similarities) if similarities else None,
        "max_similarity": max(similarities) if similarities else None,
        "same_type_count": sum(1 for v in types if v >= SAME_VALUE),
        "type_count": len(types),
        "max_price_gap": gap("price"),
        "max_size_gap": gap("size"),
        "bedroom_diff_count": sum(1 for v in bedrooms if v < SAME_VALUE),
        "bedroom_count": len(bedrooms),
    }


def _join_clauses(clauses: list) -> str:
    if not clauses:
        return ""
    return clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]


def _summary_claim(neighbours: list, location: dict) -> dict:
    """One conclusion over the whole supporting set instead of per-listing repeats."""
    facts = _summary_facts(neighbours)
    clauses, technical = [], [f"{facts['count']} qualified comparable(s)"]

    if facts["type_count"]:
        same, total = facts["same_type_count"], facts["type_count"]
        clauses.append("are all the same property type" if same == total
                       else f"share the property type in {same} of {total}")
        technical.append(f"same property type in {same}/{total}")
    if facts["max_price_gap"] is not None:
        gap = facts["max_price_gap"]
        clauses.append("all match this asking price" if gap == 0
                       else f"sit within {gap}% of this asking price")
        technical.append(f"largest price gap {gap}%")
    if facts["max_size_gap"] is not None:
        gap = facts["max_size_gap"]
        clauses.append("share the same floor size" if gap == 0
                       else f"stay within {gap}% of its floor size")
        technical.append(f"largest size gap {gap}%")
    if facts["bedroom_count"]:
        diff, total = facts["bedroom_diff_count"], facts["bedroom_count"]
        clauses.append("share the same bedroom count" if diff == 0 else
                       "differ on bedroom count in every case" if diff == total else
                       f"differ on bedroom count in {diff} of {total}")
        technical.append(f"bedroom count differs in {diff}/{total}")

    claim = {
        "kind": "comparable_summary",
        "block": "graph",
        "summary": facts,
        "evidence_ref": ["comparable_neighbours_shown"],
    }

    distances = [n for n in neighbours if n.get("distance_km") is not None]
    if location["status"] == "coordinate_level" and distances:
        raw = [n["distance_km"] for n in distances]
        shown = [round(float(d), DISTANCE_PRECISION_DP) for d in raw]
        lo, hi = min(shown), max(shown)
        clauses.append(f"about {lo} km away" if lo == hi
                       else f"lie between about {lo} km and about {hi} km away")
        technical.append(f"distances {lo}-{hi} km rounded to {DISTANCE_PRECISION_DP} dp")
        claim["distance"] = {"raw": raw, "shown": shown}

    tail = ""
    if facts["min_similarity"] is not None:
        lo = round(facts["min_similarity"] * 100)
        hi = round(facts["max_similarity"] * 100)
        tail = (f"Their overall likeness is {lo}% alike" if lo == hi
                else f"Their overall likeness runs from {lo}% to {hi}% alike")
        tail += ", so the comparison base is consistent."
        technical.append(f"similarity {lo}%-{hi}%")

    claim["text"] = "Supporting-set summary: " + "; ".join(technical) + "."
    claim["plain"] = (f"Put together, the {facts['count']} similar listings "
                      + _join_clauses(clauses) + "." + (" " + tail if tail else ""))
    return claim


def _summary_problems(record: dict, claim: dict) -> list:
    neighbours = _shown_neighbours(record)
    if not neighbours:
        return ["summary claim without retained comparable neighbours"]
    expected = _summary_facts(neighbours)
    for key, value in (claim.get("summary") or {}).items():
        if key not in expected:
            return [f"summary fact cannot be derived from the record: {key}"]
        actual = expected[key]
        if isinstance(actual, float) or isinstance(value, float):
            if not _close(value, actual, tolerance=1e-6):
                return [f"summary fact {key} does not match the retained comparable set"]
        elif value != actual:
            return [f"summary fact {key} does not match the retained comparable set"]
    if claim.get("distance"):
        return _distance_problems(record, claim)
    return []


def _market_problems(record: dict, claim: dict) -> list:
    status = record.get("market_context_status") or {}
    if not status.get("available"):
        return ["market claim without retained market-context evidence"]
    claimed = (claim.get("market_check") or {}).get("contribution")
    if claimed is None:
        return []
    stored = (record.get("score_trace") or {}).get("contributions", {}).get("market")
    if stored is None:
        return ["market contribution claimed but not retained in the score trace"]
    if not _close(claimed, stored, tolerance=1e-4):
        return ["market contribution does not match the score trace"]
    if "market" not in (record.get("active_score_components") or []):
        return ["market claimed as a scored component but is not an active component"]
    return []


def _plural(count, singular, plural=None) -> str:
    try:
        n = int(float(count))
    except (TypeError, ValueError):
        return f"{count} {singular}"
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _gate_phrases(name: str, requirement, value) -> tuple[str, str]:
    """('what you asked for', 'a sentence about what this listing has')."""
    has_value = value is not None and value != ""

    if name == "max_budget_rm":
        asked = _money(requirement) or "your budget"
        return (f"a price under {asked}",
                f"it asks {_money(value)}" if has_value else "this value is not recorded")
    if name == "min_bedrooms":
        try:
            asked = f"{float(requirement):g} or more bedrooms"
        except (TypeError, ValueError):
            asked = "a bedroom minimum"
        return (asked, f"it has {_plural(value, 'bedroom')}" if has_value else "this value is not recorded")
    if name == "min_bathrooms":
        try:
            asked = f"{float(requirement):g} or more bathrooms"
        except (TypeError, ValueError):
            asked = "a bathroom minimum"
        return (asked, f"it has {_plural(value, 'bathroom')}" if has_value else "this value is not recorded")
    if name == "parking_required":
        return ("at least one parking space",
                f"it has {_plural(value, 'parking space')}" if has_value else "this value is not recorded")
    if name == "furnished_only":
        return ("a furnished home", f"it is {value}" if has_value else "this value is not recorded")
    if name == "preferred_area (district)":
        district = str(requirement).title() if requirement else "your district"
        return (f"the {district} district", f"it is in {value}" if has_value else "this value is not recorded")
    if name == "property_type":
        return (f"a {requirement}", f"it is a {value}" if has_value else "this value is not recorded")
    if name == "tenure":
        return (f"{requirement} tenure", f"it is {value}" if has_value else "this value is not recorded")
    return (str(requirement), f"it has {value}" if has_value else "this value is not recorded")


def _claims(record: dict, profile: dict) -> list:
    """Claim list for one record. Every claim cites record references (Stage 9 input)."""
    claims = []
    components = record["direct_score_components"]
    graph = record["graph_score_component"]
    weights = record["score_trace"]["weights"]
    contributions = record["score_trace"]["contributions"]
    neighbours = record["comparable_neighbours_shown"]
    market = record["market_context_status"]
    market_component = record.get("market_score_component") or {}
    market_scored = bool(market.get("available") and market_component.get("contribution") is not None)
    location = record["location_resolution_status"]
    fallback = record["fallback_used"]
    score = record["final_score"]

    claims.append({
        "kind": "score",
        "block": "score",
        "text": (f"Ranked #{record['rank']} with a CPG final score of {score:.3f}."
                 if score is not None and record["rank"] is not None else
                 "No rank or final score was retained for this candidate."),
        "plain": (f"It is number {record['rank']} in your results."
                  if record["rank"] is not None else
                  "This listing has no rank in your results."),
        "evidence_ref": ["rank", "final_score"],
    })

    if graph["available"] and None not in (weights["direct"], weights["graph"],
                                           contributions["direct"], contributions["graph"]):
        market_clause = ""
        market_plain = ""
        market_refs = []
        if market_scored:
            market_clause = (f" and market context (M = {market_component['value']:.3f}, weight "
                             f"{market_component['normalised_weight']:.2f}, contributing "
                             f"{market_component['contribution']:.3f})")
            market_plain = " and how its asking price compares with the area"
            market_refs = ["score_trace.weights.market", "market_score_component.value",
                           "score_trace.contributions.market"]
        claims.append({
            "kind": "score",
            "block": "graph",
            "text": (f"The final score combines direct preference matching (weight "
                     f"{weights['direct']:.2f}, contributing {contributions['direct']:.3f}) with "
                     f"comparable-neighbour support (G = {graph['value']:.3f}, weight "
                     f"{weights['graph']:.2f}, contributing {contributions['graph']:.3f})"
                     f"{market_clause}."),
            "plain": ("The score blends how well it fits your preferences with how well "
                      "similar nearby listings fit them" + market_plain + "."),
            "evidence_ref": ["score_trace.weights.direct", "score_trace.contributions.direct",
                             "graph_score_component.value", "score_trace.weights.graph",
                             "score_trace.contributions.graph"] + market_refs,
        })
        ranked_components = {name: value for name, value in contributions.items()
                             if name in (record.get("active_score_components") or [])
                             and value is not None}
        dominant = max(ranked_components, key=ranked_components.get) if ranked_components else None
        rest = [n for n in ranked_components if n != dominant]
        if (dominant and len(ranked_components) > 1
                and ranked_components[dominant] - max(ranked_components[n] for n in rest) > 0.01):
            label = {"direct": "Direct preference matching", "graph": "Comparable-neighbour support",
                     "market": "Market context"}.get(dominant, dominant)
            others = {"direct": "your stated preferences", "graph": "the similar listings",
                      "market": "the market comparison"}
            plain = {"direct": "How well it fits your own preferences counts for more here than ",
                     "graph": "The similar listings count for more here than ",
                     "market": "The area price comparison counts for more here than "}[dominant] + \
                " or ".join(others[n] for n in rest) + " do."
            claims.append({
                "kind": "ranking",
                "block": dominant if dominant != "market" else "market",
                "text": (f"{label} contributes the larger share of the final score "
                         f"({ranked_components[dominant]:.3f} vs "
                         + " vs ".join(f"{ranked_components[n]:.3f}" for n in rest) + ")."),
                "plain": plain,
                "evidence_ref": [f"score_trace.contributions.{n}" for n in ranked_components],
                "ranking_check": {"type": "dominant_component", "component": dominant},
            })
    else:
        details = []
        if weights["direct"] is not None:
            details.append(f"weight {weights['direct']:.2f}")
        if contributions["direct"] is not None:
            details.append(f"contributing {contributions['direct']:.3f}")
        claims.append({
            "kind": "score",
            "block": "direct",
            "text": ("The final score uses the direct component only"
                     + (f" ({', '.join(details)})" if details else "") + "."),
            "plain": ("Your stated preferences decide the score, because there were no similar "
                      "listings to compare it with."),
            "evidence_ref": ["score_trace.weights.direct", "score_trace.contributions.direct"],
        })

    constraints = record["hard_constraint_evidence"]
    # A gate is pass/fail: it decides eligibility, not score. Each satisfied gate is
    # still explained explicitly (what was asked vs what the listing has), because a
    # homebuyer needs to see that their requirement was actually checked.
    for position, gate in enumerate(constraints):
        name = str(gate.get("constraint"))
        label = HARD_CONSTRAINT_LABELS.get(name, _human(name))
        asked, listing = _gate_phrases(name, gate.get("requirement"), gate.get("property_value"))
        claims.append({
            "kind": "constraint",
            "constraint": name,
            "block": "constraints",
            "text": (f"Hard constraint satisfied: {label} — required "
                     f"{gate.get('requirement')}, listing value {gate.get('property_value')}."),
            "plain": f"You asked for {asked}; {listing}.",
            "evidence_ref": [f"hard_constraint_evidence[{position}]"],
        })

    for preference, component in components.items():
        claims.append({
            "kind": "direct_reason",
            "component": preference,
            "label": component["label"],
            "match": component["match"],
            "block": "direct",
            "text": _direct_reason_text(preference, component),
            "plain": component.get("plain") or _plain_reason_text(preference, component),
            "comparison": component.get("comparison"),
            "evidence_ref": [f"direct_score_components.{preference}.match",
                             f"direct_score_components.{preference}.evidence"],
        })

    if graph["available"]:
        claims.append({
            "kind": "graph_reason",
            "component": "graph",
            "block": "graph",
            "text": (f"Comparable-neighbour support G = {graph['value']:.3f} is aggregated from "
                     f"{graph['neighbours_used']} qualified neighbour(s) that also satisfy the "
                     f"current hard constraints ({graph['retained_neighbours']} retained graph "
                     "edge(s))."),
            "plain": ((f"{graph['neighbours_used']} similar listings nearby also fit your filters, "
                       "and they support this ranking.") if (graph["neighbours_used"] or 0) != 1 else
                      "1 similar listing nearby also fits your filters, and it supports this ranking."),
            "evidence_ref": ["graph_score_component.value", "graph_score_component.neighbours_used",
                             "graph_score_component.retained_neighbours"],
        })
    else:
        reason = GRAPH_STATUS_REASONS.get(graph["status"], _human(graph["status"]))
        claims.append({
            "kind": "disclosure",
            "discloses": "graph_support_unavailable",
            "block": "graph",
            "text": (f"Graph support unavailable: {reason}. "
                     "No comparable-neighbour contribution is claimed."),
            "plain": ("We could not find similar listings that fit your filters, so your own "
                      "preferences alone decide this score."),
            "evidence_ref": ["graph_score_component.status"],
        })

    for position, neighbour in enumerate(neighbours):
        ref = f"comparable_neighbours_shown[{position}]"
        share = neighbour["support_share"]
        claims.append({
            "kind": "comparable",
            "neighbour_id": neighbour["id"],
            "support_share": share,
            "block": f"neighbour:{neighbour['id']}",
            "text": (f"Neighbour {neighbour['id']} ({neighbour['name']}) is a qualified comparable: "
                     f"similarity {_num(neighbour['similarity'])}, direct preference score "
                     f"{_num(neighbour['neighbour_direct_score'])}, "
                     + (f"providing {_pct(share)} of this candidate's neighbour support."
                        if share is not None else
                        "providing no measurable share of neighbour support.")),
            "plain": _plain_comparable_text(neighbour["name"], neighbour),
            "evidence_ref": [f"{ref}.id", f"{ref}.similarity",
                             f"{ref}.neighbour_direct_score", f"{ref}.support_share"],
        })

    if neighbours:
        claims.append(_summary_claim(neighbours, location))

    top = max(neighbours, key=lambda n: n["support_share"] or 0.0) if len(neighbours) > 1 else None
    if top is not None and top["support_share"] is not None:
        claims.append({
            "kind": "ranking",
            "block": f"neighbour:{top['id']}",
            "text": (f"Neighbour {top['id']} provides the largest share of this candidate's "
                     f"neighbour support ({_pct(top['support_share'])})."),
            "plain": f"{top['name']} is the closest match among these similar listings.",
            "evidence_ref": [f"comparable_neighbours_shown[{neighbours.index(top)}].support_share"],
            "ranking_check": {"type": "largest_support", "neighbour_id": top["id"]},
        })

    if market["available"]:
        position = MARKET_POSITION_PHRASES.get(
            market["position_vs_reference"], _human(market["position_vs_reference"]))
        scored_clause = ""
        scored_plain = ""
        market_refs = ["market_context_status.asking_ppsf",
                       "market_context_status.reference_mean",
                       "market_context_status.position_vs_reference"]
        contribution = market_component.get("contribution") if market_scored else None
        if market_scored:
            scored_clause = (f" It is scored: the market component M = "
                             f"{market_component['value']:.3f} carries weight "
                             f"{market_component['normalised_weight']:.2f} and contributes "
                             f"{contribution:.3f} of the final score.")
            scored_total = sum(v for v in contributions.values() if v)
            scored_share = round(contribution / scored_total * 100) if scored_total else 0
            scored_plain = (f" That comparison is part of the score: it accounts for "
                            f"{scored_share}% of the points this listing earned.")
            market_refs = market_refs + ["score_trace.weights.market",
                                         "score_trace.contributions.market"]
        claims.append({
            "kind": "market_context",
            "block": "market",
            "text": (f"Market context is available: the asking price per sq ft is {position} "
                     f"(reference-group mean RM {market['reference_mean']:,.0f} per sq ft; "
                     f"comparison only, not a valuation).{scored_clause}"),
            "plain": ("We can also compare its asking price with typical prices for its area. "
                      "This is context only, not a valuation." + scored_plain),
            "evidence_ref": market_refs,
            "market_check": {"contribution": contribution},
        })
    else:
        claims.append({
            "kind": "market",
            "discloses": "market_context_unavailable",
            "block": "market",
            "text": f"Market-context evidence unavailable: {market['reason']}.",
            "plain": "There is no local market price comparison for this listing.",
            "evidence_ref": ["market_context_status.status"],
        })

    if location["status"] == "coordinate_level":
        claims.append({
            "kind": "location",
            "block": "location",
            "text": (f"Location similarity for all {len(neighbours)} comparable(s) uses listing "
                     "coordinates."),
            "plain": "Distances come from the listings' map locations and are rounded.",
            "evidence_ref": ["location_resolution_status.resolutions"],
        })
    elif location["status"] == "district_level_fallback":
        claims.append({
            "kind": "location",
            "discloses": "location_precision_limited",
            "block": "location",
            "text": ("Location precision is limited: at least one comparable relationship uses "
                     "administrative-area matching, so no exact distance is claimed."),
            "plain": ("For some of these listings we could only match the whole district, not the "
                      "exact distance."),
            "evidence_ref": ["location_resolution_status.resolutions"],
        })
    else:
        claims.append({
            "kind": "location",
            "block": "location",
            "text": "No comparable-neighbour location evidence is available for this candidate.",
            "plain": "There are no location comparisons to show for this listing.",
            "evidence_ref": ["location_resolution_status.status"],
        })

    for item in fallback["items"]:
        if item["code"] == "direct_only_fallback":
            claims.append({
                "kind": "disclosure",
                "discloses": "direct_only_fallback",
                "block": "score",
                "text": ("Fallback used: graph support is unavailable, so the final score uses "
                         "the direct component only."),
                "plain": ("Because there were no similar listings to compare with, only your "
                          "preferences were used."),
                "evidence_ref": ["graph_score_component.fallback_direct_only",
                                 "score_trace.weights.direct"],
            })
        elif item["code"] == "sparse_neighbour_support":
            claims.append({
                "kind": "disclosure",
                "discloses": "sparse_neighbour_support",
                "block": "graph",
                "text": (f"Sparse comparable evidence: only {graph['neighbours_used']} qualified "
                         "neighbour(s) support this candidate."),
                "plain": (f"Only {graph['neighbours_used']} similar listing(s) support this result, "
                          "so treat the comparison as thin evidence."),
                "evidence_ref": ["graph_score_component.neighbours_used"],
            })

    claims.sort(key=lambda c: CLAIM_ORDER.get(c["kind"], 99))
    return claims


def _plain_summary(record: dict) -> str:
    """One-sentence, homebuyer-level summary of why this listing ranks where it does."""
    rank = record["rank"]
    graph = record["graph_score_component"]
    used = graph.get("neighbours_used") or 0
    if rank is None:
        opening = "This listing has no recorded rank"
    else:
        opening = f"This listing is number {rank} in your results"
    if not graph["available"]:
        return (f"{opening} because it fits your stated preferences. No similar listings "
                "were available to compare with.")
    if used == 1:
        support = "1 similar listing nearby supports that view"
    else:
        support = f"{used} similar listings nearby support that view"
    return f"{opening} because it fits your stated preferences, and {support}."


def _main_reason(record: dict) -> dict | None:
    """The single strongest scored reason (largest contribution to D)."""
    for preference, component in (record.get("direct_score_components") or {}).items():
        return {
            "preference": preference,
            "label": component["label"],
            "contribution": component["contribution"],
            "plain": component.get("plain"),
        }
    return None


def _generated_explanation(accepted: list, market_available: bool, record: dict) -> dict:
    sentences = [{"kind": c["kind"], "text": c["text"], "plain": c.get("plain"),
                  "comparison": c.get("comparison"), "label": c.get("label"),
                  "preference": c.get("component"), "match": c.get("match")}
                 for c in accepted]
    plain_sentences = [s for s in sentences if s["plain"]]
    limitations = [t for t in LIMITATIONS
                   if market_available or "Market context" not in t]
    return {
        "headline": next((c["text"] for c in accepted if c["kind"] == "score"), ""),
        "plain_summary": _plain_summary(record),
        "main_reason": _main_reason(record),
        "paragraph": " ".join(s["text"] for s in sentences),
        "plain_paragraph": " ".join(s["plain"] for s in plain_sentences),
        "sentences": sentences,
        "disclosures": [c["text"] for c in accepted if c.get("discloses")],
        "limitations": limitations,
    }


def build_explanation(evidence: dict, *, profile: dict, direct_weights: dict | None = None,
                      hard_constraints: list | None = None, display: dict | None = None,
                      neighbour_display: dict | None = None,
                      max_neighbours: int = DEFAULT_MAX_NEIGHBOURS) -> dict:
    """Build the Stage 8 explanation record for one ranked candidate.

    ``evidence`` comes from ``ranking.cpg_ranker.build_evidence``; it is the only
    fact source. ``display`` / ``neighbour_display`` add presentational labels that
    are already part of the same record (listing name, district, asking price).
    """
    evidence = evidence or {}
    graph_support = evidence.get("graph_support") or {}
    components = evidence.get("components") or {}
    raw_neighbours = graph_support.get("neighbours") or []

    neighbours = _neighbours(raw_neighbours, neighbour_display or {}, max_neighbours)
    graph = _graph_component(components.get("graph"), graph_support)
    market = _market_status(evidence.get("market_context") or {})
    market_component = _market_component(components.get("market"), market)
    location = _location_status(neighbours)
    active = list(evidence.get("active_score_components") or [])
    direct_components = _direct_components(evidence, direct_weights)
    unavailable = _unavailable_evidence(evidence, profile, graph, market,
                                        len(raw_neighbours), max_neighbours)
    fallback = _fallback_used(graph, location, unavailable)
    required = _required_disclosures(graph, market, location, fallback)
    direct_component = components.get("direct") or {}
    raw_id = evidence.get("Property_ID")

    record = {
        "engine_version": ENGINE_VERSION,
        "property_id": normalise_property_id(raw_id) if raw_id is not None else None,
        "rank": evidence.get("cpg_rank"),
        "final_score": _r(evidence.get("final_score"), 4),
        "score_status": evidence.get("score_status"),
        "score_trace": {
            "weights": {
                "direct": _r(direct_component.get("normalised_weight"), 4),
                "graph": _r(graph["normalised_weight"], 4),
                "market": market_component["normalised_weight"],
            },
            "contributions": {
                "direct": _r(direct_component.get("contribution"), 4),
                "graph": _r(graph["contribution"], 4),
                "market": market_component["contribution"],
            },
            "rule": ("R = w_direct*D + w_graph*G (+ w_market*M when market context is active), "
                     "weights renormalised over available components"),
        },
        "active_score_components": active,
        "omitted_score_components": list(evidence.get("omitted_score_components") or []),
        "direct_score_components": direct_components,
        "graph_score_component": graph,
        "market_score_component": market_component,
        "comparable_neighbours_shown": neighbours,
        "market_context_status": market,
        "location_resolution_status": location,
        "hard_constraint_evidence": _hard_constraint_evidence(hard_constraints),
        "unavailable_evidence": unavailable,
        "fallback_used": fallback,
        "required_disclosures": required,
        "config_ids": evidence.get("config_ids"),
        "display": display or {},
    }

    accepted, rejected = [], []
    for claim in _claims(record, profile):
        (rejected if validate_claim(record, claim) else accepted).append(claim)

    record["generated_explanation"] = _generated_explanation(accepted, market["available"], record)
    record["faithfulness"] = check_faithfulness(record, accepted, rejected, required)
    record["withheld_claims"] = [{"kind": c["kind"], "text": c["text"]} for c in rejected]
    return record


def summarise_explanations(records: list) -> dict:
    """Stage 9 aggregate indicators across every explained candidate."""
    records = [r for r in (records or []) if r]
    if not records:
        return {"engine_version": ENGINE_VERSION, "properties_explained": 0,
                "claims_displayed": 0, "claims_withheld": 0,
                "checks_passed": 0, "checks_total": 0, "fully_faithful": 0,
                "mean_metrics": {}, "disclosure_counts": {}}

    names = sorted(records[0]["faithfulness"]["metrics"])
    disclosure_counts = {}
    for record in records:
        for item in record["unavailable_evidence"]:
            disclosure_counts[item["code"]] = disclosure_counts.get(item["code"], 0) + 1

    return {
        "engine_version": ENGINE_VERSION,
        "properties_explained": len(records),
        "claims_displayed": sum(r["faithfulness"]["claims_displayed"] for r in records),
        "claims_withheld": sum(r["faithfulness"]["claims_withheld"] for r in records),
        "checks_passed": sum(r["faithfulness"]["checks_passed"] for r in records),
        "checks_total": sum(r["faithfulness"]["checks_total"] for r in records),
        "fully_faithful": sum(1 for r in records if r["faithfulness"]["passed"]),
        "mean_metrics": {
            name: round(sum(r["faithfulness"]["metrics"][name] for r in records) / len(records), 4)
            for name in names
        },
        "disclosure_counts": dict(sorted(disclosure_counts.items())),
    }


# ------------------------------------------------------------------------ self-test
def _synth_evidence(*, graph=True, neighbours=3, location="coordinates",
                    market_status="disabled_by_variant") -> dict:
    nbs = [{
        "neighbour_Property_ID": str(100 + i),
        "edge_similarity": 0.9 - 0.05 * i,
        "neighbour_direct_score": 0.7 - 0.05 * i,
        "neighbour_support_share": 0.5 - 0.1 * i,
        "location_resolution": location,
        "distance_km": 0.12 + i,
        "sim_location": 0.9, "sim_price": 1.0, "sim_size": 0.8,
    } for i in range(neighbours)]
    market_on = market_status == "available"
    components = {
        "direct": {"value": 0.7868, "lambda": 0.6, "normalised_weight": 0.5 if market_on else 0.6,
                   "contribution": 0.3934 if market_on else 0.4721},
        "graph": ({"value": 0.7517, "lambda": 0.4, "normalised_weight": 0.3333 if market_on else 0.4,
                   "contribution": 0.2506 if market_on else 0.3007} if graph else {}),
    }
    if market_on:
        components["market"] = {"value": 0.344042, "lambda": 0.2, "normalised_weight": 0.1667,
                                "contribution": 0.0573}
    return {
        "Property_ID": "595", "cpg_rank": 1, "final_score": 0.7013 if market_on else 0.7727,
        "score_status": "scored",
        "components": components,
        "active_score_components": (["direct", "graph", "market"] if market_on
                                    else (["direct", "graph"] if graph else ["direct"])),
        "omitted_score_components": [] if graph else ["graph"],
        "direct_preference_matches": [
            {"preference": "size_fit", "match": 0.95,
             "evidence": {"column": "built_up_area", "value": "980.0", "target": 1000.0}},
            {"preference": "price_value_fit", "match": 0.6235,
             "evidence": {"column": "price", "value": 320000.0, "reference_budget": 850000.0}},
            {"preference": "location_fit", "match": None, "evidence": "inactive"},
        ],
        "graph_support": {
            "status": "available" if graph else "unavailable_no_retained_neighbours",
            "retained_neighbours": 7, "qualified_neighbours": neighbours,
            "neighbours_used": neighbours if graph else 0,
            "fallback_direct_only": not graph,
            "neighbours": nbs if graph else [],
        },
        "market_context": ({"enabled": True, "status": "available",
                            "asking_ppsf": 397.96, "reference_mean": 504.52,
                            "position_vs_reference": "within_reference_iqr"}
                           if market_on else
                           {"enabled": False, "status": market_status,
                            "asking_ppsf": None, "reference_mean": None,
                            "position_vs_reference": None}),
        "config_ids": {"ranking": "RANK-test", "graph": "SIM-test",
                       "market_context": "MC-test" if market_on else None},
    }


def _self_test() -> int:
    profile = {"preferred_area": "petaling", "max_budget_rm": 850000,
               "property_type": "Condominium", "min_bedrooms": 3, "min_bathrooms": 2,
               "tenure": None, "furnished_only": False}
    weights = {"price_value_fit": 0.25, "size_fit": 0.25, "facility_fit": 0.15}
    audit = [{"constraint": "max_budget_rm", "value": 850000, "candidates_after": 96},
             {"constraint": "preferred_area (district)", "value": "petaling",
              "candidates_after": 96}]

    failures = 0
    for label, kwargs in [
        ("full graph evidence", {}),
        ("graph unavailable", {"graph": False}),
        ("district-level location", {"location": "district"}),
        ("sparse neighbours", {"neighbours": 1}),
        ("market context available", {"market_status": "available"}),
    ]:
        record = build_explanation(
            _synth_evidence(**kwargs), profile=profile, direct_weights=weights,
            hard_constraints=audit,
            display={"name": "Prima U1 Condominium", "district": "Petaling", "price": 320000.0},
            neighbour_display={str(100 + i): {"name": f"Comparable {i}", "district": "Petaling",
                                              "price": 300000.0 + i}
                               for i in range(kwargs.get("neighbours", 3))},
        )
        faith = record["faithfulness"]
        ok = faith["passed"] and not record["withheld_claims"]
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: "
              f"{faith['checks_passed']}/{faith['checks_total']} checks, "
              f"{faith['claims_displayed']} claims, {faith['claims_withheld']} withheld")
        for check in faith["checks"]:
            if not check["passed"]:
                print(f"         - {check['label']}: {check['detail']}")
        for claim in record["withheld_claims"]:
            print(f"         - withheld {claim['kind']}: {claim['text'][:110]}")

    # Negative control: a fabricated claim must be rejected, not displayed.
    record = build_explanation(_synth_evidence(), profile=profile, direct_weights=weights)
    fake = {"kind": "comparable", "neighbour_id": "999999", "block": "neighbour:999999",
            "text": "Neighbour 999999 is a qualified comparable.",
            "evidence_ref": ["comparable_neighbours_shown[0].id"]}
    fake_distance = {"kind": "distance", "block": "location",
                     "text": "Exactly 0.1234 km away.",
                     "evidence_ref": ["comparable_neighbours_shown[0].distance_km"],
                     "distance": {"raw": [0.12], "shown": [0.1234]}}
    fake_summary = {"kind": "comparable_summary", "block": "graph",
                    "text": "Supporting-set summary: 99 comparables, all identical.",
                    "evidence_ref": ["comparable_neighbours_shown"],
                    "summary": {"count": 99}}
    caught = (bool(validate_claim(record, fake)) and bool(validate_claim(record, fake_distance))
              and bool(validate_claim(record, fake_summary)))
    failures += 0 if caught else 1
    print(f"[{'PASS' if caught else 'FAIL'}] negative control: fabricated comparable, "
          "over-precise distance and invented comparable summary are rejected")

    summary = summarise_explanations([record])
    print(f"summary: {summary['properties_explained']} record(s), "
          f"unsupported-claim rate {summary['mean_metrics']['unsupported_explanation_claim_rate']}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(_self_test())
