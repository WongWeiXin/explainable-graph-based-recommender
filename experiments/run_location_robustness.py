"""Location robustness (Section 3.14) - Stage 11, reduced location precision.

Run from the project root:
    python experiments/run_location_robustness.py
    python experiments/run_location_robustness.py --conditions coords_exact,district_only

Design
    Same v2 queries, same eligible sets, same direct scores, same similarity and graph
    parameters. Only the location evidence changes, so any difference is attributable to
    location precision:

        coords_exact        full coordinates (frozen reference configuration)
        coords_rounded_3dp  coordinates rounded to 3 dp (~110 m grid)
        coords_rounded_2dp  coordinates rounded to 2 dp (~1.1 km grid)
        district_only       coordinates withheld, administrative-area matching only
                            (same district -> delta 0.5, otherwise 0)
        location_none       location component unavailable (weights renormalised)

    The property set is identical in every condition; nothing is dropped or re-filtered,
    only the location similarity used to retain graph edges changes.

Pre-registered analysis plan (fixed in this file before any outcome is inspected)
    primary metric      NDCG@5 under the REL-v1 rubric. The rubric reads only query attributes
                        and listing attributes, never a model score, so it grades 100% of the
                        retrieved pairs; coverage of the frozen pooled label file is reported
                        separately instead of being folded into the effect size.
    secondary           P@5, Recall@5, MRR (rubric)
    population          ranking-role queries (>= 5 eligible properties). The per-query file keeps
                        `partition`, so a validation+test-only reanalysis is reproducible.
    test                two-sided Wilcoxon signed-rank, each degraded condition vs the reference,
                        paired on query, zero_method="wilcox"
    multiplicity        Holm adjustment across the non-reference conditions (family = 4)
    effect size         matched-pairs rank-biserial correlation
    uncertainty         percentile bootstrap, 10,000 resamples, seed 20261031, for the mean paired
                        difference and for each condition's own NDCG@5 level
    decision rule       report a difference only when p_holm < 0.05
    frozen              recorded in the JSON `analysis_plan` block

Reported per condition
    query funnel       total / empty-result / sparse-candidate / ranking, so that 135 out of 250 is
                       explained rather than implied
    ranking change     Top-5 and Top-10 overlap with the reference, identical-order %, changed
                       set %, mean |rank shift|
    relevance          rubric NDCG@5 / P@5 / Recall@5 / MRR; frozen-file metrics only on the
                       fully-labelled subset, with coverage stated
    statistics         paired mean difference, bootstrap CI, Wilcoxon p, Holm p, effect size
    graph density      edges, GRAPH-level edge overlap, density, isolated nodes, components
    neighbours         graph-support availability AND query-qualified neighbour overlap: per
                       candidate Jaccard / identical-set %, neighbours lost and gained, plus the
                       per-query (candidate, neighbour) pool overlap
    explanation        coordinate / district / unavailable evidence %, limited-location disclosure
                       required vs actually emitted, fallback frequency

Outputs
    outputs/robustness/location_robustness.csv                 one row per condition
    outputs/robustness/location_robustness_per_query.csv       paired per-query metrics + partition
    outputs/robustness/location_robustness_labelling_pool.csv  retrieved pairs the frozen label
                                                              file does not cover (blind, ungraded)
    outputs/robustness/location_robustness.json                configs, analysis plan, statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "experiments", ROOT / "baseline_models"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common as X  # noqa: E402  (shared experiment helpers)
import recommender_service as svc  # noqa: E402
from rule_based_baseline import (LABEL_COLUMNS, _context, _int, _listing_name,  # noqa: E402
                                 grade_pair)
from cpg_ranker import RankingConfig, build_evidence, rank_candidates  # noqa: E402
from explanation_engine import build_explanation  # noqa: E402

OUT_DIR = ROOT / "outputs" / "robustness"
LABELS = ROOT / "frozen_queries" / "relevance_labels_v2.csv"
FROZEN_TOP5 = ROOT / "outputs" / "top5" / "cpg_top5_recommendations.csv"
REFERENCE = "coords_exact"
RELEVANT_GRADE = 2          # grade >= 2 counts as relevant for P@5 / Recall@5 / MRR
LOCATION_DISCLOSURE = "Location precision is limited"
EVAL_SPLITS = {"validation", "test"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261031

ANALYSIS_PLAN = {
    "frozen_before_running": True,
    "primary_metric": "ndcg5_rubric",
    "secondary_metrics": ["p5_rubric", "recall5_rubric", "mrr_rubric"],
    "population": "benchmark_role == 'ranking' (>= 5 eligible properties); the per-query file "
                  "keeps `partition` for split-restricted reanalysis",
    "comparison": "each degraded condition vs coords_exact, paired on query_id",
    "test": "two-sided Wilcoxon signed-rank, zero_method='wilcox'",
    "alpha": 0.05,
    "multiplicity": "Holm across the 4 non-reference conditions",
    "effect_size": "matched-pairs rank-biserial correlation",
    "uncertainty": f"percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}",
    "decision_rule": "report a difference only when p_holm < 0.05",
    "label_grader": "REL-v1 rubric (query + listing attributes only, no model score) -> 100% of "
                    "retrieved pairs graded; the frozen pooled label file is a cross-check",
}

CONDITIONS = [
    {"name": "coords_exact", "location_mode": "coordinates", "round": None,
     "label": "full coordinates (reference)"},
    {"name": "coords_rounded_3dp", "location_mode": "coordinates", "round": 3,
     "label": "coordinates rounded to 3 dp (~110 m)"},
    {"name": "coords_rounded_2dp", "location_mode": "coordinates", "round": 2,
     "label": "coordinates rounded to 2 dp (~1.1 km)"},
    {"name": "district_only", "location_mode": "district_only", "round": None,
     "label": "area-level location (district fallback only)"},
    {"name": "location_none", "location_mode": "none", "round": None,
     "label": "location unavailable"},
]


# --------------------------------------------------------------------------- metric helpers
def ndcg_at_k(grades: list[int], k: int = 5) -> float:
    gains = [(2 ** g) - 1 for g in grades[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def graph_metrics(cpg, edges: pd.DataFrame, n_nodes: int,
                  reference_edges: set | None) -> tuple[dict, set]:
    """Graph-level structure. Edge overlap here is a GRAPH metric, not a neighbour metric."""
    n_edges = int(len(edges))
    edge_set = ({tuple(sorted((int(a), int(b)))) for a, b in
                 zip(edges["source_Property_ID"], edges["target_Property_ID"])}
                if n_edges else set())
    degree = pd.concat([edges["source_Property_ID"], edges["target_Property_ID"]]).value_counts() \
        if n_edges else pd.Series(dtype=int)
    isolated = int(n_nodes - len(degree))
    multi = [c for c in nx.connected_components(cpg.graph) if len(c) > 1]
    out = {
        "edges": n_edges,
        "density": round(n_edges / max(n_nodes * (n_nodes - 1) / 2, 1), 6),
        "isolated_nodes": isolated,
        "isolated_pct": round(isolated / n_nodes * 100, 2),
        "degree_mean": round(float(degree.mean()), 3) if len(degree) else 0.0,
        "degree_max": int(degree.max()) if len(degree) else 0,
        "connected_components": len(multi),
        "largest_component_pct": round(max((len(c) for c in multi), default=0) / n_nodes * 100, 3),
    }
    if reference_edges is not None:
        out["graph_edge_overlap_vs_reference_pct"] = round(
            len(edge_set & reference_edges) / len(reference_edges) * 100, 2) if reference_edges else 0.0
    return out, edge_set


def build_pools(used: dict) -> dict:
    """(candidate, neighbour) pairs shown in each query's Top-5."""
    pools = defaultdict(set)
    for (qid, pid), ids in used.items():
        pools[qid] |= {(pid, n) for n in ids}
    return pools


def neighbour_overlap(used: dict, reference: dict, reference_pools: dict) -> dict:
    """Query-qualified comparable-neighbour overlap (distinct from graph edge overlap).

    ``used[(query_id, candidate_id)]`` holds the neighbour Property_IDs that actually carried
    graph support for that candidate in that query, i.e. exactly the comparables the explanation
    may show: already eligibility-filtered, limited and score-weighted. For the reference
    condition a condition is compared with itself, which yields identity values.
    """
    jaccards, identical, lost, gained = [], 0, [], []
    for key, ids in used.items():
        if key not in reference:
            continue
        ref_ids = reference[key]
        union = ids | ref_ids
        jaccards.append(len(ids & ref_ids) / len(union) if union else 1.0)
        identical += int(ids == ref_ids)
        lost.append(len(ref_ids - ids))
        gained.append(len(ids - ref_ids))

    pools = build_pools(used)
    pool_jaccards, pool_identical = [], 0
    for qid, pool in pools.items():
        if qid not in reference_pools:
            continue
        ref_pool = reference_pools[qid]
        union = pool | ref_pool
        pool_jaccards.append(len(pool & ref_pool) / len(union) if union else 1.0)
        pool_identical += int(pool == ref_pool)

    return {
        "candidates_compared": len(jaccards),
        "candidate_neighbour_jaccard_mean": round(float(np.mean(jaccards)), 4) if jaccards else None,
        "candidates_identical_neighbour_set_pct": round(
            identical / max(len(jaccards), 1) * 100, 2),
        "neighbours_lost_per_candidate": round(float(np.mean(lost)), 3) if lost else 0.0,
        "neighbours_gained_per_candidate": round(float(np.mean(gained)), 3) if gained else 0.0,
        "query_neighbour_pool_jaccard_mean": (round(float(np.mean(pool_jaccards)), 4)
                                             if pool_jaccards else None),
        "queries_identical_neighbour_pool_pct": round(
            pool_identical / max(len(pool_jaccards), 1) * 100, 2),
    }


def evaluate_query(qid: str, cur: list[int], ref: list[int], grades: dict,
                   labels: dict) -> dict:
    """Per-query metrics; this is the unit of the paired statistical test."""
    k = min(5, len(cur), len(ref)) or 1
    cur_grades = [grades.get(p, 0) for p in cur]
    relevant_total = sum(1 for g in grades.values() if g >= RELEVANT_GRADE)
    pooled = [labels.get((qid, p)) for p in cur]
    fully_labelled = all(g is not None for g in pooled)
    shift = [abs(i - ref.index(p)) for i, p in enumerate(cur) if p in ref]
    return {
        "query_id": qid,
        "overlap5": len(set(cur) & set(ref)) / k,
        "identical_order": int(cur == ref),
        "set_changed": int(set(cur) != set(ref)),
        "rank_shift": float(np.mean(shift)) if shift else 0.0,
        "ndcg5": ndcg_at_k(cur_grades),
        "p5": sum(1 for g in cur_grades if g >= RELEVANT_GRADE) / 5,
        "recall5": (sum(1 for g in cur_grades if g >= RELEVANT_GRADE) / relevant_total
                    if relevant_total else 0.0),
        "rr": next((1 / (i + 1) for i, g in enumerate(cur_grades) if g >= RELEVANT_GRADE), 0.0),
        "pooled_covered": sum(1 for g in pooled if g is not None),
        "pooled_retrieved": len(cur),
        "fully_labelled": int(fully_labelled),
        "ndcg5_pooled": ndcg_at_k([labels[(qid, p)] for p in cur]) if fully_labelled else None,
        "p5_pooled": (sum(1 for p in cur if labels[(qid, p)] >= RELEVANT_GRADE) / 5
                      if fully_labelled else None),
    }


def aggregate(per_query: list[dict]) -> dict:
    frame = pd.DataFrame(per_query)
    labelled = frame[frame["fully_labelled"] == 1]
    retrieved = int(frame["pooled_retrieved"].sum())
    eval_frame = frame[frame["partition"].isin(EVAL_SPLITS)]
    eval_labelled = eval_frame[eval_frame["fully_labelled"] == 1]
    return {
        "ranking_queries": len(frame),
        "top5_overlap_at5_mean": round(float(frame["overlap5"].mean()), 4),
        "topk_overlap_at10_mean": round(float(frame["overlap10"].mean()), 4),
        "top5_identical_order_pct": round(float(frame["identical_order"].mean() * 100), 2),
        "queries_with_changed_top5_pct": round(float(frame["set_changed"].mean() * 100), 2),
        "mean_abs_rank_shift": round(float(frame["rank_shift"].mean()), 3),
        "ndcg5_rubric": round(float(frame["ndcg5"].mean()), 4),
        "p5_rubric": round(float(frame["p5"].mean()), 4),
        "recall5_rubric": round(float(frame["recall5"].mean()), 4),
        "mrr_rubric": round(float(frame["rr"].mean()), 4),
        "ranking_queries_eval": len(eval_frame),
        "ndcg5_rubric_eval": round(float(eval_frame["ndcg5"].mean()), 4),
        "label_pairs_retrieved": retrieved,
        "label_pairs_covered": int(frame["pooled_covered"].sum()),
        "label_coverage_pct": round(float(frame["pooled_covered"].sum() / max(retrieved, 1) * 100), 2),
        "label_pairs_retrieved_eval": int(eval_frame["pooled_retrieved"].sum()),
        "label_pairs_covered_eval": int(eval_frame["pooled_covered"].sum()),
        "label_coverage_eval_pct": round(float(
            eval_frame["pooled_covered"].sum()
            / max(int(eval_frame["pooled_retrieved"].sum()), 1) * 100), 2),
        "queries_fully_labelled": int(len(labelled)),
        "queries_fully_labelled_pct": round(len(labelled) / max(len(frame), 1) * 100, 2),
        "ndcg5_pooled_complete_subset": (round(float(labelled["ndcg5_pooled"].mean()), 4)
                                         if len(labelled) else None),
        "p5_pooled_complete_subset": (round(float(labelled["p5_pooled"].mean()), 4)
                                      if len(labelled) else None),
        "queries_fully_labelled_eval": int(len(eval_labelled)),
        "ndcg5_pooled_eval_complete_subset": (round(float(eval_labelled["ndcg5_pooled"].mean()), 4)
                                              if len(eval_labelled) else None),
    }


# --------------------------------------------------------------------------- statistics
def paired_statistics(ref_values: np.ndarray, cur_values: np.ndarray, rng) -> dict:
    """Paired Wilcoxon + rank-biserial + bootstrap CI. Deterministic given the seed."""
    from scipy.stats import rankdata, wilcoxon

    diff = np.asarray(cur_values, dtype=float) - np.asarray(ref_values, dtype=float)
    nonzero = diff[diff != 0]
    out = {"n_pairs": int(len(diff)), "n_zero_diff": int((diff == 0).sum()),
           "mean_paired_diff": round(float(diff.mean()), 4)}
    if len(nonzero) >= 5:
        _, p = wilcoxon(cur_values, ref_values, zero_method="wilcox", alternative="two-sided")
        ranks = rankdata(np.abs(nonzero))
        t_pos = float(ranks[nonzero > 0].sum())
        t_neg = float(ranks[nonzero < 0].sum())
        out["p_wilcoxon"] = round(float(p), 6)
        out["effect_rank_biserial"] = (round((t_pos - t_neg) / (t_pos + t_neg), 4)
                                       if (t_pos + t_neg) else 0.0)
        out["test_status"] = "tested"
    else:
        out["p_wilcoxon"] = None
        out["effect_rank_biserial"] = None
        out["test_status"] = f"not_testable_only_{len(nonzero)}_nonzero_pairs"
    draws = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    low, high = np.percentile(diff[draws].mean(axis=1), [2.5, 97.5])
    out["diff_ci95_low"], out["diff_ci95_high"] = round(float(low), 4), round(float(high), 4)
    return out


def level_ci(values: np.ndarray, rng) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    low, high = np.percentile(np.asarray(values, dtype=float)[draws].mean(axis=1), [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def holm(pvalues: dict) -> dict:
    """Holm step-down adjustment over the non-reference conditions.

    An untestable condition keeps its place in the family with p = 1.0 (most conservative), so the
    declared family size of 4 is preserved instead of silently shrinking to 3.
    """
    items = sorted(((k, 1.0 if v is None else v) for k, v in pvalues.items()), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = round(running, 6)
    return adjusted


# --------------------------------------------------------------------------- explanation
def disclosure_metrics(records: list[dict], profiles: dict, audits: dict,
                       listing_info: pd.DataFrame, weights: dict) -> dict:
    """Build the real Stage 8 record for every Top-5 row and read what it discloses."""
    rows = coordinate = district = loc_none = no_neighbours = required = displayed = 0
    for rec in records:
        qid, pid = rec["query_id"], rec["Property_ID"]
        rows += 1
        shown = (rec.get("graph_support") or {}).get("neighbours", [])
        resolutions = {n.get("location_resolution") for n in shown} - {None}
        if not shown:
            no_neighbours += 1
        elif "district" in resolutions:
            district += 1
        elif resolutions <= {"coordinates"}:
            coordinate += 1
        else:                       # only "none" -> the location component was not usable
            loc_none += 1
        explanation = build_explanation(
            rec, profile=profiles[qid], direct_weights=weights[qid],
            hard_constraints=svc._property_audit(audits[qid], listing_info.loc[pid]),
        )
        if "location_precision_limited" in explanation["required_disclosures"]:
            required += 1
            if any(d.startswith(LOCATION_DISCLOSURE) for d in
                   explanation["generated_explanation"]["disclosures"]):
                displayed += 1

    def pct(value: int) -> float:
        return round(value / rows * 100, 2) if rows else 0.0

    return {
        "explanation_rows": rows,
        "coordinate_evidence_pct": pct(coordinate),
        "district_fallback_pct": pct(district),
        "location_unavailable_pct": pct(loc_none),
        "no_comparable_evidence_pct": pct(no_neighbours),
        "location_disclosure_required_pct": pct(required),
        "location_disclosure_shown_pct": round(displayed / required * 100, 2) if required else None,
        "fallback_frequency_pct": pct(district + loc_none + no_neighbours),
    }


# --------------------------------------------------------------------------- audit / pool
def verify_against_frozen(cond_top: dict) -> float | None:
    """Quote check: the reference condition must reproduce the frozen CPG Top-5 file."""
    if not FROZEN_TOP5.exists():
        return None
    frozen = pd.read_csv(FROZEN_TOP5)
    frozen = frozen[frozen["rank"] <= 5]
    matched = total = 0
    for qid, group in frozen.groupby("query_id"):
        if qid not in cond_top:
            continue
        total += 1
        expected = [int(p) for p in group.sort_values("rank")["Property_ID"]]
        matched += int(expected == [p for p, _ in cond_top[qid]])
    return round(matched / total * 100, 2) if total else None


def load_frozen_labels() -> tuple[dict, dict]:
    """Frozen pooled labels plus provenance; a missing file warns instead of reporting silent zeros."""
    if not LABELS.exists():
        print(f"WARNING: {LABELS.name} not found in frozen_queries/ - the pooled-label cross-checks "
              f"will read 0% coverage. Rebuild it from outputs/baseline/*_top5.csv with "
              f"baseline_models.rule_based_baseline.build_labels(queries, catalogue).")
        return {}, {"path": str(LABELS), "found": False, "pairs": 0}
    frame = pd.read_csv(LABELS)
    labels = {(r["query_id"], int(r["Property_ID"])): int(r["relevance"])
              for r in frame.to_dict("records")}
    return labels, {"path": str(LABELS), "found": True, "pairs": len(frame),
                    "splits": {str(k): int(v) for k, v in frame["split"].value_counts().items()}}


def write_labelling_pool(pairs: set, labelled: set, profiles: dict, catalogue: pd.DataFrame,
                         query_split: dict, path: Path) -> pd.DataFrame:
    """Blind, ungraded pairs the frozen label file does not cover, in its own column schema."""
    pending = sorted(pairs - labelled)
    info = catalogue.set_index("Property_ID")
    rows = []
    for qid, pid in pending:
        prof = profiles[qid]
        listing = info.loc[pid]
        price = pd.to_numeric(listing["price"], errors="coerce")
        size = pd.to_numeric(listing["built_up_area"], errors="coerce")
        pref = prof["preferred_built_up_sqft"]
        rows.append({
            "query_id": qid, "split": query_split[qid], "Property_ID": pid,
            "listing": _listing_name(listing),
            "district": str(listing["district"]).title(),
            "property_type": str(listing["property_type"]),
            "asking_price_rm": _int(price), "budget_rm": int(prof["max_budget_rm"]),
            "ask_pct_of_budget": (round(float(price) / float(prof["max_budget_rm"]) * 100)
                                  if pd.notna(price) else ""),
            "built_up_sqft": _int(size),
            "preferred_sqft": int(pref) if pref else "",
            "size_gap_pct": (round(abs(float(size) - pref) / pref * 100)
                             if (pd.notna(size) and pref) else ""),
            "bedrooms": _int(listing["bedroom"]), "bathrooms": _int(listing["bathroom"]),
            "main_priority": prof["main_priority"],
            "facility_match": "",
            "relevance": "", "relevance_text": "", "label_reason": "",
            "label_method": "pending", "reviewer_note": "",
        })
    frame = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    frame.to_csv(path, index=False)
    return frame


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Location-precision robustness (Section 3.14).")
    ap.add_argument("--conditions", default="", help="comma separated subset of condition names")
    args = ap.parse_args()
    wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
    unknown = wanted - {c["name"] for c in CONDITIONS}
    if unknown:
        raise SystemExit(f"unknown condition(s): {', '.join(sorted(unknown))}")
    conditions = [c for c in CONDITIONS if not wanted or c["name"] in wanted]
    if REFERENCE not in {c["name"] for c in conditions}:
        conditions.insert(0, CONDITIONS[0])

    started = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    catalogue = X.load_catalogue()
    market = X.load_market()
    query_rows = X.load_queries()
    n_nodes = len(catalogue)
    listing_info = catalogue.set_index("Property_ID")

    # graph-independent preparation: eligible sets, direct scores, rubric grades
    contexts = X.prepare_queries(catalogue, query_rows)
    profiles, audits, weights, roles, grades, splits = {}, {}, {}, {}, {}, {}
    for ctx, row in zip(contexts, query_rows.to_dict("records")):
        prof = svc.parse_profile(row)
        profiles[ctx.query_id] = prof
        weights[ctx.query_id] = svc.build_direct_query(prof)["weights"]
        roles[ctx.query_id], splits[ctx.query_id] = ctx.role, ctx.partition
        eligible, audit = svc.apply_hard_constraints(catalogue, prof)
        audits[ctx.query_id] = audit
        if not eligible.empty:
            rubric = _context(eligible, prof)
            grades[ctx.query_id] = {int(pid): grade_pair(rubric, listing_info.loc[pid])[0]
                                    for pid in eligible["Property_ID"]}
    labels, label_info = load_frozen_labels()

    role_counts = pd.Series(roles).value_counts().to_dict()
    split_role = pd.crosstab(pd.Series(splits, name="partition"), pd.Series(roles, name="role"))
    funnel = {
        "queries_total": len(contexts),
        "queries_empty_result": int(role_counts.get("empty_result_test", 0)),
        "queries_sparse_candidate": int(role_counts.get("sparse_candidate", 0)),
        "queries_ranking": int(role_counts.get("ranking", 0)),
        "ranking_queries_in_eval_splits": int(sum(
            1 for q, r in roles.items() if r == "ranking" and splits[q] in EVAL_SPLITS)),
        "by_partition": {str(k): {str(c): int(v) for c, v in row.items()}
                         for k, row in split_role.iterrows()},
        "why_not_250": ("All 250 frozen queries are executed. 89 have no property passing the 8 hard "
                        "gates (empty_result_test) and 26 have 1-4 eligible properties, below "
                        "min_ranking_candidates = 5 (sparse_candidate). Neither has a Top-5 to rank, "
                        "so only the 135 ranking queries contribute ranking metrics; the 115 "
                        "non-ranking queries are reported by the eligibility summary."),
    }
    print(f"{len(contexts)} queries · {funnel['queries_ranking']} ranking "
          f"({funnel['ranking_queries_in_eval_splits']} in validation+test) · "
          f"{funnel['queries_empty_result']} empty · {funnel['queries_sparse_candidate']} sparse · "
          f"{n_nodes} properties · {len(labels)} frozen labels\n")

    cfg = RankingConfig(variant="full", top_k=5)
    rows, detail, p_family = [], {}, {}
    per_query_all, retrieved_pairs = [], set()
    reference_top = reference_top10 = None
    reference_edges = reference_used = reference_pools = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_parameters = {}

    for condition in conditions:
        t0 = time.perf_counter()
        sim_cfg = X.similarity_config(location_mode=condition["location_mode"],
                                      coordinate_round_decimals=condition["round"])
        graph_parameters = {
            "graph_threshold": sim_cfg.graph_threshold,
            "neighbour_limit": sim_cfg.neighbour_limit,
            "retention_rule": sim_cfg.retention_rule,
            "min_valid_components": sim_cfg.min_valid_components,
            "sigma_loc_km": sim_cfg.sigma_loc_km,
            "district_fallback_value": sim_cfg.district_fallback_value,
        }
        cpg, edges, _ = X.build_graph(catalogue, sim_cfg)
        g_metrics, edge_set = graph_metrics(cpg, edges, n_nodes, reference_edges)

        cond_top, cond_top10, used, parts, records, per_query = {}, {}, {}, [], [], []
        for ctx in contexts:
            if ctx.direct is None:
                continue
            ranked, evidence = rank_candidates(ctx.direct, cfg, cpg, market)
            top = ranked[ranked["cpg_rank"] <= 5].sort_values("cpg_rank")
            cur = [int(p) for p in top["Property_ID"]]
            cond_top[ctx.query_id] = [(int(p), float(s)) for p, s
                                      in zip(top["Property_ID"], top["final_score"])]
            cond_top10[ctx.query_id] = (ranked[ranked["cpg_rank"] <= 10]
                                        .sort_values("cpg_rank")["Property_ID"].astype(int).tolist())
            for pid in cur:
                used[(ctx.query_id, pid)] = set()
            if not evidence.empty:
                for pid, group in evidence.groupby("Property_ID"):
                    key = (ctx.query_id, int(pid))
                    if key in used:      # only the comparables actually shown in Top-5
                        used[key] = set(group["neighbour_Property_ID"].astype(int))
            top = top.copy()
            top["query_id"] = ctx.query_id
            parts.append(top)
            records.extend(build_evidence(top, evidence, cfg, cpg.config_id, ctx.query_id))
            retrieved_pairs |= {(ctx.query_id, pid) for pid in cur}
        ranked_all = pd.concat(parts, ignore_index=True)

        check = None
        pools = build_pools(used)
        if reference_top is None:
            reference_top = {q: list(v) for q, v in cond_top.items()}
            reference_top10 = {q: list(v) for q, v in cond_top10.items()}
            reference_edges, reference_used, reference_pools = edge_set, used, pools
            check = verify_against_frozen(cond_top)
            if check is not None:
                print(f"  reference reproduces frozen Top-5: {check}% of queries")
        n_metrics = neighbour_overlap(used, reference_used, reference_pools)

        for qid, cur_pairs in cond_top.items():
            if roles.get(qid) != "ranking":
                continue
            cur = [p for p, _ in cur_pairs]
            ref = [p for p, _ in reference_top[qid]]
            record = evaluate_query(qid, cur, ref, grades.get(qid, {}), labels)
            record["condition"], record["partition"] = condition["name"], splits[qid]
            k10 = min(10, len(cond_top10[qid]), len(reference_top10[qid])) or 1
            record["overlap10"] = (len(set(cond_top10[qid]) & set(reference_top10[qid])) / k10)
            record["neighbour_jaccard"] = _pair_jaccard(used, reference_used, qid)
            record["top5"] = "|".join(str(p) for p in cur)
            per_query.append(record)

        agg = aggregate(per_query)
        if condition["name"] != REFERENCE:
            ref_index = pd.DataFrame([r for r in per_query_all
                                      if r["condition"] == REFERENCE]).set_index("query_id")
            cur_index = pd.DataFrame(per_query).set_index("query_id")
            common = ref_index.index.intersection(cur_index.index)
            stats = paired_statistics(ref_index.loc[common, "ndcg5"].to_numpy(),
                                      cur_index.loc[common, "ndcg5"].to_numpy(), rng)
            in_eval = np.array([splits[q] in EVAL_SPLITS for q in common])
            stats["n_pairs_eval_splits"] = int(in_eval.sum())
            stats["mean_paired_diff_eval_splits"] = (
                round(float((cur_index.loc[common, "ndcg5"].to_numpy()
                             - ref_index.loc[common, "ndcg5"].to_numpy())[in_eval].mean()), 4)
                if in_eval.any() else None)
        else:
            stats = {"n_pairs": agg["ranking_queries"], "n_zero_diff": agg["ranking_queries"],
                     "mean_paired_diff": 0.0, "p_wilcoxon": None, "effect_rank_biserial": 0.0,
                     "diff_ci95_low": 0.0, "diff_ci95_high": 0.0,
                     "n_pairs_eval_splits": funnel["ranking_queries_in_eval_splits"],
                     "mean_paired_diff_eval_splits": 0.0}
        level_low, level_high = level_ci(np.array([r["ndcg5"] for r in per_query]), rng)
        if condition["name"] != REFERENCE:
            p_family[condition["name"]] = stats["p_wilcoxon"]

        rows.append({
            "condition": condition["name"], "condition_label": condition["label"],
            "location_mode": condition["location_mode"],
            "coordinate_round_decimals": condition["round"],
            "similarity_config_id": cpg.config_id,
            "queries_total": funnel["queries_total"],
            "queries_empty_result": funnel["queries_empty_result"],
            "queries_sparse_candidate": funnel["queries_sparse_candidate"],
            **g_metrics,
            **n_metrics,
            **agg,
            "ndcg5_ci95_low": level_low, "ndcg5_ci95_high": level_high,
            **stats,
            **disclosure_metrics(records, profiles, audits, listing_info, weights),
            "frozen_top5_match_pct": check,
            "condition_seconds": round(time.perf_counter() - t0, 2),
        })
        per_query_all.extend(per_query)
        moved = [q for q in cond_top
                 if roles.get(q) == "ranking"
                 and {p for p, _ in cond_top[q]} != {p for p, _ in reference_top[q]}]
        detail[condition["name"]] = {
            "changed_top5_query_ids": moved,
            "changed_top5_examples": {q: {"reference": reference_top[q], "condition": cond_top[q]}
                                      for q in moved[:3]},
        }
        print(f"  {condition['name']:19s} edges {rows[-1]['edges']:5d} · overlap "
              f"{agg['top5_overlap_at5_mean']} · nDCG@5 {agg['ndcg5_rubric']} "
              f"[{level_low}, {level_high}] · neighbourJ "
              f"{n_metrics['candidate_neighbour_jaccard_mean']} · "
              f"coord {rows[-1]['coordinate_evidence_pct']:5.1f}%")

    out = pd.DataFrame(rows)
    adjusted = holm(p_family)
    out["p_holm"] = out.apply(
        lambda r: adjusted.get(r["condition"]) if r["test_status"] == "tested" else None, axis=1)
    out.loc[out["condition"] == REFERENCE, "p_holm"] = None
    out.to_csv(OUT_DIR / "location_robustness.csv", index=False)

    per_query_frame = pd.DataFrame(per_query_all)[[
        "condition", "query_id", "partition", "ndcg5", "p5", "recall5", "rr",
        "ndcg5_pooled", "p5_pooled", "fully_labelled", "pooled_covered", "pooled_retrieved",
        "overlap5", "overlap10", "identical_order", "set_changed", "rank_shift",
        "neighbour_jaccard", "top5"]]
    per_query_frame.to_csv(OUT_DIR / "location_robustness_per_query.csv", index=False)

    pool = write_labelling_pool(retrieved_pairs, set(labels), profiles, catalogue, splits,
                                OUT_DIR / "location_robustness_labelling_pool.csv")

    payload = {
        "experiment": "location robustness - reduced location precision",
        "section": "3.14 / Stage 11",
        "reference_condition": REFERENCE,
        "catalogue_properties": n_nodes,
        "queries": len(contexts),
        "queries_with_candidates": sum(1 for c in contexts if c.direct is not None),
        "ranking_config_id": cfg.config_id(),
        "graph_parameters": graph_parameters,
        "analysis_plan": ANALYSIS_PLAN,
        "query_funnel": funnel,
        "measurement_scopes": {
            "ranking_metrics": "135 ranking-role queries (39 in validation+test)",
            "neighbour_overlap": "Top-5 candidates of all 161 queries that have candidates, so "
                                 "sparse-result queries are included",
            "graph_edge_overlap": "all 2951 properties' retained edges, independent of any query",
            "explanation_metrics": "Top-5 rows of all queries with candidates (732 per condition)",
        },
        "label_coverage": {
            "grader": "REL-v1 rubric applied to every retrieved pair",
            "grader_coverage_pct": 100.0,
            "label_file": label_info,
            "frozen_label_file": LABELS.name,
            "frozen_label_pairs": len(labels),
            "retrieved_pairs_per_condition": [int(v) for v in out["label_pairs_retrieved"]],
            "frozen_coverage_of_retrieved_pct": [float(v) for v in out["label_coverage_pct"]],
            "pairs_needing_labels": int(len(pool)),
            "pairs_needing_labels_by_split": {str(k): int(v) for k, v in
                                              pool["split"].value_counts().items()},
            "pairs_needing_labels_eval_splits": int(pool["split"].isin(EVAL_SPLITS).sum()),
            "pool_file": "location_robustness_labelling_pool.csv",
            "note": ("Ranking metrics use the rubric grader, which reads only query and listing "
                     "attributes, so they are not limited by pooled-label coverage. The frozen "
                     "label file covers the validation+test pairs the baselines retrieved, so its "
                     "shortfall is concentrated in the train-split queries. Labelling the "
                     "eval-split pairs still listed in the pool closes the gap on the reporting "
                     "population; ndcg5_pooled_complete_subset is the cross-check that does not "
                     "depend on unlabelled items being scored as irrelevant."),
        },
        "statistics": {
            "primary_metric": "ndcg5_rubric",
            "tests": [{"condition": r["condition"], "n_pairs": r["n_pairs"],
                       "n_zero_diff": r["n_zero_diff"],
                       "mean_paired_diff": r["mean_paired_diff"],
                       "diff_ci95": [r["diff_ci95_low"], r["diff_ci95_high"]],
                       "p_wilcoxon": r["p_wilcoxon"],
                       "test_status": r["test_status"],
                       "p_holm": (adjusted.get(r["condition"])
                                  if r["test_status"] == "tested" else None),
                       "effect_rank_biserial": r["effect_rank_biserial"],
                       "significant": (r["test_status"] == "tested"
                                       and adjusted[r["condition"]] < ANALYSIS_PLAN["alpha"])}
                      for r in rows if r["condition"] != REFERENCE],
        },
        "definitions": {
            "top5_overlap_at5_mean": "mean |Top-5(ref) ∩ Top-5(cond)| / min(5, sizes), ranking queries",
            "topk_overlap_at10_mean": "same at K=10, to show stability beyond the recommendation cut",
            "ndcg5_rubric": "graded NDCG@5, gains 2^rel-1, rel from rubric REL-v1 (100% coverage)",
            "ndcg5_pooled_complete_subset": "same metric on frozen-file grades, only for queries "
                                            "whose five retrieved items are all labelled",
            "label_coverage_pct": "share of retrieved pairs covered by the frozen label file",
            "graph_edge_overlap_vs_reference_pct": "GRAPH-level retained-edge overlap; not a "
                                                   "query-qualified neighbour measure",
            "candidate_neighbour_jaccard_mean": "per Top-5 candidate, Jaccard of the neighbour sets "
                                                "that actually carried graph support vs reference "
                                                "(candidates_compared spans all 161 queries with "
                                                "candidates, while ranking metrics use the 135 "
                                                "ranking queries)",
            "query_neighbour_pool_jaccard_mean": "per query, Jaccard of the (candidate, neighbour) "
                                                 "pairs shown in Top-5 vs reference",
            "mean_paired_diff": "mean of per-query (condition - reference) NDCG@5",
            "diff_ci95": "percentile bootstrap CI of that mean paired difference",
            "p_holm": "Wilcoxon p adjusted across the four non-reference conditions",
            "effect_rank_biserial": "matched-pairs rank-biserial correlation (+ favours condition)",
            "district_fallback_pct": "Top-5 rows where a shown comparable was matched by district",
            "fallback_frequency_pct": "Top-5 rows without full coordinate-level location evidence",
            "location_disclosure_shown_pct": "share of required limited-location disclosures the "
                                             "Stage 8/9 engine actually emits",
        },
        "conditions": rows,
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (OUT_DIR / "location_robustness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nHolm-adjusted p (nDCG@5 vs reference, family = 4): "
          + ", ".join(f"{k}={v}" for k, v in adjusted.items()))
    print(f"Saved location_robustness.csv ({len(out)} conditions), "
          f"location_robustness_per_query.csv ({len(per_query_frame)} rows), "
          f"location_robustness_labelling_pool.csv ({len(pool)} unlabelled pairs), "
          f"location_robustness.json · {payload['elapsed_seconds']} s")


def _pair_jaccard(used: dict, reference: dict | None, qid: str) -> float | None:
    """Per-query Jaccard of the (candidate, neighbour) pairs shown in Top-5."""
    if reference is None:
        return None
    cur = {(p, n) for (q, p), ids in used.items() if q == qid for n in ids}
    ref = {(p, n) for (q, p), ids in reference.items() if q == qid for n in ids}
    union = cur | ref
    return round(len(cur & ref) / len(union), 4) if union else 1.0


if __name__ == "__main__":
    main()
