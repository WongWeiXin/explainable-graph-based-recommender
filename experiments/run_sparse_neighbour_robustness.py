"""Sparse comparable-neighbour robustness (Section 3.14) - Stage 11, second part.

Run from the project root:
    python experiments/run_sparse_neighbour_robustness.py
    python experiments/run_sparse_neighbour_robustness.py --conditions tau_0.85,catalogue_50

Frozen definitions (declared here, before any outcome is inspected)
    sparse_candidate    graph support status != "available" OR qualified neighbours < 3
    isolated_node       degree 0 in the retained graph
    explanation complete  all seven Stage 9 checks pass for the Top-5 row (evidence presence,
                        score linkage, comparable linkage, ranking linkage, component coverage,
                        fallback disclosure, location disclosure)
    grids               tau in {0.80, 0.85, 0.90}; k_G in {5, 3, 1}; catalogue fractions
                        {0.50, 0.25} with a fixed seed; slices are derived from the full
                        catalogue before any ranking outcome is computed

Conditions
    reference        tau 0.75, k_G 10, mutual top-k, full catalogue, coordinates
    tau_0.80/0.85/0.90   raise the retention threshold (fewer, purer edges)
    k_5/k_3/k_1      reduce the neighbour limit (thinner support per candidate)
    location_features_removed   withhold the location similarity component (weights renormalise)
    type_*           keep only one property type -> low-density communities
    low_density_districts  keep only districts in the bottom quartile of property counts
    catalogue_50/25  keep a seeded random fraction of the catalogue

    Every condition is compared with the reference on the queries that still have a Top-5 in
    both, so degradation is like-for-like; queries that lose ranking eligibility are reported
    separately as affected queries rather than silently dropped.

Reported per condition
    isolated-node rate        isolated_pct, degree statistics, components, largest component
    fallback rate             Top-5 rows without usable neighbour support, status breakdown,
                              sparse_candidate_pct under the frozen definition
    ranking degradation       overlap@5/@10, changed-set %, nDCG@5 / P@5 / Recall@5 / MRR and the
                              paired difference with bootstrap CI, Wilcoxon p and Holm p
    explanation completeness  all-checks-passed %, evidence coverage, comparable validity,
                              disclosure accuracy, comparable evidence shown, withheld claims
    affected query types      how many queries lose ranking eligibility, broken down by
                              priority, difficulty, requested property type and district

Outputs
    outputs/robustness/sparse_neighbour_robustness.csv
    outputs/robustness/sparse_neighbour_robustness_per_query.csv
    outputs/robustness/sparse_neighbour_robustness.json

The Stage 11 statistics, graph, neighbour and per-query helpers are imported from
run_location_robustness.py so both robustness stages use one definition of each metric.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "experiments", ROOT / "baseline_models"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common as X  # noqa: E402
import recommender_service as svc  # noqa: E402
from rule_based_baseline import _context, grade_pair  # noqa: E402
from cpg_ranker import RankingConfig, build_evidence, rank_candidates  # noqa: E402
from explanation_engine import build_explanation  # noqa: E402
from run_location_robustness import (  # noqa: E402  (frozen Stage 11 primitives)
    BOOTSTRAP_SEED, EVAL_SPLITS, aggregate, build_pools, evaluate_query, graph_metrics, holm,
    level_ci, load_frozen_labels, neighbour_overlap, paired_statistics, verify_against_frozen)

OUT_DIR = ROOT / "outputs" / "robustness"
LABELS = ROOT / "frozen_queries" / "relevance_labels_v2.csv"
REFERENCE = "reference"
SAMPLE_SEED = 20261031
K_SPARSE = 3

SPARSE_DEFINITION = {
    "sparse_candidate": "graph support status != 'available' OR qualified_neighbours < K_SPARSE",
    "isolated_node": "retained degree == 0",
    "explanation_complete": "all 7 Stage 9 faithfulness checks pass for that Top-5 row",
    "tau_grid": [0.80, 0.85, 0.90],
    "neighbour_limit_grid": [5, 3, 1],
    "catalogue_fraction_grid": [0.50, 0.25],
    "k_sparse": K_SPARSE,
    "sample_seed": SAMPLE_SEED,
    "district_density_rule": "keep the lowest-density districts until the slice would exceed 25% of "
                             "the catalogue, computed before any condition is run",
    "type_rule": "the two most frequent property types in the full catalogue, descending",
}

ANALYSIS_PLAN = {
    "frozen_before_running": True,
    "primary_metric": "ndcg5_rubric",
    "secondary_metrics": ["p5_rubric", "recall5_rubric", "mrr_rubric"],
    "population": "ranking-role queries that keep a Top-5 in both the condition and the reference",
    "excluded": "queries that lose ranking eligibility are reported as affected queries, not scored",
    "comparison": "each condition vs reference, paired on query_id",
    "test": "two-sided Wilcoxon signed-rank, zero_method='wilcox'",
    "alpha": 0.05,
    "multiplicity": "Holm across all non-reference conditions in this run",
    "effect_size": "matched-pairs rank-biserial correlation",
    "uncertainty": "percentile bootstrap, 10000 resamples, seed 20261031",
    "decision_rule": "report a difference only when p_holm < 0.05",
}


def build_conditions() -> list[dict]:
    conds = [{"name": REFERENCE, "group": "reference", "sim": {}, "catalogue_key": "full",
              "label": "tau 0.75, k_G 10, full catalogue (reference)"}]
    for tau in SPARSE_DEFINITION["tau_grid"]:
        conds.append({"name": f"tau_{tau:.2f}", "group": "graph_threshold",
                      "sim": {"graph_threshold": tau}, "catalogue_key": "full",
                      "label": f"graph threshold tau = {tau:.2f} (k_G 10)"})
    for k in SPARSE_DEFINITION["neighbour_limit_grid"]:
        conds.append({"name": f"k_{k}", "group": "neighbour_limit",
                      "sim": {"neighbour_limit": k}, "catalogue_key": "full",
                      "label": f"neighbour limit k_G = {k} (tau 0.75)"})
    conds.append({"name": "location_features_removed", "group": "location_features",
                  "sim": {}, "drop": ("location",), "catalogue_key": "full",
                  "label": "location features removed (similarity component withheld)"})
    for i in (1, 2):
        conds.append({"name": f"type_{i}", "group": "property_type", "sim": {},
                      "catalogue_key": f"type_{i}",
                      "label": f"only the #{i} most frequent property type"})
    conds.append({"name": "low_density_districts", "group": "low_density_area", "sim": {},
                  "catalogue_key": "low_density",
                  "label": "low-density districts only (<= 25% of the catalogue)"})
    for frac in SPARSE_DEFINITION["catalogue_fraction_grid"]:
        conds.append({"name": f"catalogue_{int(frac * 100)}", "group": "catalogue_coverage",
                      "sim": {}, "catalogue_key": f"sample_{int(frac * 100)}",
                      "label": f"seeded {int(frac * 100)}% catalogue sample"})
    return conds


# --------------------------------------------------------------------------- catalogue slices
def catalogue_slices(catalogue: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Slice definitions derived from the full catalogue before any ranking is computed."""
    type_counts = catalogue["property_type"].value_counts()
    per_district = catalogue.groupby("district")["Property_ID"].size()
    budget = 0.25 * len(catalogue)      # low-density slice holds at most a quarter of the catalogue
    low_districts, kept = [], 0
    for district, count in per_district.sort_values().items():   # ascending density
        if kept + count > budget:
            break
        low_districts.append(district)
        kept += count
    rng = np.random.default_rng(SAMPLE_SEED)

    slices = {"full": (catalogue, {"rule": "no restriction", "properties": len(catalogue)})}
    for i, (dtype, count) in enumerate(type_counts.head(2).items(), start=1):
        subset = catalogue[catalogue["property_type"] == dtype]
        slices[f"type_{i}"] = (subset, {"rule": f"property_type == {dtype} (n={count})",
                                        "properties": len(subset)})
    subset = catalogue[catalogue["district"].isin(low_districts)]
    slices["low_density"] = (subset, {
        "rule": "lowest-density districts until the slice would exceed 25% of the catalogue",
        "districts": len(low_districts), "max_properties_per_district": int(
            per_district.loc[low_districts].max()), "properties": len(subset)})
    for frac in SPARSE_DEFINITION["catalogue_fraction_grid"]:
        keep = rng.random(len(catalogue)) < frac
        subset = catalogue[keep]
        slices[f"sample_{int(frac * 100)}"] = (subset, {
            "rule": f"seeded {int(frac * 100)}% property sample", "seed": SAMPLE_SEED,
            "properties": len(subset)})
    return slices


def prepare_pack(catalogue: pd.DataFrame, query_rows: pd.DataFrame, labels: dict) -> dict:
    """Eligible sets, direct scores, weights, audits and rubric grades for one catalogue."""
    contexts = X.prepare_queries(catalogue, query_rows)
    info = catalogue.set_index("Property_ID")
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
            grades[ctx.query_id] = {int(pid): grade_pair(rubric, info.loc[pid])[0]
                                    for pid in eligible["Property_ID"]}
    return {"catalogue": catalogue, "contexts": contexts, "profiles": profiles, "audits": audits,
            "weights": weights, "roles": roles, "grades": grades, "splits": splits,
            "labels": labels,
            "cond_role": {ctx.query_id: ("ranking" if ctx.eligible_count >= 5
                                        else "sparse_candidate" if ctx.eligible_count > 0
                                        else "empty_result_test")
                          for ctx in contexts}}


# --------------------------------------------------------------------------- sparse metrics
def sparse_metrics(ranked_all: pd.DataFrame) -> dict:
    """Fallback exposure of the Top-5 under the frozen sparse-neighbour definition."""
    top = ranked_all[ranked_all["cpg_rank"] <= 5]
    status = top["graph_support_status"]
    qualified = pd.to_numeric(top["qualified_neighbours"], errors="coerce").fillna(0)
    available = status.eq("available")
    sparse = (~available) | (qualified < K_SPARSE)
    return {
        "top5_graph_available_pct": round(float(available.mean() * 100), 2),
        "fallback_rate_pct": round(float((~available).mean() * 100), 2),
        "sparse_candidate_pct": round(float(sparse.mean() * 100), 2),
        "mean_qualified_neighbours_top5": round(float(qualified.mean()), 3),
        "status_isolated_pct": round(float(
            status.eq("unavailable_no_retained_neighbours").mean() * 100), 2),
        "status_none_eligible_pct": round(float(
            status.eq("unavailable_no_neighbour_meets_hard_constraints").mean() * 100), 2),
        "status_no_direct_pct": round(float(
            status.eq("unavailable_neighbours_have_no_direct_score").mean() * 100), 2),
        "status_not_in_graph_pct": round(float(
            status.eq("unavailable_candidate_not_in_graph").mean() * 100), 2),
    }


def explanation_metrics(records: list[dict], profiles: dict, audits: dict,
                        listing_info: pd.DataFrame, weights: dict) -> dict:
    """Explanation completeness for the Top-5 rows: does the engine still explain fully?"""
    rows = passed = with_comparable = no_comparable = no_comparable_disclosed = 0
    evidence_cov, comparable_validity, disclosure_acc, withheld = [], [], [], []
    graph_disclosure_required = graph_disclosure_shown = 0
    for rec in records:
        qid, pid = rec["query_id"], rec["Property_ID"]
        rows += 1
        shown = (rec.get("graph_support") or {}).get("neighbours", [])
        with_comparable += int(bool(shown))
        explanation = build_explanation(
            rec, profile=profiles[qid], direct_weights=weights[qid],
            hard_constraints=svc._property_audit(audits[qid], listing_info.loc[pid]),
        )
        faith = explanation["faithfulness"]
        passed += int(faith["passed"])
        evidence_cov.append(faith["metrics"]["evidence_coverage"])
        comparable_validity.append(faith["metrics"]["comparable_neighbour_validity"])
        disclosure_acc.append(faith["metrics"]["fallback_disclosure_accuracy"])
        withheld.append(faith["claims_withheld"])
        disclosed_graph = any(d.startswith("Graph support unavailable")
                              for d in explanation["generated_explanation"]["disclosures"])
        if "graph_support_unavailable" in explanation["required_disclosures"]:
            graph_disclosure_required += 1
            graph_disclosure_shown += int(disclosed_graph)
        if not shown:
            no_comparable += 1
            no_comparable_disclosed += int(disclosed_graph)

    def mean(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 4) if values else None

    def pct(value: int) -> float:
        return round(value / rows * 100, 2) if rows else 0.0

    return {
        "explanation_rows": rows,
        "explanations_all_checks_passed_pct": pct(passed),
        "explanations_with_comparable_evidence_pct": pct(with_comparable),
        "explanations_without_comparables_pct": pct(no_comparable),
        "missing_comparables_disclosed_pct": (round(no_comparable_disclosed / no_comparable * 100, 2)
                                             if no_comparable else None),
        "evidence_coverage_mean": mean(evidence_cov),
        "comparable_neighbour_validity_mean": mean(comparable_validity),
        "fallback_disclosure_accuracy_mean": mean(disclosure_acc),
        "mean_claims_withheld": mean(withheld),
        "graph_unavailable_disclosure_required_pct": pct(graph_disclosure_required),
        "graph_unavailable_disclosure_shown_pct": (
            round(graph_disclosure_shown / graph_disclosure_required * 100, 2)
            if graph_disclosure_required else None),
    }


def affected_queries(cond_roles: dict, ref_roles: dict, meta: dict) -> dict:
    """Which query types lose ranking eligibility under this condition."""
    lost = [q for q, role in ref_roles.items() if role == "ranking" and cond_roles.get(q) != "ranking"]
    total = sum(1 for role in ref_roles.values() if role == "ranking")
    breakdown = {}
    for field in ("main_priority", "difficulty", "property_type"):
        counts: dict[str, int] = {}
        for qid in lost:
            key = str(meta.get(qid, {}).get(field, "unknown"))
            counts[key] = counts.get(key, 0) + 1
        breakdown[field] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return {
        "reference_ranked_queries": total,
        "condition_ranked_queries": sum(1 for q, role in cond_roles.items()
                                        if role == "ranking" and q in ref_roles),
        "queries_lost_ranking": len(lost),
        "queries_lost_ranking_pct_of_reference": round(len(lost) / max(total, 1) * 100, 2),
        "denominator": f"{total} reference ranking queries (benchmark_role == 'ranking' on the "
                       f"full catalogue), not all 250 frozen queries",
        "lost_by_priority": breakdown["main_priority"],
        "lost_by_difficulty": breakdown["difficulty"],
        "lost_by_property_type": breakdown["property_type"],
        "lost_query_ids": lost,
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Sparse comparable-neighbour robustness (3.14).")
    ap.add_argument("--conditions", default="", help="comma separated subset of condition names")
    args = ap.parse_args()
    all_conditions = build_conditions()
    wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
    unknown = wanted - {c["name"] for c in all_conditions}
    if unknown:
        raise SystemExit(f"unknown condition(s): {', '.join(sorted(unknown))}")
    conditions = [c for c in all_conditions if not wanted or c["name"] in wanted]
    if REFERENCE not in {c["name"] for c in conditions}:
        conditions.insert(0, all_conditions[0])

    started = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    catalogue = X.load_catalogue()
    market = X.load_market()
    query_rows = X.load_queries()
    labels, label_info = load_frozen_labels()
    slices = catalogue_slices(catalogue)
    meta = {r["query_id"]: r for r in query_rows.to_dict("records")}

    packs: dict[str, dict] = {}

    def pack(key: str) -> dict:
        if key not in packs:
            t = time.perf_counter()
            packs[key] = prepare_pack(slices[key][0], query_rows, labels)
            print(f"  prepared catalogue slice '{key}' ({len(slices[key][0])} properties) "
                  f"in {time.perf_counter() - t:.1f}s")
        return packs[key]

    slice_summary = ", ".join(f"{key}={value[1]['properties']}" for key, value in slices.items())
    print(f"{len(query_rows)} queries · {len(catalogue)} properties · slices: {slice_summary}\n")

    cfg = RankingConfig(variant="full", top_k=5)
    rows, detail, p_family, per_query_all = [], {}, {}, []
    reference_top = reference_top10 = None
    reference_edges = reference_used = reference_pools = None
    reference_roles = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_parameters: dict = {}

    for condition in conditions:
        t0 = time.perf_counter()
        current = pack(condition["catalogue_key"])
        sim_cfg = X.similarity_config(drop=condition.get("drop", ()), **condition["sim"])
        graph_parameters = {
            "graph_threshold": sim_cfg.graph_threshold,
            "neighbour_limit": sim_cfg.neighbour_limit,
            "retention_rule": sim_cfg.retention_rule,
            "active_components": list(sim_cfg.active_components),
            "min_valid_components": sim_cfg.min_valid_components,
        }
        cpg, edges, _ = X.build_graph(current["catalogue"], sim_cfg)
        g_metrics, edge_set = graph_metrics(cpg, edges, len(current["catalogue"]), reference_edges)
        g_metrics["properties_in_scope"] = len(current["catalogue"])

        cond_top, cond_top10, used, parts, records, per_query = {}, {}, {}, [], [], []
        for ctx in current["contexts"]:
            if ctx.direct is None:
                continue
            ranked, evidence = rank_candidates(ctx.direct, cfg, cpg, market)
            top = ranked[ranked["cpg_rank"] <= 5].sort_values("cpg_rank")
            cur = [int(p) for p in top["Property_ID"]]
            cond_top[ctx.query_id] = [(int(p), float(s)) for p, s in
                                      zip(top["Property_ID"], top["final_score"])]
            cond_top10[ctx.query_id] = (ranked[ranked["cpg_rank"] <= 10]
                                        .sort_values("cpg_rank")["Property_ID"].astype(int).tolist())
            for pid in cur:
                used[(ctx.query_id, pid)] = set()
            if not evidence.empty:
                for pid, group in evidence.groupby("Property_ID"):
                    key = (ctx.query_id, int(pid))
                    if key in used:
                        used[key] = set(group["neighbour_Property_ID"].astype(int))
            top = top.copy()
            top["query_id"] = ctx.query_id
            parts.append(top)
            records.extend(build_evidence(top, evidence, cfg, cpg.config_id, ctx.query_id))
        ranked_all = pd.concat(parts, ignore_index=True)

        check = None
        pools = build_pools(used)
        if reference_top is None:
            reference_top = {q: list(v) for q, v in cond_top.items()}
            reference_top10 = {q: list(v) for q, v in cond_top10.items()}
            reference_edges, reference_used, reference_pools = edge_set, used, pools
            reference_roles = dict(current["cond_role"])
            check = verify_against_frozen(cond_top)
            if check is not None:
                print(f"  reference reproduces frozen Top-5: {check}% of queries")

        paired = [q for q, role in current["cond_role"].items()
                  if role == "ranking" and q in cond_top and len(cond_top[q]) == 5
                  and q in reference_top and len(reference_top[q]) == 5]
        paired_ids = set(paired)
        records = [r for r in records if r["query_id"] in paired_ids]
        n_metrics = neighbour_overlap(used, reference_used, reference_pools)
        condition_ids = set(current["catalogue"]["Property_ID"].astype(int))

        for qid in paired:
            cur = [p for p, _ in cond_top[qid]]
            ref = [p for p, _ in reference_top[qid]]
            record = evaluate_query(qid, cur, ref, current["grades"].get(qid, {}), labels)
            record["condition"], record["partition"] = condition["name"], current["splits"][qid]
            k10 = min(10, len(cond_top10[qid]), len(reference_top10[qid])) or 1
            record["overlap10"] = len(set(cond_top10[qid]) & set(reference_top10[qid])) / k10
            # Universe-aware overlap: the reference Top-5 restricted to properties that still exist
            # in this condition's catalogue, so removed properties are not scored as rank errors.
            available = [p for p in ref if p in condition_ids]
            record["reference_top5_available"] = len(available) / 5
            record["overlap_vs_available_reference"] = (
                len(set(cur) & set(available)) / (min(5, len(available)) or 1))
            record["neighbour_jaccard"] = None
            record["top5"] = "|".join(str(p) for p in cur)
            per_query.append(record)

        agg = aggregate(per_query)
        frame = pd.DataFrame(per_query)
        agg["reference_top5_available_pct"] = round(
            float(frame["reference_top5_available"].mean() * 100), 2)
        agg["top5_overlap_vs_available_reference"] = round(
            float(frame["overlap_vs_available_reference"].mean()), 4)
        sparse = sparse_metrics(ranked_all[ranked_all["query_id"].isin(paired_ids)])
        affected = affected_queries(current["cond_role"], reference_roles, meta)

        if condition["name"] == REFERENCE:
            stats = {"n_pairs": len(per_query), "n_zero_diff": len(per_query),
                     "mean_paired_diff": 0.0, "p_wilcoxon": None, "effect_rank_biserial": 0.0,
                     "diff_ci95_low": 0.0, "diff_ci95_high": 0.0, "test_status": "reference",
                     "n_pairs_eval_splits": 0, "mean_paired_diff_eval_splits": 0.0}
        else:
            ref_index = pd.DataFrame([r for r in per_query_all
                                      if r["condition"] == REFERENCE]).set_index("query_id")
            cur_index = pd.DataFrame(per_query).set_index("query_id")
            common = ref_index.index.intersection(cur_index.index)
            stats = paired_statistics(ref_index.loc[common, "ndcg5"].to_numpy(),
                                      cur_index.loc[common, "ndcg5"].to_numpy(), rng)
            in_eval = np.array([current["splits"][q] in EVAL_SPLITS for q in common])
            stats["n_pairs_eval_splits"] = int(in_eval.sum())
            stats["mean_paired_diff_eval_splits"] = (
                round(float((cur_index.loc[common, "ndcg5"].to_numpy()
                             - ref_index.loc[common, "ndcg5"].to_numpy())[in_eval].mean()), 4)
                if in_eval.any() else None)
            p_family[condition["name"]] = stats["p_wilcoxon"]
        level_low, level_high = level_ci(np.array([r["ndcg5"] for r in per_query]), rng)

        rows.append({
            "condition": condition["name"], "condition_group": condition["group"],
            "condition_label": condition["label"],
            "catalogue_slice": condition["catalogue_key"],
            "graph_threshold": sim_cfg.graph_threshold,
            "neighbour_limit": sim_cfg.neighbour_limit,
            "location_component": "location" in sim_cfg.active_components,
            "sparse_definition": f"unavailable or qualified_neighbours < {K_SPARSE}",
            **g_metrics,
            **sparse,
            **n_metrics,
            **agg,
            "ndcg5_ci95_low": level_low, "ndcg5_ci95_high": level_high,
            **stats,
            "queries_lost_ranking": affected["queries_lost_ranking"],
            "queries_lost_ranking_pct_of_reference":
                affected["queries_lost_ranking_pct_of_reference"],
            **explanation_metrics(records, current["profiles"], current["audits"],
                                  current["catalogue"].set_index("Property_ID"),
                                  current["weights"]),
            "frozen_top5_match_pct": check,
            "condition_seconds": round(time.perf_counter() - t0, 2),
        })
        per_query_all.extend(per_query)
        detail[condition["name"]] = {
            "affected": affected,
            "slice": slices[condition["catalogue_key"]][1],
            "changed_top5_query_ids": [q for q in paired
                                       if {p for p, _ in cond_top[q]}
                                       != {p for p, _ in reference_top[q]}],
        }
        print(f"  {condition['name']:22s} nodes {len(current['catalogue']):5d} · "
              f"edges {rows[-1]['edges']:5d} · isolated {rows[-1]['isolated_pct']:5.2f}% · "
              f"fallback {rows[-1]['fallback_rate_pct']:5.2f}% · "
              f"nDCG@5 {agg['ndcg5_rubric']} · overlap {agg['top5_overlap_at5_mean']} · "
              f"expl-complete {rows[-1]['explanations_all_checks_passed_pct']:5.1f}% · "
              f"lost {affected['queries_lost_ranking']}")

    out = pd.DataFrame(rows)
    adjusted = holm(p_family)
    out["p_holm"] = out.apply(
        lambda r: adjusted.get(r["condition"]) if r["test_status"] == "tested" else None, axis=1)
    out.loc[out["condition"] == REFERENCE, "p_holm"] = None
    out.to_csv(OUT_DIR / "sparse_neighbour_robustness.csv", index=False)

    per_query_frame = pd.DataFrame(per_query_all)[[
        "condition", "query_id", "partition", "ndcg5", "p5", "recall5", "rr",
        "ndcg5_pooled", "p5_pooled", "fully_labelled", "pooled_covered", "pooled_retrieved",
        "overlap5", "overlap10", "identical_order", "set_changed", "rank_shift", "top5"]]
    per_query_frame.to_csv(OUT_DIR / "sparse_neighbour_robustness_per_query.csv", index=False)

    payload = {
        "experiment": "sparse comparable-neighbour robustness",
        "section": "3.14 / Stage 11 (second part)",
        "reference_condition": REFERENCE,
        "catalogue_properties": len(catalogue),
        "queries": len(query_rows),
        "ranking_config_id": cfg.config_id(),
        "sparse_definition": SPARSE_DEFINITION,
        "analysis_plan": ANALYSIS_PLAN,
        "label_coverage_source": label_info,
        "slices": {k: v[1] for k, v in slices.items()},
        "graph_parameters_reference": graph_parameters,
        "statistics": {
            "primary_metric": "ndcg5_rubric",
            "tests": [{"condition": r["condition"], "n_pairs": r["n_pairs"],
                       "n_zero_diff": r["n_zero_diff"],
                       "mean_paired_diff": r["mean_paired_diff"],
                       "diff_ci95": [r["diff_ci95_low"], r["diff_ci95_high"]],
                       "p_wilcoxon": r["p_wilcoxon"], "test_status": r["test_status"],
                       "p_holm": (adjusted.get(r["condition"])
                                  if r["test_status"] == "tested" else None),
                       "effect_rank_biserial": r["effect_rank_biserial"],
                       "significant": (r["test_status"] == "tested"
                                       and adjusted[r["condition"]] < ANALYSIS_PLAN["alpha"])}
                      for r in rows if r["condition"] != REFERENCE],
        },
        "definitions": {
            "isolated_pct": "properties with retained degree 0 (isolated-node rate)",
            "fallback_rate_pct": "Top-5 rows whose graph support status is not 'available'",
            "sparse_candidate_pct": "Top-5 rows matching the frozen sparse-neighbour definition",
            "mean_qualified_neighbours_top5": "eligible neighbours available to each Top-5 candidate",
            "ndcg5_rubric": "graded NDCG@5 on the paired population, rubric REL-v1 (100% coverage)",
            "mean_paired_diff": "mean per-query (condition - reference) NDCG@5",
            "top5_overlap_vs_available_reference": "|Top-5(condition) ∩ Top-5(reference restricted to "
                                                   "properties still in this catalogue)| / min(5, that "
                                                   "size) - separates rank change from universe change",
            "reference_top5_available_pct": "share of the reference Top-5 that still exists in this "
                                            "condition's catalogue",
            "queries_lost_ranking": "queries ranked in the reference but below 5 eligible properties here",
            "queries_lost_ranking_pct_of_reference": "queries_lost_ranking / 135 reference ranking "
                                                     "queries (not 250 frozen queries)",
            "explanations_all_checks_passed_pct": "denominator = Top-5 rows of the paired queries in "
                                                  "this condition; numerator = rows where all 7 Stage 9 "
                                                  "checks pass (evidence presence, score linkage, "
                                                  "comparable linkage, ranking linkage, component "
                                                  "coverage, fallback disclosure, location disclosure)",
            "explanations_with_comparable_evidence_pct": "denominator = Top-5 rows; numerator = rows "
                                                         "whose retained neighbour list "
                                                         "(graph_support.neighbours) is non-empty",
            "missing_comparables_disclosed_pct": "denominator = rows with an empty neighbour list; "
                                                 "numerator = those rows whose generated explanation "
                                                 "contains the accepted claim that discloses "
                                                 "graph_support_unavailable",
            "comparable_neighbour_validity_mean": "per row, displayed comparable claims whose "
                                                  "neighbour_id is in the shown neighbour set / all "
                                                  "displayed comparable claims; 1.0 when no comparable "
                                                  "claim was displayed",
            "fallback_disclosure_accuracy_mean": "per row, required disclosures present / required "
                                                 "disclosures; 1.0 when nothing is required to be "
                                                 "disclosed",
            "evidence_coverage_mean": "per row, explanation blocks referenced by an accepted claim / "
                                      "blocks available in that evidence record",
            "graph_unavailable_disclosure_shown_pct": "denominator = rows where graph_support_unavailable "
                                                      "is required; numerator = rows that emit it",
            "mean_claims_withheld": "mean number of generated claims rejected by the Stage 9 validator "
                                    "per Top-5 row",
        },
        "conditions": rows,
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (OUT_DIR / "sparse_neighbour_robustness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nHolm-adjusted p (nDCG@5 vs reference): "
          + ", ".join(f"{k}={v}" for k, v in adjusted.items()))
    print(f"Saved sparse_neighbour_robustness.csv ({len(out)} conditions), "
          f"sparse_neighbour_robustness_per_query.csv ({len(per_query_frame)} rows) and "
          f"sparse_neighbour_robustness.json · {payload['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
