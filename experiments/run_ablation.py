"""Component ablation (Sec 3.13), robustness and parameter sensitivity (Sec 3.14).

Run from the project root:
    python experiments/run_ablation.py                  # ablation + robustness (10 conditions)
    python experiments/run_ablation.py --sensitivity    # + tau/k/sigma grid and lambda grid

Conditions
    full                    frozen CPG configuration (reference condition)
    no_<component>          graph rebuilt without one similarity component; the remaining
                            weights are renormalised (location, price, size, property_type,
                            bedroom, bathroom, facility)
    district_only           location robustness: coordinates withheld, district fallback only
    rank_direct_only        ranking-level ablation: no graph at all (= CBR)
    rank_full_market        separately reported variant: + market context
    sens_t<tau>_k<k>_s<sigma>   graph-parameter sensitivity grid      (--sensitivity)
    sens_lambda_<d>_<g>         direct/graph weight sensitivity       (--sensitivity)

Every condition uses the same v2 queries, hard constraints and direct scores; only the graph
(or the ranking variant) changes. The frozen graph files are NOT overwritten.

Outputs (outputs/ablation/):
    conditions_summary.csv        one row per condition: graph structure + evidence + Top-5 change
    top5_all_conditions.csv       condition x query x rank -> Property_ID (input for evaluation)
    labelling_pool_additions.csv  (query, property) pairs from validation/test ranking queries that
                                  are NOT already in the CPG / CBR / LightGCN Top-5 -> add these to
                                  the relevance-labelling pool BEFORE labelling
    graphs/<condition>_edges.csv  retained edges for each rebuilt graph
    ablation_run.json             config ids, sanity checks, timing

Ranking metrics (NDCG@5, P@5, Recall@5, MRR) are computed later, once the graded relevance
labels exist, from top5_all_conditions.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from common import (COMPONENTS, FROZEN_EDGES, G_AVAILABLE, ROOT, build_graph, load_catalogue,
                    load_market, load_queries, prepare_queries, rank_query, similarity_config)

OUT_DIR = ROOT / "outputs" / "ablation"
TOP5_DIR = ROOT / "outputs" / "top5"
LIGHTGCN_TOP5 = ROOT / "baseline_models" / "lightgcn_outputs" / "lightgcn_top5_recommendations.csv"

# Predeclared grids (Sec 4.11 candidates; 0.75 = current frozen tau, added so the reference is in
# the grid). Select on VALIDATION queries only: NDCG@5 -> MRR -> Recall@5.
TAU_GRID = (0.50, 0.60, 0.70, 0.75)
K_GRID = (10, 20, 30)
SIGMA_GRID = (1.0, 3.0, 5.0)
LAMBDA_GRID = ((0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5))


def conditions(sensitivity: bool) -> list[dict]:
    conds = [{"condition": "full", "group": "reference", "sim": {}, "variant": "full"}]
    for c in COMPONENTS:
        conds.append({"condition": f"no_{c}", "group": "component_ablation",
                      "sim": {"drop": (c,)}, "variant": "full", "removed": c})
    conds.append({"condition": "district_only", "group": "location_robustness",
                  "sim": {"location_mode": "district_only"}, "variant": "full"})
    conds.append({"condition": "rank_direct_only", "group": "ranking_ablation", "sim": None,
                  "variant": "direct_only"})
    conds.append({"condition": "rank_full_market", "group": "ranking_variant", "sim": {},
                  "variant": "full_market"})
    if sensitivity:
        for tau, k, s in itertools.product(TAU_GRID, K_GRID, SIGMA_GRID):
            conds.append({"condition": f"sens_t{tau:.2f}_k{k}_s{s:g}", "group": "graph_sensitivity",
                          "sim": {"graph_threshold": tau, "neighbour_limit": k, "sigma_loc_km": s},
                          "variant": "full"})
        for d, g in LAMBDA_GRID:
            conds.append({"condition": f"sens_lambda_{d:.1f}_{g:.1f}", "group": "lambda_sensitivity",
                          "sim": {}, "variant": "full",
                          "lambdas": {"direct": d, "graph": g, "market": 0.2}})
    return conds


def evidence_stats(top: pd.DataFrame, subset: str) -> dict:
    t = top if subset == "all" else top[top["benchmark_role"] == "ranking"]
    if t.empty:
        return {}
    g = t["graph_support_status"].eq(G_AVAILABLE)
    return {f"{subset}_top5_rows": int(len(t)),
            f"{subset}_graph_supported_pct": round(float(g.mean() * 100), 2),
            f"{subset}_mean_qualified_neighbours": round(float(t["qualified_neighbours"].fillna(0).mean()), 3),
            f"{subset}_queries_all_top5_graph_supported_pct": round(float(
                t.groupby("query_id")["graph_support_status"].apply(lambda s: s.eq(G_AVAILABLE).all()).mean()
                * 100), 2)}


def change_vs_reference(top: pd.DataFrame, ref: pd.DataFrame) -> dict:
    a = ref.groupby("query_id")["Property_ID"].apply(list)
    b = top.groupby("query_id")["Property_ID"].apply(list)
    qs = a.index.intersection(b.index)
    if len(qs) == 0:
        return {}
    overlap = [len(set(a[q]) & set(b[q])) / max(len(a[q]), 1) for q in qs]
    return {"top5_overlap_with_full": round(float(np.mean(overlap)), 4),
            "identical_top5_order_pct": round(float(np.mean([a[q] == b[q] for q in qs]) * 100), 2),
            "queries_with_changed_set": int(sum(o < 1 for o in overlap))}


def existing_pool() -> set:
    pairs = set()
    files = [TOP5_DIR / "cpg_top5_recommendations.csv", TOP5_DIR / "cbr_top5_recommendations.csv",
             LIGHTGCN_TOP5]
    for f in files:
        if f.exists():
            d = pd.read_csv(f).dropna(subset=["Property_ID"])
            pairs |= set(zip(d["query_id"].astype(str), d["Property_ID"].astype(int)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation, robustness and sensitivity conditions.")
    ap.add_argument("--sensitivity", action="store_true", help="Also run the tau/k/sigma and lambda grids.")
    ap.add_argument("--only", nargs="*", help="Run only these conditions (plus 'full'); written to "
                                               "outputs/ablation/partial_runs/.")
    args = ap.parse_args()
    started = time.perf_counter()

    catalogue = load_catalogue()
    market = load_market()
    queries = load_queries()
    print(f"Preparing {len(queries)} queries (hard constraints + direct scores) ...")
    contexts = prepare_queries(catalogue, queries)
    active = [c for c in contexts if c.direct is not None]

    conds = conditions(args.sensitivity)
    out_dir = OUT_DIR
    if args.only:
        keep = set(args.only) | {"full"}
        unknown = keep - {c["condition"] for c in conds}
        if unknown:
            raise SystemExit(f"Unknown condition(s): {sorted(unknown)} (add --sensitivity for sens_*)")
        conds = [c for c in conds if c["condition"] in keep]
        out_dir = OUT_DIR / "partial_runs"   # never overwrite the main results with a partial run

    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    graph_cache: dict[str, tuple] = {}
    rows, top_rows, checks = [], [], {}
    ref_top = None

    for cond in conds:
        name = cond["condition"]
        t0 = time.perf_counter()
        cpg, summary, sim_id = None, {}, None
        if cond["sim"] is not None:
            sim_cfg = similarity_config(**cond["sim"])
            sim_id = sim_cfg.config_id()
            if sim_id not in graph_cache:
                cpg, edges, _ = build_graph(catalogue, sim_cfg)
                graph_cache[sim_id] = (cpg, round(time.perf_counter() - t0, 2))
                edges.to_csv(out_dir / "graphs" / f"{name}_edges.csv", index=False)
            cpg, build_s = graph_cache[sim_id]
            summary = cpg.summary()
            summary["build_seconds"] = build_s

        recs = []
        for ctx in active:
            top = rank_query(ctx, cond["variant"], cpg, market, cond.get("lambdas"))
            for r in top.itertuples(index=False):
                recs.append({"condition": name, "query_id": ctx.query_id, "partition": ctx.partition,
                             "benchmark_role": ctx.role, "rank": int(r.cpg_rank),
                             "Property_ID": int(r.Property_ID),
                             "final_score": round(float(r.final_score), 6),
                             "graph_support_status": r.graph_support_status,
                             "qualified_neighbours": r.qualified_neighbours})
        top = pd.DataFrame(recs)
        top_rows.append(top)
        if name == "full":
            ref_top = top

        deg = summary.get("degree", {})
        row = {"condition": name, "group": cond["group"], "removed_component": cond.get("removed"),
               "ranking_variant": cond["variant"], "similarity_config_id": sim_id,
               "edges": summary.get("edges"), "density": summary.get("density"),
               "isolated_nodes": summary.get("isolated_nodes"), "isolated_pct": summary.get("isolated_pct"),
               "degree_mean": deg.get("mean"), "degree_median": deg.get("median"), "degree_max": deg.get("max"),
               "connected_components": summary.get("connected_components_excluding_isolated"),
               "largest_component_pct": summary.get("largest_component_pct"),
               "edges_with_similarity_1": summary.get("edges_with_similarity_1"),
               "graph_build_seconds": summary.get("build_seconds"),
               **evidence_stats(top, "all"), **evidence_stats(top, "ranking"),
               **change_vs_reference(top, ref_top),
               "condition_seconds": round(time.perf_counter() - t0, 2)}
        rows.append(row)
        print(f"{name:<28} edges={str(row['edges']):>6}  isolated%={str(row['isolated_pct']):>6}  "
              f"graph-supported Top-5%={row.get('all_graph_supported_pct')}  "
              f"overlap={row.get('top5_overlap_with_full')}  ({row['condition_seconds']} s)")

        if name == "full":
            frozen = pd.read_csv(FROZEN_EDGES) if FROZEN_EDGES.exists() else None
            checks["full_matches_frozen_graph"] = None if frozen is None else bool(
                len(frozen) == summary["edges"]
                and frozen["config_id"].iloc[0] == summary["config_id"])
            checks["full_similarity_config_id"] = summary["config_id"]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "conditions_summary.csv", index=False)
    all_top = pd.concat(top_rows, ignore_index=True)
    all_top.to_csv(out_dir / "top5_all_conditions.csv", index=False)

    # Pairs to add to the relevance-labelling pool (validation + test ranking queries only).
    eval_top = all_top[(all_top["benchmark_role"] == "ranking")
                       & all_top["partition"].isin(["validation", "test"])]
    pool = existing_pool()
    pairs = eval_top.drop_duplicates(["query_id", "Property_ID"])
    new = pairs[[(q, p) not in pool for q, p in zip(pairs["query_id"].astype(str), pairs["Property_ID"])]]
    new_pairs = (new.groupby(["query_id", "partition", "Property_ID"])["condition"]
                 .apply(lambda s: ";".join(sorted(set(s)))).reset_index()
                 .rename(columns={"condition": "found_by_conditions"}))
    new_pairs.to_csv(out_dir / "labelling_pool_additions.csv", index=False)

    checks["queries_ranked"] = len(active)
    checks["empty_queries_skipped"] = len(contexts) - len(active)
    checks["existing_pool_pairs"] = len(pool)
    checks["labelling_pool_additions"] = int(len(new_pairs))
    (out_dir / "ablation_run.json").write_text(json.dumps({
        "conditions": [c["condition"] for c in conds],
        "grids": {"tau": TAU_GRID, "k": K_GRID, "sigma_km": SIGMA_GRID, "lambda_direct_graph": LAMBDA_GRID}
        if args.sensitivity else None,
        "checks": checks,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nChecks: {checks}")
    print(f"Saved: {out_dir / 'conditions_summary.csv'}")
    print(f"       {out_dir / 'top5_all_conditions.csv'}")
    print(f"       {out_dir / 'labelling_pool_additions.csv'} ({len(new_pairs)} new pairs to label)")


if __name__ == "__main__":
    main()
