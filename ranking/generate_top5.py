"""Top-5 recommendations for every benchmark query (CPG, or CBR via --variant direct_only).

Run from the project root:
    python ranking/generate_top5.py                         # CPG (direct + graph support)
    python ranking/generate_top5.py --variant direct_only   # CBR baseline (same pipeline, no graph)

Uses exactly the same code as the web app (recommender_service.py):
    hard constraints -> direct preference score -> qualified CPG neighbours + graph support
    -> final score -> deterministic ranking
so the benchmark results and the prototype cannot disagree.

Inputs:
    frozen_queries/frozen_queries_set_v2.csv
    benchmark/eligibility_manifest_v2.csv
    outputs/comparable_property_graph/* (frozen graph), outputs/market_context/* (if used)

Outputs (outputs/top5/):
    <method>_top5_recommendations.csv   one row per (query, rank); empty queries get one row
    <method>_top5_evidence.json         per-query score trace of the Top-5 (explanation input)
    <method>_top5_summary.json          config ids, counts, integrity checks, timing
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
sys.path.insert(0, str(ROOT))

import recommender_service as svc  # noqa: E402
from cpg_ranker import G_AVAILABLE, VARIANTS, RankingConfig, rank_candidates  # noqa: E402

QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
MANIFEST = ROOT / "benchmark" / "eligibility_manifest_v2.csv"
OUT_DIR = ROOT / "outputs" / "top5"
TOP_N = 5
METHOD_NAME = {"full": "cpg", "direct_only": "cbr", "full_market": "cpg_market",
               "direct_market": "cbr_market"}
DETAIL_COLS = ["price", "property_type", "district", "bedroom", "bathroom", "built_up_area",
               "car_parks", "furnishing", "tenure"]


def _clean(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-5 per benchmark query (CPG / CBR).")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--queries", type=Path, default=QUERIES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()
    method = METHOD_NAME[args.variant]
    started = time.perf_counter()

    queries = pd.read_csv(args.queries, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False).set_index("query_id")
    catalogue, cpg, market = svc._Resources.get()
    info = catalogue.set_index("Property_ID")
    cfg = RankingConfig(variant=args.variant, top_k=TOP_N)

    rows, evidence, errors = [], {}, []
    violations = 0
    for _, q in queries.iterrows():
        qid = q["query_id"]
        meta = manifest.loc[qid] if qid in manifest.index else {}
        base = {
            "method": method,
            "variant": args.variant,
            "query_id": qid,
            "partition": q.get("partition", ""),
            "difficulty": q.get("difficulty", ""),
            "benchmark_role": meta.get("benchmark_role", "") if len(meta) else "",
        }
        try:
            prof = svc.parse_profile(q.to_dict())
        except svc.QueryError as exc:
            errors.append({"query_id": qid, "error": str(exc)})
            rows.append({**base, "eligible_count": None, "rank": None, "Property_ID": None,
                         "status": f"invalid_query: {exc}"})
            continue

        eligible, _ = svc.apply_hard_constraints(catalogue, prof)
        base["eligible_count"] = len(eligible)
        if eligible.empty:
            rows.append({**base, "rank": None, "Property_ID": None,
                         "status": "empty_result_constraints_not_relaxed"})
            continue

        direct_query = svc.build_direct_query(prof)
        direct = svc.score_direct(eligible, direct_query)
        ranked, nb = rank_candidates(direct, cfg, cpg if "graph" in cfg.components else None,
                                     market if "market" in cfg.components else None)
        top = ranked.head(TOP_N)
        eligible_ids = set(eligible["Property_ID"])
        violations += int((~top["Property_ID"].isin(eligible_ids)).sum())
        nb_by = {pid: g for pid, g in nb.groupby("Property_ID")} if not nb.empty else {}

        q_evidence = []
        for _, r in top.iterrows():
            pid = r["Property_ID"]
            graph_ok = r.get("graph_support_status") == G_AVAILABLE
            used = nb_by.get(pid)
            nb_ids = used["neighbour_Property_ID"].tolist() if used is not None else []
            row = {
                **base,
                "rank": int(r["cpg_rank"]),
                "Property_ID": int(pid),
                "final_score": _clean(r["final_score"]),
                "direct_score": _clean(r["direct_preference_score"]),
                "graph_support": _clean(r["graph_support"]) if graph_ok else None,
                "graph_support_status": r.get("graph_support_status"),
                "contribution_direct": _clean(r.get("contribution_direct")),
                "contribution_graph": _clean(r.get("contribution_graph")) if graph_ok else None,
                "qualified_neighbours": _clean(r.get("qualified_neighbours")),
                "neighbours_used": ";".join(str(int(x)) for x in nb_ids),
                "status": "ok" if len(eligible) >= TOP_N else "fewer_than_top_n_eligible",
            }
            row.update({c: _clean(info.loc[pid, c]) for c in DETAIL_COLS})
            rows.append(row)
            q_evidence.append({
                "rank": row["rank"], "Property_ID": row["Property_ID"],
                "final_score": row["final_score"], "direct_score": row["direct_score"],
                "graph_support": row["graph_support"], "graph_support_status": row["graph_support_status"],
                "direct_reasons": svc._direct_reasons(r, direct_query["weights"]),
                "neighbours": [] if used is None else [
                    {k: _clean(v) for k, v in n.items()} for n in used.to_dict(orient="records")],
            })
        evidence[qid] = {"profile": prof, "direct_weights": direct_query["weights"],
                         "eligible_count": len(eligible), "top": q_evidence}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{method}_top5_recommendations.csv"
    out.to_csv(out_csv, index=False)
    (OUT_DIR / f"{method}_top5_evidence.json").write_text(
        json.dumps(evidence, indent=1, default=str), encoding="utf-8")

    per_q = out.drop_duplicates("query_id")
    rec = out.dropna(subset=["Property_ID"])
    summary = {
        "method": method,
        "variant": args.variant,
        "ranking_config_id": cfg.config_id(),
        "graph_config_id": cpg.config_id if "graph" in cfg.components else None,
        "queries_file": args.queries.name,
        "queries": int(len(queries)),
        "status_counts_rows": out["status"].value_counts().to_dict(),
        "queries_by_role": per_q["benchmark_role"].value_counts().to_dict(),
        "recommendation_rows": int(len(rec)),
        "graph_support_available_in_top5_pct": (
            round(float((rec["graph_support_status"] == G_AVAILABLE).mean() * 100), 2)
            if "graph_support_status" in rec and len(rec) else None),
        "integrity": {
            "hard_constraint_violations_in_top5": violations,
            "invalid_queries": errors,
            "eligible_count_matches_manifest": bool(
                (per_q.set_index("query_id")["eligible_count"].astype("Int64")
                 == manifest["eligible_count"].reindex(per_q["query_id"]).astype("Int64")).all()),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (OUT_DIR / f"{method}_top5_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                          encoding="utf-8")
    print(f"{method.upper()} Top-{TOP_N}: {summary['queries']} queries, {summary['recommendation_rows']} "
          f"recommendation rows ({summary['elapsed_seconds']} s)")
    print(f"Row status: {summary['status_counts_rows']}")
    print(f"Graph support available in Top-5: {summary['graph_support_available_in_top5_pct']}%")
    print(f"Integrity: {summary['integrity']}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
