"""CPG scoring and deterministic ranking (thesis Sections 4.7.3 - 4.7.7).

Pipeline position:
    hard_constraint_filter.py  ->  direct_preference_baseline.py  ->  THIS SCRIPT

Run from the project root (after the filter and direct-preference steps):
    python ranking/cpg_ranker.py                          # full CPG: direct + graph
    python ranking/cpg_ranker.py --variant full_market    # + market context (separate variant)
    python ranking/cpg_ranker.py --variant direct_only    # ablation: no graph

For each eligible candidate i of query q:
    D_qi  direct preference score               (from direct_preference_baseline.py)
    N_qi  retained CPG neighbours that are ALSO in the candidate set C_q        (Eq. 4.16)
    G_qi  = sum_j S_ij D_qj / sum_j S_ij   over j in N_qi with a valid D_qj     (Eq. 4.17)
    M_i   market-context consistency score  (optional, only when enabled + valid)
    R_qi  = sum_x lambda_x X_qi^x / sum_x lambda_x   over AVAILABLE components  (Eq. 4.21)
    c_qi^x = lambda_x X_qi^x / sum_y lambda_y        (contribution, sum = R)   (Eq. 4.22-4.23)
    gamma_qij = S_ij D_qj / sum_l S_il D_ql          (neighbour share of G)     (Eq. 4.24)

Unavailable components are removed and weights renormalised; they are never scored 0.
Ranking: R desc, D desc, G desc, M desc, Property_ID asc (Section 4.7.7).

Outputs (outputs/cpg_ranking/):
    cpg_rankings.csv            one row per candidate: components, weights, contributions, R
    cpg_neighbour_evidence.csv  one row per (candidate, qualified neighbour) with edge evidence
    cpg_evidence_topK.json      nested evidence record per Top-K candidate (explanation input)
    run_summary.json            config, availability/fallback counts, timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "graph"))
sys.path.insert(0, str(PROJECT_ROOT / "market_context"))

from build_comparable_graph import ComparableGraph, EDGE_EVIDENCE  # noqa: E402
from calculate_market_context import STATUS_AVAILABLE  # noqa: E402

DIRECT_RANKINGS = PROJECT_ROOT / "outputs" / "direct_preference_baseline" / "direct_preference_rankings.csv"
MARKET_SCORES = PROJECT_ROOT / "outputs" / "market_context" / "market_context_scores.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cpg_ranking"

VARIANTS = {
    "full": ("direct", "graph"),                     # primary CPG (market context disabled)
    "full_market": ("direct", "graph", "market"),    # separately reported variant
    "direct_only": ("direct",),                      # ablation: content-based equivalent
    "direct_market": ("direct", "market"),           # ablation
}

# Graph-support status codes (stable strings for the explanation engine).
G_AVAILABLE = "available"
G_NOT_IN_GRAPH = "unavailable_candidate_not_in_graph"
G_ISOLATED = "unavailable_no_retained_neighbours"
G_NONE_ELIGIBLE = "unavailable_no_neighbour_meets_hard_constraints"
G_NO_DIRECT = "unavailable_neighbours_have_no_direct_score"
G_DISABLED = "disabled_by_variant"


def _default_lambdas() -> dict:
    return {"direct": 0.6, "graph": 0.4, "market": 0.2}


@dataclass
class RankingConfig:
    """Recommendation-score configuration to freeze before held-out evaluation."""

    variant: str = "full"
    lambdas: dict = field(default_factory=_default_lambdas)
    top_k: int = 10
    max_neighbours_used: int | None = None  # None = all qualified neighbours (<= k_G already)
    score_decimals: int = 10                # rounding before ranking for reproducible ties

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        if any(self.lambdas.get(c, 0) <= 0 for c in VARIANTS[self.variant]):
            raise ValueError("Every component in the variant needs a positive lambda.")

    @property
    def components(self) -> tuple:
        return VARIANTS[self.variant]

    def config_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return "RANK-" + hashlib.sha1(blob).hexdigest()[:10]


# ---------------------------------------------------------------------- graph support
def compute_graph_support(candidates: pd.DataFrame, cpg: ComparableGraph,
                          max_neighbours: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eq. 4.16, 4.17 and 4.24. Returns (per-candidate support, per-neighbour evidence)."""
    direct = candidates.set_index("Property_ID")["direct_preference_score"]
    eligible = set(direct.index)

    support_rows, evidence_rows = [], []
    for pid in direct.index:
        if pid not in cpg.graph:
            support_rows.append({"Property_ID": pid, "graph_support": np.nan,
                                 "graph_support_status": G_NOT_IN_GRAPH,
                                 "retained_neighbours": 0, "qualified_neighbours": 0,
                                 "neighbours_used": 0})
            continue
        retained = cpg.graph.degree(pid)
        qualified = cpg.get_neighbours(pid, eligible_ids=eligible, limit=max_neighbours)
        used = [n for n in qualified if pd.notna(direct.get(n["neighbour_Property_ID"]))]

        if retained == 0:
            status = G_ISOLATED
        elif not qualified:
            status = G_NONE_ELIGIBLE
        elif not used:
            status = G_NO_DIRECT
        else:
            status = G_AVAILABLE

        g = np.nan
        if status == G_AVAILABLE:
            s = np.array([n["similarity"] for n in used], dtype=float)
            d = np.array([direct[n["neighbour_Property_ID"]] for n in used], dtype=float)
            g = float((s * d).sum() / s.sum())
            sd_total = float((s * d).sum())
            for n, s_ij, d_j in zip(used, s, d):
                rec = {
                    "Property_ID": pid,
                    "neighbour_Property_ID": n["neighbour_Property_ID"],
                    "edge_similarity": float(s_ij),
                    "neighbour_direct_score": float(d_j),
                    "weighted_support": float(s_ij * d_j),
                    # gamma: share of neighbour-support evidence (Eq. 4.24), not a % of R.
                    "neighbour_support_share": (float(s_ij * d_j / sd_total) if sd_total > 0 else None),
                    "display_eligible": bool(sd_total > 0 and s_ij * d_j > 0),
                }
                rec.update({k: n.get(k) for k in EDGE_EVIDENCE if k not in ("combined_similarity",)})
                evidence_rows.append(rec)

        support_rows.append({"Property_ID": pid, "graph_support": g, "graph_support_status": status,
                             "retained_neighbours": int(retained), "qualified_neighbours": len(qualified),
                             "neighbours_used": len(used)})

    support = pd.DataFrame(support_rows)
    evidence = pd.DataFrame(evidence_rows)
    if not evidence.empty:
        evidence = evidence.sort_values(["Property_ID", "neighbour_support_share", "neighbour_Property_ID"],
                                        ascending=[True, False, True], kind="mergesort")
        evidence.insert(1, "neighbour_display_rank", evidence.groupby("Property_ID").cumcount() + 1)
    return support, evidence


# ---------------------------------------------------------------------- final score
def combine_scores(df: pd.DataFrame, cfg: RankingConfig) -> pd.DataFrame:
    """Eq. 4.21 - 4.23 with renormalisation over available components, then Section 4.7.7."""
    values = {
        "direct": df["direct_preference_score"],
        "graph": df["graph_support"],
        "market": df["market_context_score"],
    }
    out = df.copy()
    denom = pd.Series(0.0, index=df.index)
    avail = {}
    for comp in ("direct", "graph", "market"):
        ok = values[comp].notna() & (comp in cfg.components)
        avail[comp] = ok
        denom += np.where(ok, cfg.lambdas[comp], 0.0)

    active, omitted = [], []
    for comp in ("direct", "graph", "market"):
        lam = cfg.lambdas[comp]
        ok = avail[comp]
        out[f"lambda_{comp}"] = np.where(ok, lam, np.nan)
        out[f"weight_{comp}_normalised"] = np.where(ok & (denom > 0), lam / denom.where(denom > 0), np.nan)
        out[f"contribution_{comp}"] = np.where(ok & (denom > 0),
                                               lam * values[comp].fillna(0) / denom.where(denom > 0), np.nan)
        active.append(np.where(ok, comp, ""))
        in_variant = comp in cfg.components
        omitted.append(np.where(in_variant & ~ok, comp, ""))

    out["active_score_components"] = ["|".join(x for x in row if x) for row in zip(*active)]
    out["omitted_score_components"] = ["|".join(x for x in row if x) for row in zip(*omitted)]
    contrib = out[[f"contribution_{c}" for c in ("direct", "graph", "market")]]
    out["final_score"] = np.where(denom > 0, contrib.sum(axis=1, min_count=1), np.nan)
    out["final_score"] = out["final_score"].round(cfg.score_decimals)
    out["score_status"] = np.where(denom > 0, "scored", "unscored_no_available_component")
    out["fallback_direct_only"] = ("graph" in cfg.components) & ~avail["graph"] & avail["direct"]

    # Deterministic ranking (4.7.7). NaN sorts last via fill with -inf.
    key = pd.DataFrame({
        "R": out["final_score"].fillna(-np.inf),
        "D": out["direct_preference_score"].round(cfg.score_decimals).fillna(-np.inf),
        "G": (out["graph_support"].where(avail["graph"]).round(cfg.score_decimals).fillna(-np.inf)),
        "M": (out["market_context_score"].where(avail["market"]).round(cfg.score_decimals).fillna(-np.inf)),
        "pid": out["Property_ID"],
    }, index=out.index)
    order = key.sort_values(["R", "D", "G", "M", "pid"], ascending=[False, False, False, False, True],
                            kind="mergesort").index
    out = out.loc[order].reset_index(drop=True)
    out.insert(0, "cpg_rank", range(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------- evidence records
def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, str):
        return v
    return None if pd.isna(v) else v


def build_evidence(ranked: pd.DataFrame, evidence: pd.DataFrame, cfg: RankingConfig,
                   graph_config_id: str | None, query_id: str | None) -> list[dict]:
    """Nested evidence record per Top-K candidate: the explanation engine's only fact source."""
    direct_cols = [c for c in ranked.columns if c.startswith("score_") and c != "score_status"]
    nb = evidence.groupby("Property_ID") if not evidence.empty else None
    records = []
    for _, r in ranked.head(cfg.top_k).iterrows():
        pid = r["Property_ID"]
        direct_matches = []
        for c in direct_cols:
            name = c[len("score_"):]
            ev = r.get(f"evidence_{name}")
            try:
                ev = json.loads(ev) if isinstance(ev, str) and ev.startswith("{") else ev
            except json.JSONDecodeError:
                pass
            direct_matches.append({"preference": name, "match": _clean(r[c]), "evidence": _clean(ev)
                                   if not isinstance(ev, dict) else ev})
        neighbours = []
        if nb is not None and pid in nb.groups:
            for _, n in nb.get_group(pid).iterrows():
                neighbours.append({k: _clean(n[k]) for k in n.index if k != "Property_ID"})
        records.append({
            "query_id": query_id,
            "cpg_rank": int(r["cpg_rank"]),
            "Property_ID": _clean(pid),
            "final_score": _clean(r["final_score"]),
            "score_status": r["score_status"],
            "components": {
                comp: {
                    "value": _clean(r[{"direct": "direct_preference_score", "graph": "graph_support",
                                       "market": "market_context_score"}[comp]]),
                    "lambda": _clean(r[f"lambda_{comp}"]),
                    "normalised_weight": _clean(r[f"weight_{comp}_normalised"]),
                    "contribution": _clean(r[f"contribution_{comp}"]),
                } for comp in ("direct", "graph", "market") if comp in cfg.components
            },
            "active_score_components": r["active_score_components"].split("|") if r["active_score_components"] else [],
            "omitted_score_components": r["omitted_score_components"].split("|") if r["omitted_score_components"] else [],
            "direct_preference_matches": direct_matches,
            "graph_support": {
                "status": r["graph_support_status"],
                "retained_neighbours": _clean(r["retained_neighbours"]),
                "qualified_neighbours": _clean(r["qualified_neighbours"]),
                "neighbours_used": _clean(r["neighbours_used"]),
                "fallback_direct_only": bool(r["fallback_direct_only"]),
                "neighbours": neighbours,
            },
            "market_context": {
                "enabled": "market" in cfg.components,
                "status": r.get("market_context_status"),
                "unavailable_reason": _clean(r.get("market_context_unavailable_reason")),
                "asking_ppsf": _clean(r.get("asking_ppsf")),
                "reference_mean": _clean(r.get("ref_ppsf_mean")),
                "reference_q1": _clean(r.get("ref_ppsf_q1")),
                "reference_q3": _clean(r.get("ref_ppsf_q3")),
                "position_vs_reference": _clean(r.get("ppsf_position_vs_reference")),
                "z": _clean(r.get("market_z")),
            },
            "config_ids": {
                "ranking": cfg.config_id(),
                "graph": graph_config_id,
                "market_context": _clean(r.get("market_context_config_id")),
            },
        })
    return records


# ---------------------------------------------------------------------- orchestration
def rank_candidates(direct_df: pd.DataFrame, cfg: RankingConfig, cpg: ComparableGraph | None = None,
                    market_df: pd.DataFrame | None = None):
    """Library entry point: returns (ranked, neighbour_evidence)."""
    if "Property_ID" not in direct_df or "direct_preference_score" not in direct_df:
        raise ValueError("Direct-preference output needs Property_ID and direct_preference_score.")
    if direct_df["Property_ID"].duplicated().any():
        raise ValueError("Candidate set contains duplicate Property_IDs.")

    # Avoid name clashes with the baseline's own columns.
    base = direct_df.drop(columns=[c for c in ("rank",) if c in direct_df]).rename(
        columns={"active_components": "direct_active_components", "active_weights": "direct_active_weights",
                 "score_status": "direct_score_status"})

    if "graph" in cfg.components:
        if cpg is None:
            cpg = ComparableGraph.load()
        support, evidence = compute_graph_support(base, cpg, cfg.max_neighbours_used)
    else:
        support = pd.DataFrame({"Property_ID": base["Property_ID"], "graph_support": np.nan,
                                "graph_support_status": G_DISABLED, "retained_neighbours": np.nan,
                                "qualified_neighbours": np.nan, "neighbours_used": np.nan})
        evidence = pd.DataFrame()
    df = base.merge(support, on="Property_ID", how="left", validate="one_to_one")

    mc_cols = ["market_context_score", "market_context_status", "market_context_unavailable_reason",
               "asking_ppsf", "market_z", "ref_ppsf_mean", "ref_ppsf_q1", "ref_ppsf_q3",
               "ppsf_position_vs_reference", "market_context_config_id"]
    if market_df is not None:
        mc = market_df[["Property_ID", *[c for c in mc_cols if c in market_df]]]
        df = df.merge(mc, on="Property_ID", how="left", validate="one_to_one")
        df.loc[df["market_context_status"].ne(STATUS_AVAILABLE), "market_context_score"] = np.nan
        df["market_context_status"] = df["market_context_status"].fillna("unavailable_not_computed")
    else:
        df["market_context_score"] = np.nan
        df["market_context_status"] = "disabled_by_variant" if "market" not in cfg.components \
            else "unavailable_not_computed"
    if "market" not in cfg.components:
        df["market_context_score"] = np.nan
        df["market_context_status"] = "disabled_by_variant"

    ranked = combine_scores(df, cfg)
    return ranked, evidence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPG graph support, final score and deterministic ranking.")
    p.add_argument("--direct", type=Path, default=DIRECT_RANKINGS,
                   help="direct_preference_rankings.csv for ONE query (eligible candidates only).")
    p.add_argument("--market", type=Path, default=MARKET_SCORES)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--variant", choices=list(VARIANTS), default="full")
    p.add_argument("--lambda-direct", type=float, default=_default_lambdas()["direct"])
    p.add_argument("--lambda-graph", type=float, default=_default_lambdas()["graph"])
    p.add_argument("--lambda-market", type=float, default=_default_lambdas()["market"])
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not args.direct.exists():
        raise FileNotFoundError(f"Direct-preference rankings not found: {args.direct}\n"
                                "Run hard_constraint_filter.py and direct_preference_baseline.py first.")
    cfg = RankingConfig(variant=args.variant, top_k=args.top_k,
                        lambdas={"direct": args.lambda_direct, "graph": args.lambda_graph,
                                 "market": args.lambda_market})
    direct_df = pd.read_csv(args.direct)
    query_id = str(direct_df["query_id"].iloc[0]) if "query_id" in direct_df and len(direct_df) else None

    market_df = None
    if "market" in cfg.components:
        if not args.market.exists():
            raise FileNotFoundError(f"Market-context scores not found: {args.market}\n"
                                    "Run market_context/calculate_market_context.py first.")
        market_df = pd.read_csv(args.market)

    cpg = ComparableGraph.load() if "graph" in cfg.components else None
    t0 = time.perf_counter()
    ranked, evidence = rank_candidates(direct_df, cfg, cpg, market_df)
    rank_seconds = time.perf_counter() - t0

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lead = ["cpg_rank", "Property_ID", "final_score", "score_status", "active_score_components",
            "omitted_score_components", "direct_preference_score", "graph_support", "graph_support_status",
            "market_context_score", "market_context_status",
            *[col for c in ("direct", "graph", "market")
              for col in (f"lambda_{c}", f"weight_{c}_normalised", f"contribution_{c}")],
            "retained_neighbours", "qualified_neighbours", "neighbours_used", "fallback_direct_only"]
    lead = [c for c in lead if c in ranked.columns]
    ranked = ranked[lead + [c for c in ranked.columns if c not in lead]]
    ranked.insert(1, "ranking_config_id", cfg.config_id())
    ranked.insert(2, "variant", cfg.variant)
    ranked.to_csv(out / "cpg_rankings.csv", index=False)
    evidence.to_csv(out / "cpg_neighbour_evidence.csv", index=False)
    records = build_evidence(ranked, evidence, cfg, cpg.config_id if cpg else None, query_id)
    (out / f"cpg_evidence_top{cfg.top_k}.json").write_text(json.dumps(records, indent=2, default=str),
                                                          encoding="utf-8")

    status_counts = ranked["graph_support_status"].value_counts().to_dict()
    summary = {
        "query_id": query_id,
        "variant": cfg.variant,
        "ranking_config_id": cfg.config_id(),
        "config": asdict(cfg),
        "graph_config_id": cpg.config_id if cpg else None,
        "candidates": int(len(ranked)),
        "scored": int(ranked["score_status"].eq("scored").sum()),
        "graph_support_status": status_counts,
        "graph_support_available_pct": round(float(ranked["graph_support_status"].eq(G_AVAILABLE).mean() * 100), 2)
        if len(ranked) else 0.0,
        "fallback_direct_only": int(ranked["fallback_direct_only"].sum()),
        "market_context_status": ranked["market_context_status"].value_counts().to_dict(),
        "top_k_ids": ranked["Property_ID"].head(cfg.top_k).tolist(),
        "ranking_seconds": round(rank_seconds, 4),
        "total_seconds": round(time.perf_counter() - started, 4),
        "integrity_checks": {
            "contributions_sum_to_final_score": bool(np.allclose(
                ranked[[c for c in ranked if c.startswith("contribution_")]].sum(axis=1, min_count=1)
                .fillna(-1).round(8), ranked["final_score"].fillna(-1).round(8))),
            "ranks_unique": bool(ranked["cpg_rank"].is_unique),
            "neighbours_all_in_candidate_set": bool(evidence.empty or evidence["neighbour_Property_ID"]
                                                    .isin(set(ranked["Property_ID"])).all()),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"CPG ranking completed ({cfg.variant}, {cfg.config_id()}).")
    print(f"Candidates: {summary['candidates']:,} | graph support available: "
          f"{summary['graph_support_available_pct']}% | direct-only fallback: {summary['fallback_direct_only']:,}")
    for k, v in status_counts.items():
        print(f"  - {k}: {v}")
    print(f"Integrity: {summary['integrity_checks']}")
    print(f"Top {cfg.top_k}: {summary['top_k_ids']}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
