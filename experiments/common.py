"""Shared helpers for the ablation / robustness and efficiency experiments.

Everything goes through the same code as the web app and generate_top5.py
(recommender_service + cpg_ranker), so experiment results and the prototype cannot disagree.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "graph", "ranking", "baseline_models", "market_context"):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import recommender_service as svc  # noqa: E402
from build_comparable_graph import NODE_ATTRIBUTES, ComparableGraph  # noqa: E402
from cpg_ranker import G_AVAILABLE, RankingConfig, rank_candidates  # noqa: E402
from similarity_engine import COMPONENTS, SimilarityConfig, SimilarityEngine  # noqa: E402

DATA_PATH = svc.DATA_PATH
MARKET_PATH = svc.MARKET_PATH
QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
MANIFEST = ROOT / "benchmark" / "eligibility_manifest_v2.csv"
FROZEN_EDGES = ROOT / "outputs" / "comparable_property_graph" / "comparable_property_edges.csv"
TOP_N = 5


def load_catalogue() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df["Property_ID"].duplicated().any():
        raise ValueError("Property_ID must be unique.")
    return df


def load_market() -> pd.DataFrame | None:
    return pd.read_csv(MARKET_PATH) if MARKET_PATH.exists() else None


def similarity_config(drop=(), location_mode="coordinates", **overrides) -> SimilarityConfig:
    """Same construction rule as similarity_engine.py --drop-component (weights renormalise)."""
    drop = set(drop)
    active = tuple(c for c in COMPONENTS if c not in drop)
    if "location" in drop:
        location_mode = "none"
    mandatory = tuple(c for c in SimilarityConfig.mandatory_components if c in active) or active[:1]
    return SimilarityConfig(
        active_components=active, mandatory_components=mandatory, location_mode=location_mode,
        min_valid_components=min(SimilarityConfig.min_valid_components, len(active)), **overrides)


def build_graph(df: pd.DataFrame, cfg: SimilarityConfig) -> tuple[ComparableGraph, pd.DataFrame, dict]:
    """Similarity engine -> retained edges -> ComparableGraph (in memory, nothing overwritten)."""
    engine = SimilarityEngine(df, cfg)
    cand, stats = engine.candidate_neighbours()
    edges = engine.retain_edges(cand)
    nodes = df[["Property_ID", *[c for c in NODE_ATTRIBUTES if c in df.columns]]] \
        .sort_values("Property_ID").reset_index(drop=True)
    cols = ["source_Property_ID", "target_Property_ID", "combined_similarity", "config_id"]
    cpg = ComparableGraph.from_frames(nodes, edges if not edges.empty else pd.DataFrame(columns=cols))
    return cpg, edges, stats


@dataclass
class QueryContext:
    """Graph-independent part of a query: profile, eligible set, direct scores."""
    query_id: str
    partition: str
    difficulty: str
    role: str
    eligible_count: int
    direct: pd.DataFrame | None  # None = empty result (constraints not relaxed)


def load_queries() -> pd.DataFrame:
    q = pd.read_csv(QUERIES, dtype=str, keep_default_na=False)
    m = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)[["query_id", "benchmark_role"]]
    return q.merge(m, on="query_id", how="left")


def prepare_queries(catalogue: pd.DataFrame, queries: pd.DataFrame) -> list[QueryContext]:
    out = []
    for _, q in queries.iterrows():
        prof = svc.parse_profile(q.to_dict())
        eligible, _ = svc.apply_hard_constraints(catalogue, prof)
        direct = None
        if not eligible.empty:
            direct = svc.score_direct(eligible, svc.build_direct_query(prof))
        out.append(QueryContext(q["query_id"], q.get("partition", ""), q.get("difficulty", ""),
                                q.get("benchmark_role", ""), len(eligible), direct))
    return out


def rank_query(ctx: QueryContext, variant: str, cpg: ComparableGraph | None,
               market: pd.DataFrame | None, lambdas: dict | None = None) -> pd.DataFrame:
    cfg = RankingConfig(variant=variant, top_k=TOP_N) if lambdas is None else \
        RankingConfig(variant=variant, top_k=TOP_N, lambdas=lambdas)
    ranked, _ = rank_candidates(ctx.direct, cfg,
                                cpg if "graph" in cfg.components else None,
                                market if "market" in cfg.components else None)
    return ranked.head(TOP_N)
