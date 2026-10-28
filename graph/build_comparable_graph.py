"""Build the Comparable-Property Graph (CPG) from the similarity engine's retained edges.

This script does NOT compute similarity. All similarity rules live in
graph/similarity_engine.py so that the graph, neighbour retrieval, ranking and explanations
all read the same frozen edges and config_id.

Run from the project root:
    python graph/build_comparable_graph.py                 # use existing similarity output
    python graph/build_comparable_graph.py --rebuild       # rerun similarity engine first
    python graph/build_comparable_graph.py --rebuild --tau 0.8 --k 8

Inputs:
    dataset/Malaysia_Property_Cleaned_Raw.csv              node attributes (raw units)
    outputs/similarity_engine/comparable_edges.csv         retained edges + component evidence

Outputs (outputs/comparable_property_graph/):
    comparable_property_nodes.csv    one row per property incl. degree / isolated flag
    comparable_property_edges.csv    frozen edge list used by retrieval and ranking
    comparable_property_graph.gexf   for Gephi / NetworkX
    comparable_property_graph.png    visual sample
    graph_summary.json               size, density, isolation, components, config_id

Neighbour retrieval for ranking (thesis Eq. 4.16):
    from graph.build_comparable_graph import ComparableGraph
    cpg = ComparableGraph.load()
    cpg.get_neighbours(property_id, eligible_ids=candidate_ids)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

from similarity_engine import (  # noqa: E402
    COMPONENTS, SimilarityConfig, SimilarityEngine, OUTPUT_DIR as SIM_OUTPUT_DIR,
)

DATA_PATH = PROJECT_ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
EDGES_PATH = SIM_OUTPUT_DIR / "comparable_edges.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparable_property_graph"
NODES_OUT = OUTPUT_DIR / "comparable_property_nodes.csv"
EDGES_OUT = OUTPUT_DIR / "comparable_property_edges.csv"

NODE_ATTRIBUTES = [
    "property_type", "price", "built_up_area", "bedroom", "bathroom",
    "tenure", "furnishing", "area", "district", "state", "original_link",
]
SIM_COLUMNS = [f"sim_{c}" for c in COMPONENTS]
EDGE_EVIDENCE = SIM_COLUMNS + [
    "combined_similarity", "active_components", "omitted_components", "active_weight_sum",
    "location_resolution", "distance_km", "rank_in_source", "rank_in_target",
    "retention_rule", "config_id",
]


def _clean(value):
    """GEXF/JSON-safe scalar: NaN/None -> None, numpy -> python."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


class ComparableGraph:
    """Frozen CPG with deterministic neighbour retrieval."""

    def __init__(self, graph: nx.Graph, config_id: str | None):
        self.graph = graph
        self.config_id = config_id

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> "ComparableGraph":
        ids = set(nodes["Property_ID"])
        unknown = (set(edges["source_Property_ID"]) | set(edges["target_Property_ID"])) - ids
        if unknown:
            raise ValueError(f"{len(unknown)} edge endpoints are not in the catalogue "
                             f"(e.g. {sorted(unknown)[:5]}). Rebuild the similarity output.")
        config_ids = edges["config_id"].dropna().unique().tolist() if "config_id" in edges else []
        if len(config_ids) > 1:
            raise ValueError(f"Edges mix several similarity configurations: {config_ids}")

        g = nx.Graph(name="Comparable-Property Graph", config_id=config_ids[0] if config_ids else "")
        attr_cols = [c for c in NODE_ATTRIBUTES if c in nodes.columns]
        for row in nodes[["Property_ID", *attr_cols]].itertuples(index=False):
            pid, *vals = row
            g.add_node(pid, **{k: _clean(v) for k, v in zip(attr_cols, vals)})

        ev_cols = [c for c in EDGE_EVIDENCE if c in edges.columns]
        for row in edges[["source_Property_ID", "target_Property_ID", *ev_cols]].itertuples(index=False):
            u, v, *vals = row
            attrs = {k: _clean(x) for k, x in zip(ev_cols, vals)}
            for rk in ("rank_in_source", "rank_in_target"):
                if attrs.get(rk) is not None:
                    attrs[rk] = int(attrs[rk])
            attrs["weight"] = attrs["combined_similarity"]
            g.add_edge(u, v, **attrs)
        return cls(g, config_ids[0] if config_ids else None)

    @classmethod
    def load(cls, nodes_path: Path = NODES_OUT, edges_path: Path = EDGES_OUT) -> "ComparableGraph":
        """Load the frozen CPG written by this script (used at ranking time)."""
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError("Frozen CPG not found. Run: python graph/build_comparable_graph.py")
        return cls.from_frames(pd.read_csv(nodes_path), pd.read_csv(edges_path))

    # ---------------------------------------------------------------- retrieval
    def get_neighbours(self, property_id, eligible_ids=None, limit: int | None = None) -> list[dict]:
        """Retained neighbours of `property_id`, strongest first (ties: Property_ID asc).

        eligible_ids: the current candidate set C_q. When given, only neighbours that also
        satisfy the active hard constraints are returned (scenario-qualified neighbours,
        Eq. 4.16). An empty list means graph support is UNAVAILABLE, not zero.
        """
        if property_id not in self.graph:
            raise KeyError(f"Property_ID {property_id} is not in the graph.")
        allowed = None if eligible_ids is None else set(eligible_ids)
        out = []
        for nb, data in self.graph[property_id].items():
            if allowed is not None and nb not in allowed:
                continue
            rec = {"neighbour_Property_ID": nb, "similarity": data["combined_similarity"]}
            rec.update({k: data.get(k) for k in EDGE_EVIDENCE if k != "combined_similarity"})
            out.append(rec)
        out.sort(key=lambda r: (-r["similarity"], r["neighbour_Property_ID"]))
        return out[:limit] if limit is not None else out

    def neighbour_table(self, candidate_ids, eligible_ids=None) -> pd.DataFrame:
        """Long table of qualified neighbours for many candidates (for scoring/evidence store)."""
        eligible = set(candidate_ids) if eligible_ids is None else set(eligible_ids)
        rows = []
        for pid in candidate_ids:
            nbs = self.get_neighbours(pid, eligible_ids=eligible) if pid in self.graph else []
            if not nbs:
                rows.append({"Property_ID": pid, "neighbour_Property_ID": None,
                             "graph_support_status": "unavailable_no_qualified_neighbour"})
            for r in nbs:
                rows.append({"Property_ID": pid, **r, "graph_support_status": "available"})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- reporting
    def summary(self) -> dict:
        g = self.graph
        n, m = g.number_of_nodes(), g.number_of_edges()
        deg = pd.Series(dict(g.degree()), dtype=float)
        comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        non_trivial = [c for c in comps if c > 1]
        w = pd.Series([d["combined_similarity"] for _, _, d in g.edges(data=True)], dtype=float)
        res = pd.Series([d.get("location_resolution") for _, _, d in g.edges(data=True)]).value_counts()
        return {
            "config_id": self.config_id,
            "nodes": n,
            "edges": m,
            "density": round(2 * m / max(n * (n - 1), 1), 6),
            "isolated_nodes": int((deg == 0).sum()),
            "isolated_pct": round(float((deg == 0).mean() * 100), 2) if n else 0.0,
            "degree": {"mean": round(float(deg.mean()), 3), "median": float(deg.median()),
                       "max": int(deg.max()) if n else 0},
            "connected_components_excluding_isolated": len(non_trivial),
            "largest_component_size": comps[0] if comps else 0,
            "largest_component_pct": round(comps[0] / n * 100, 2) if n and comps else 0.0,
            "edge_similarity": {} if w.empty else {
                "min": round(float(w.min()), 4), "median": round(float(w.median()), 4),
                "max": round(float(w.max()), 4)},
            "edges_with_similarity_1": int((w >= 0.9999).sum()),
            "edge_location_resolution": {str(k): int(v) for k, v in res.items()},
        }


# -------------------------------------------------------------------- outputs
def write_gexf(g: nx.Graph, path: Path) -> None:
    """GEXF cannot store None; drop empty attributes on a copy."""
    h = nx.Graph(**{k: v for k, v in g.graph.items() if v is not None})
    for n, d in g.nodes(data=True):
        h.add_node(str(n), **{k: v for k, v in d.items() if v is not None})
    for u, v, d in g.edges(data=True):
        h.add_edge(str(u), str(v), **{k: v2 for k, v2 in d.items() if v2 is not None})
    nx.write_gexf(h, path)


def draw_sample(g: nx.Graph, path: Path, hubs: int = 45) -> bool:
    if g.number_of_edges() == 0:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG.")
        return False
    order = sorted(g.degree, key=lambda x: (-x[1], x[0]))
    selected = {node for node, _ in order[:hubs]}
    for node in list(selected):
        selected.update(sorted(g.neighbors(node))[:3])
    sample = g.subgraph(selected).copy()
    types = sorted({str(d.get("property_type")) for _, d in sample.nodes(data=True)})
    cmap = plt.get_cmap("tab10")
    colour = {t: cmap(i % 10) for i, t in enumerate(types)}
    plt.figure(figsize=(15, 11))
    pos = nx.spring_layout(sample, seed=42, weight="weight", k=0.65)
    widths = [0.5 + 3.0 * sample[u][v]["weight"] for u, v in sample.edges()]
    nx.draw_networkx_edges(sample, pos, alpha=0.25, width=widths, edge_color="#64748b")
    nx.draw_networkx_nodes(sample, pos, node_size=75, alpha=0.9,
                           node_color=[colour[str(d.get("property_type"))] for _, d in sample.nodes(data=True)])
    for t in types:
        plt.scatter([], [], color=colour[t], label=t)
    plt.legend(loc="lower left", fontsize=8, frameon=False)
    plt.title(f"Comparable-Property Graph sample ({g.graph.get('config_id', '')})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def run_similarity_engine(df: pd.DataFrame, tau: float, k: int) -> pd.DataFrame:
    cfg = SimilarityConfig(graph_threshold=tau, neighbour_limit=k)
    engine = SimilarityEngine(df, cfg)
    cand, _ = engine.candidate_neighbours()
    edges = engine.retain_edges(cand)
    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cand.to_csv(SIM_OUTPUT_DIR / "similarity_candidates.csv", index=False)
    edges.to_csv(EDGES_PATH, index=False)
    print(f"Similarity engine rerun: {cfg.config_id()} -> {len(edges):,} edges")
    return edges


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the frozen Comparable-Property Graph.")
    p.add_argument("--input", type=Path, default=DATA_PATH)
    p.add_argument("--edges", type=Path, default=EDGES_PATH)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--rebuild", action="store_true", help="Rerun the similarity engine first.")
    p.add_argument("--tau", type=float, default=SimilarityConfig.graph_threshold)
    p.add_argument("--k", type=int, default=SimilarityConfig.neighbour_limit)
    p.add_argument("--no-png", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    df = pd.read_csv(args.input)
    if df["Property_ID"].duplicated().any():
        raise ValueError("Property_ID must be unique before building the graph.")

    if args.rebuild or not args.edges.exists():
        if not args.rebuild:
            print(f"{args.edges} not found; running the similarity engine.")
        edges = run_similarity_engine(df, args.tau, args.k)
    else:
        edges = pd.read_csv(args.edges)

    nodes = df[["Property_ID", *[c for c in NODE_ATTRIBUTES if c in df.columns]]] \
        .sort_values("Property_ID").reset_index(drop=True)
    cpg = ComparableGraph.from_frames(nodes, edges)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    degree = dict(cpg.graph.degree())
    nodes_out = nodes.assign(neighbour_count=nodes["Property_ID"].map(degree).fillna(0).astype(int))
    nodes_out["isolated"] = nodes_out["neighbour_count"].eq(0)
    nodes_out.to_csv(out / NODES_OUT.name, index=False)
    edges.to_csv(out / EDGES_OUT.name, index=False)
    write_gexf(cpg.graph, out / "comparable_property_graph.gexf")
    png = False if args.no_png else draw_sample(cpg.graph, out / "comparable_property_graph.png")

    summary = cpg.summary()
    summary.update({"dataset_path": str(args.input), "edges_path": str(args.edges),
                    "build_seconds": round(time.perf_counter() - started, 3)})
    (out / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Comparable-property graph built.")
    print(f"Config: {summary['config_id']}")
    print(f"Nodes: {summary['nodes']:,} | edges: {summary['edges']:,} | density: {summary['density']}")
    print(f"Isolated: {summary['isolated_nodes']:,} ({summary['isolated_pct']}%) | "
          f"largest component: {summary['largest_component_size']:,} ({summary['largest_component_pct']}%)")
    print(f"Edges with similarity 1.0 (check duplicates): {summary['edges_with_similarity_1']:,}")
    print(f"Outputs: {out}" + ("" if png else " (no PNG)"))


if __name__ == "__main__":
    main()
