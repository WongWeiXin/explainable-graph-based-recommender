"""Property-property similarity engine for the Comparable-Property Graph (thesis 4.6.2 - 4.6.6).

Run from the project root:
    python graph/similarity_engine.py
    python graph/similarity_engine.py --tau 0.75 --k 10
    python graph/similarity_engine.py --drop-component location          # ablation (RQ3)
    python graph/similarity_engine.py --location-mode district_only      # reduced location precision
    python graph/similarity_engine.py --round-coords 2                   # rounded coordinates (~1.1 km)
    python graph/similarity_engine.py --pair 0 1                         # audit one pair

Component similarities (all in [0, 1], symmetric):
    location       exp(-d_ij / sigma_loc) with Haversine d_ij (Eq. 4.1-4.3);
                   district fallback delta if same district else 0 (Eq. 4.4)
    price          1 - |p_i - p_j| / max(p_i, p_j, eps)                (Eq. 4.5)
    size           1 - |a_i - a_j| / max(a_i, a_j, eps)                (Eq. 4.6)
    property_type  1 if same type else 0                               (Eq. 4.7)
    bedroom        1 - |b_i - b_j| / max(b_i, b_j, 1)                  (Eq. 4.8)
    bathroom       1 - |h_i - h_j| / max(h_i, h_j, 1)                  (Eq. 4.9)
    facility       cosine of nearby-facility count vectors             (Eq. 4.10)

Combined similarity over VALID components only (Eq. 4.11):
    S_ij = sum_{f in A_ij} w_f s_f / sum_{f in A_ij} w_f
A missing component is removed from A_ij, never scored as 0 (missing != dissimilar).

Edge eligibility: minimum-evidence predicate AND S_ij >= tau (Eq. 4.12).
Retention: each node ranks eligible neighbours by (S desc, Property_ID asc) and keeps at
most k_G; under mutual Top-k an edge survives only if both endpoints select each other.

Outputs (outputs/similarity_engine/):
    similarity_candidates.csv   directed Top-k candidates per node with full component record
    comparable_edges.csv        retained undirected edges (input to graph construction)
    node_neighbour_counts.csv   neighbour count per property (isolated-node audit)
    similarity_report.json      frozen config + efficiency + graph-density diagnostics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "similarity_engine"

EARTH_RADIUS_KM = 6371.0
COMPONENTS = ("location", "price", "size", "property_type", "bedroom", "bathroom", "facility")

FACILITY_COLUMNS = [
    "train_station_count_3km", "shopping_mall_count_3km", "supermarket_count_3km",
    "cafe_count_3km", "school_count_3km", "hospital_count_3km", "pharmacy_count_3km",
    "park_count_1km", "office_count_3km", "university_count_3km",
]

LOC_COORD, LOC_DISTRICT, LOC_NONE = "coordinates", "district", "unavailable"


def _default_weights() -> dict:
    return {
        "location": 0.25,
        "price": 0.20,
        "size": 0.15,
        "property_type": 0.15,
        "bedroom": 0.075,
        "bathroom": 0.075,
        "facility": 0.10,
    }


@dataclass
class SimilarityConfig:
    """Graph configuration to freeze before held-out evaluation."""

    weights: dict = field(default_factory=_default_weights)
    active_components: tuple = COMPONENTS
    # Minimum-evidence predicate: >=1 of these must be valid, plus min_valid_components overall.
    mandatory_components: tuple = ("location", "price", "size")
    min_valid_components: int = 3
    graph_threshold: float = 0.75          # tau
    neighbour_limit: int = 10              # k_G
    retention_rule: str = "mutual_topk"    # or "union_topk" (sensitivity only)
    # Location
    location_mode: str = "coordinates"     # coordinates | district_only | none
    sigma_loc_km: float = 5.0
    district_fallback_value: float = 0.5   # delta
    coordinate_round_decimals: int | None = None
    # Validity rules (raw units: RM, sq ft)
    min_built_up_area: float = 150.0
    max_built_up_area: float = 50_000.0
    max_room_count: float = 20.0
    # Facility profile
    facility_columns: tuple = tuple(FACILITY_COLUMNS)
    facility_scaling: str = "p95"          # p95 | none  (p95 stops cafe/office counts dominating)
    eps: float = 1e-9
    score_decimals: int = 10               # rounding before threshold/ranking for deterministic ties

    def __post_init__(self):
        unknown = set(self.active_components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown similarity components: {sorted(unknown)}")
        if self.location_mode not in {"coordinates", "district_only", "none"}:
            raise ValueError("location_mode must be coordinates | district_only | none")
        if self.retention_rule not in {"mutual_topk", "union_topk"}:
            raise ValueError("retention_rule must be mutual_topk | union_topk")
        if self.facility_scaling not in {"p95", "none"}:
            raise ValueError("facility_scaling must be p95 | none")
        if not any(c in self.active_components for c in self.mandatory_components):
            raise ValueError("At least one mandatory component must be active.")
        if sum(self.weights.get(c, 0) for c in self.active_components) <= 0:
            raise ValueError("Active component weights must sum to a positive value.")

    def config_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=list).encode("utf-8")
        return "SIM-" + hashlib.sha1(blob).hexdigest()[:10]


def _norm_text(series: pd.Series) -> np.ndarray:
    s = series.astype("string").str.strip().str.casefold()
    s = s.mask(s.isin(["", "nan", "none", "unknown", "-1", "-1.0"]))
    return s.to_numpy(dtype=object, na_value=None)


def _relative_similarity(x: np.ndarray, xi: float, floor: float) -> np.ndarray:
    """1 - |x - xi| / max(x, xi, floor), bounded to [0, 1]."""
    return np.clip(1.0 - np.abs(x - xi) / np.maximum(np.maximum(x, xi), floor), 0.0, 1.0)


class SimilarityEngine:
    """Vectorised, auditable pairwise similarity over a frozen property catalogue."""

    def __init__(self, df: pd.DataFrame, config: SimilarityConfig | None = None):
        self.cfg = config or SimilarityConfig()
        if "Property_ID" not in df.columns:
            raise ValueError("Property_ID column is required.")
        if df["Property_ID"].duplicated().any():
            raise ValueError("Property_ID must be unique (run duplicate handling first).")

        # Deterministic node order: ascending Property_ID. Position == tie-break rank.
        self.df = df.sort_values("Property_ID", kind="mergesort").reset_index(drop=True)
        self.n = len(self.df)
        self.ids = self.df["Property_ID"].to_numpy()
        self._pos = {pid: k for k, pid in enumerate(self.ids)}
        self._prepare()

    # ------------------------------------------------------------------ preparation
    def _num(self, col: str) -> np.ndarray:
        if col not in self.df.columns:
            return np.full(self.n, np.nan)
        return pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)

    def _prepare(self) -> None:
        c = self.cfg
        price = self._num("price")
        self.price_valid = np.isfinite(price) & (price > 0)
        self.price = np.where(self.price_valid, price, np.nan)

        size = self._num("built_up_area")
        self.size_valid = np.isfinite(size) & (size >= c.min_built_up_area) & (size <= c.max_built_up_area)
        self.size = np.where(self.size_valid, size, np.nan)

        bed, bath = self._num("bedroom"), self._num("bathroom")
        self.bed_valid = np.isfinite(bed) & (bed >= 0) & (bed <= c.max_room_count)
        self.bath_valid = np.isfinite(bath) & (bath >= 0) & (bath <= c.max_room_count)
        self.bed, self.bath = np.where(self.bed_valid, bed, np.nan), np.where(self.bath_valid, bath, np.nan)

        ptype = _norm_text(self.df["property_type"]) if "property_type" in self.df else np.full(self.n, None, object)
        self.type_valid = np.array([v is not None for v in ptype])
        self.type_code = pd.factorize(pd.Series(ptype, dtype=object), use_na_sentinel=True)[0]

        district = _norm_text(self.df["district"]) if "district" in self.df else np.full(self.n, None, object)
        self.district_valid = np.array([v is not None for v in district])
        self.district_code = pd.factorize(pd.Series(district, dtype=object), use_na_sentinel=True)[0]

        lat, lon = self._num("latitude"), self._num("longitude")
        if c.coordinate_round_decimals is not None:
            lat, lon = np.round(lat, c.coordinate_round_decimals), np.round(lon, c.coordinate_round_decimals)
        self.coord_valid = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
                            & ~((lat == 0) & (lon == 0)))
        self.lat_r, self.lon_r = np.radians(lat), np.radians(lon)
        self.cos_lat = np.cos(self.lat_r)

        fac_cols = [f for f in c.facility_columns if f in self.df.columns]
        self.facility_columns_used = fac_cols
        if fac_cols:
            F = np.column_stack([self._num(f) for f in fac_cols])
            F = np.where(np.isfinite(F) & (F >= 0), F, np.nan)
            row_ok = ~np.isnan(F).any(axis=1)
            F = np.nan_to_num(F, nan=0.0)
            if c.facility_scaling == "p95":
                scale = np.maximum(np.nanpercentile(F, 95, axis=0), 1.0)
                F = F / scale
                self.facility_scale = dict(zip(fac_cols, np.round(scale, 4).tolist()))
            else:
                self.facility_scale = {f: 1.0 for f in fac_cols}
            norms = np.linalg.norm(F, axis=1)
            self.fac_valid = row_ok & (norms > 0)          # zero vector -> unavailable (4.6.5)
            self.F_unit = np.where(self.fac_valid[:, None], F / np.maximum(norms, c.eps)[:, None], 0.0)
        else:
            self.facility_scale = {}
            self.fac_valid = np.zeros(self.n, dtype=bool)
            self.F_unit = np.zeros((self.n, 1))

        act = set(c.active_components)
        self.w = {k: float(c.weights.get(k, 0.0)) if k in act else 0.0 for k in COMPONENTS}

    # ------------------------------------------------------------------ components
    def _location(self, i: int):
        """Return (similarity, valid, resolution_code, distance_km). code 1=coord, 2=district, 0=none."""
        c, n = self.cfg, self.n
        sim, dist = np.zeros(n), np.full(n, np.nan)
        res = np.zeros(n, dtype=np.int8)
        if c.location_mode == "none":
            return sim, np.zeros(n, bool), res, dist

        if c.location_mode == "coordinates" and self.coord_valid[i]:
            both = self.coord_valid
            dlat = self.lat_r - self.lat_r[i]
            dlon = self.lon_r - self.lon_r[i]
            a = np.sin(dlat / 2) ** 2 + self.cos_lat[i] * self.cos_lat * np.sin(dlon / 2) ** 2
            a = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)
            d = 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            sim = np.where(both, np.exp(-d / c.sigma_loc_km), 0.0)
            dist = np.where(both, d, np.nan)
            res[both] = 1
        else:
            both = np.zeros(n, bool)

        # District fallback for pairs without two valid coordinates.
        need = ~both
        if self.district_valid[i]:
            fb = need & self.district_valid
            same = self.district_code == self.district_code[i]
            sim = np.where(fb, np.where(same, c.district_fallback_value, 0.0), sim)
            res[fb] = 2
        valid = res > 0
        return sim, valid, res, dist

    def components_for(self, i: int) -> dict:
        """All component similarities of node i against every node (vectorised)."""
        out = {}
        loc, loc_ok, res, dist = self._location(i)
        out["location"] = (loc, loc_ok)
        out["_resolution"], out["_distance_km"] = res, dist

        def pairwise(valid_vec, values, floor):
            ok = valid_vec & valid_vec[i]
            s = _relative_similarity(values, values[i], floor) if valid_vec[i] else np.zeros(self.n)
            return np.where(ok, np.nan_to_num(s), 0.0), ok

        out["price"] = pairwise(self.price_valid, self.price, self.cfg.eps)
        out["size"] = pairwise(self.size_valid, self.size, self.cfg.eps)
        out["bedroom"] = pairwise(self.bed_valid, self.bed, 1.0)
        out["bathroom"] = pairwise(self.bath_valid, self.bath, 1.0)

        t_ok = self.type_valid & self.type_valid[i]
        out["property_type"] = ((self.type_code == self.type_code[i]).astype(float) * t_ok, t_ok)

        f_ok = self.fac_valid & self.fac_valid[i]
        f_sim = np.clip(self.F_unit @ self.F_unit[i], 0.0, 1.0)
        out["facility"] = (np.where(f_ok, f_sim, 0.0), f_ok)
        return out

    def combine(self, comps: dict):
        """Eq. 4.11 over valid active components + minimum-evidence predicate."""
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        n_valid = np.zeros(self.n, dtype=np.int16)
        mandatory_ok = np.zeros(self.n, dtype=bool)
        for name in COMPONENTS:
            w = self.w[name]
            if w <= 0:
                continue
            s, ok = comps[name]
            num += w * s * ok
            den += w * ok
            n_valid += ok
            if name in self.cfg.mandatory_components:
                mandatory_ok |= ok
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.where(den > 0, num / den, np.nan)
        S = np.round(S, self.cfg.score_decimals)
        min_evidence = (den > 0) & mandatory_ok & (n_valid >= self.cfg.min_valid_components)
        return S, min_evidence, den, n_valid

    # ------------------------------------------------------------------ single pair (audit)
    def pair(self, id_i, id_j) -> dict:
        """Full evidence record for one pair; used by the explanation/audit layer."""
        i, j = self._pos[id_i], self._pos[id_j]
        comps = self.components_for(i)
        S, ok, den, n_valid = self.combine(comps)
        return self._record(i, j, comps, S, ok, den)

    def _record(self, i, j, comps, S, ok, den) -> dict:
        res_name = {1: LOC_COORD, 2: LOC_DISTRICT, 0: LOC_NONE}[int(comps["_resolution"][j])]
        rec = {"source_Property_ID": self.ids[i], "target_Property_ID": self.ids[j]}
        active, omitted = [], []
        for name in COMPONENTS:
            s, valid = comps[name]
            if self.w[name] <= 0:
                rec[f"sim_{name}"] = None
                continue
            if valid[j]:
                rec[f"sim_{name}"] = round(float(s[j]), 6)
                active.append(name)
            else:
                rec[f"sim_{name}"] = None
                omitted.append(name)
        d = comps["_distance_km"][j]
        rec.update({
            "combined_similarity": None if np.isnan(S[j]) else float(S[j]),
            "active_components": "|".join(active),
            "omitted_components": "|".join(omitted),
            "active_weight_sum": round(float(den[j]), 6),
            "min_evidence_ok": bool(ok[j]),
            "passes_threshold": bool(ok[j] and not np.isnan(S[j]) and S[j] >= self.cfg.graph_threshold),
            "location_resolution": res_name,
            "distance_km": None if np.isnan(d) else round(float(d), 4),
        })
        return rec

    # ------------------------------------------------------------------ all nodes
    def candidate_neighbours(self):
        """Directed Top-k_G eligible neighbours for every node + pairwise statistics."""
        c = self.cfg
        rows = []
        stats = {
            "unique_pairs_evaluated": self.n * (self.n - 1) // 2,
            "pairs_min_evidence_ok": 0,
            "pairs_at_or_above_threshold": 0,
            "component_valid_pairs": {k: 0 for k in COMPONENTS if self.w[k] > 0},
            "location_resolution_pairs": {LOC_COORD: 0, LOC_DISTRICT: 0, LOC_NONE: 0},
        }
        for i in range(self.n):
            comps = self.components_for(i)
            S, ok, den, _ = self.combine(comps)
            ok[i] = False
            eligible = ok & (S >= c.graph_threshold)

            upper = slice(i + 1, None)  # count each unordered pair once
            stats["pairs_min_evidence_ok"] += int(ok[upper].sum())
            stats["pairs_at_or_above_threshold"] += int(eligible[upper].sum())
            for k in stats["component_valid_pairs"]:
                stats["component_valid_pairs"][k] += int(comps[k][1][upper].sum())
            r = comps["_resolution"][upper]
            stats["location_resolution_pairs"][LOC_COORD] += int((r == 1).sum())
            stats["location_resolution_pairs"][LOC_DISTRICT] += int((r == 2).sum())
            stats["location_resolution_pairs"][LOC_NONE] += int((r == 0).sum())

            idx = np.flatnonzero(eligible)
            if idx.size == 0:
                continue
            # Deterministic: similarity desc, then Property_ID asc (array position).
            order = np.lexsort((idx, -S[idx]))[: c.neighbour_limit]
            for rank, j in enumerate(idx[order], start=1):
                rec = self._record(i, j, comps, S, ok, den)
                rec["neighbour_rank"] = rank
                rows.append(rec)
        cand = pd.DataFrame(rows)
        return cand, stats

    def retain_edges(self, cand: pd.DataFrame) -> pd.DataFrame:
        """Mutual (or union) Top-k retention -> undirected edge list, one row per pair."""
        if cand.empty:
            return cand
        pos = self._pos
        src_pos = cand["source_Property_ID"].map(pos).to_numpy()
        tgt_pos = cand["target_Property_ID"].map(pos).to_numpy()
        lo, hi = np.minimum(src_pos, tgt_pos), np.maximum(src_pos, tgt_pos)
        c = cand.assign(_lo=lo, _hi=hi, _fwd=src_pos == lo)

        # Rank each endpoint gave the other (NaN if it did not select it).
        fwd = c[c["_fwd"]].set_index(["_lo", "_hi"])["neighbour_rank"].rename("rank_in_source")
        rev = c[~c["_fwd"]].set_index(["_lo", "_hi"])["neighbour_rank"].rename("rank_in_target")
        ranks = pd.concat([fwd, rev], axis=1)
        if self.cfg.retention_rule == "mutual_topk":
            ranks = ranks.dropna()

        # Component record is symmetric; take one direction and orient lower ID -> higher ID.
        rec = c.sort_values("_fwd", ascending=False, kind="mergesort").drop_duplicates(["_lo", "_hi"])
        rec = rec.set_index(["_lo", "_hi"]).drop(columns=["neighbour_rank", "_fwd"])
        rec = rec.join(ranks, how="inner").reset_index()
        rec["source_Property_ID"] = self.ids[rec["_lo"].to_numpy()]
        rec["target_Property_ID"] = self.ids[rec["_hi"].to_numpy()]
        rec = rec.drop(columns=["_lo", "_hi"])
        rec["retention_rule"] = self.cfg.retention_rule
        rec["config_id"] = self.cfg.config_id()
        return rec.sort_values(["combined_similarity", "source_Property_ID", "target_Property_ID"],
                               ascending=[False, True, True], kind="mergesort").reset_index(drop=True)

    def neighbour_counts(self, edges: pd.DataFrame) -> pd.DataFrame:
        deg = pd.concat([edges["source_Property_ID"], edges["target_Property_ID"]]).value_counts() \
            if not edges.empty else pd.Series(dtype=int)
        out = pd.DataFrame({"Property_ID": self.ids})
        out["neighbour_count"] = out["Property_ID"].map(deg).fillna(0).astype(int)
        out["isolated"] = out["neighbour_count"].eq(0)
        for col in ("property_type", "state", "district"):
            if col in self.df.columns:
                out[col] = self.df[col].to_numpy()
        return out


# ---------------------------------------------------------------------- reporting / CLI
def build_report(engine: SimilarityEngine, stats: dict, cand: pd.DataFrame, edges: pd.DataFrame,
                 counts: pd.DataFrame, elapsed: float, peak_mb: float, input_path: Path) -> dict:
    n = engine.n
    upairs = max(stats["unique_pairs_evaluated"], 1)
    deg = counts["neighbour_count"]

    def group_isolation(col):
        if col not in counts.columns:
            return {}
        g = counts.groupby(col)["isolated"].agg(["size", "mean"])
        g = g[g["size"] >= 5].sort_values("mean", ascending=False)
        return {str(k): {"properties": int(r["size"]), "isolated_pct": round(float(r["mean"] * 100), 2)}
                for k, r in g.iterrows()}

    comp_share = {}
    if not edges.empty:
        for k in COMPONENTS:
            col = f"sim_{k}"
            if col in edges and engine.w[k] > 0:
                comp_share[k] = {"available_pct": round(float(edges[col].notna().mean() * 100), 2),
                                 "mean_similarity": None if edges[col].isna().all()
                                 else round(float(edges[col].mean()), 4)}
    return {
        "component": "similarity_engine",
        "config_id": engine.cfg.config_id(),
        "config": asdict(engine.cfg),
        "normalised_weights_when_all_valid": {
            k: round(v / sum(engine.w.values()), 4) for k, v in engine.w.items() if v > 0},
        "facility_columns_used": engine.facility_columns_used,
        "facility_scale_p95": engine.facility_scale,
        "input_path": str(input_path),
        "efficiency": {
            "properties": n,
            "unique_pairs_evaluated": stats["unique_pairs_evaluated"],
            "elapsed_seconds": round(elapsed, 3),
            "pairs_per_second": round(upairs / max(elapsed, 1e-9)),
            "peak_traced_memory_mb": round(peak_mb, 2),
        },
        "evidence": {
            "node_validity": {
                "price_valid": int(engine.price_valid.sum()),
                "size_valid": int(engine.size_valid.sum()),
                "coordinates_valid": int(engine.coord_valid.sum()),
                "district_valid": int(engine.district_valid.sum()),
                "facility_vector_valid": int(engine.fac_valid.sum()),
            },
            "component_valid_pair_pct": {k: round(v / upairs * 100, 2)
                                         for k, v in stats["component_valid_pairs"].items()},
            "location_resolution_pair_pct": {k: round(v / upairs * 100, 2)
                                             for k, v in stats["location_resolution_pairs"].items()},
            "pairs_min_evidence_ok_pct": round(stats["pairs_min_evidence_ok"] / upairs * 100, 2),
            "pairs_at_or_above_threshold": stats["pairs_at_or_above_threshold"],
            "pairs_at_or_above_threshold_pct": round(stats["pairs_at_or_above_threshold"] / upairs * 100, 4),
        },
        "graph": {
            "directed_topk_candidates": int(len(cand)),
            "retained_edges": int(len(edges)),
            "density": round(2 * len(edges) / max(n * (n - 1), 1), 6),
            "isolated_nodes": int(counts["isolated"].sum()),
            "isolated_pct": round(float(counts["isolated"].mean() * 100), 2),
            "neighbour_count_distribution": {
                "min": int(deg.min()), "p25": float(deg.quantile(.25)), "median": float(deg.median()),
                "p75": float(deg.quantile(.75)), "max": int(deg.max()), "mean": round(float(deg.mean()), 3)},
            "neighbour_count_histogram": {int(k): int(v) for k, v in deg.value_counts().sort_index().items()},
            "edge_similarity": {} if edges.empty else {
                "min": round(float(edges["combined_similarity"].min()), 4),
                "median": round(float(edges["combined_similarity"].median()), 4),
                "max": round(float(edges["combined_similarity"].max()), 4)},
            "edge_component_summary": comp_share,
            "isolation_by_property_type": group_isolation("property_type"),
            "isolation_by_state": group_isolation("state"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute comparable-property similarities and Top-k edges.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--tau", type=float, default=SimilarityConfig.graph_threshold)
    p.add_argument("--k", type=int, default=SimilarityConfig.neighbour_limit)
    p.add_argument("--sigma-loc-km", type=float, default=SimilarityConfig.sigma_loc_km)
    p.add_argument("--district-fallback", type=float, default=SimilarityConfig.district_fallback_value)
    p.add_argument("--location-mode", choices=["coordinates", "district_only", "none"], default="coordinates")
    p.add_argument("--round-coords", type=int, default=None, help="Round lat/lon to N decimals.")
    p.add_argument("--retention", choices=["mutual_topk", "union_topk"], default="mutual_topk")
    p.add_argument("--facility-scaling", choices=["p95", "none"], default="p95")
    p.add_argument("--drop-component", action="append", default=[], choices=list(COMPONENTS),
                   help="Remove a component (repeatable) for ablation.")
    p.add_argument("--pair", nargs=2, metavar=("ID_I", "ID_J"), help="Print one pair record and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)

    active = tuple(c for c in COMPONENTS if c not in set(args.drop_component))
    location_mode = "none" if "location" in args.drop_component else args.location_mode
    mandatory = tuple(c for c in SimilarityConfig.mandatory_components if c in active) or active[:1]
    cfg = SimilarityConfig(
        active_components=active, mandatory_components=mandatory,
        graph_threshold=args.tau, neighbour_limit=args.k, retention_rule=args.retention,
        location_mode=location_mode, sigma_loc_km=args.sigma_loc_km,
        district_fallback_value=args.district_fallback, coordinate_round_decimals=args.round_coords,
        facility_scaling=args.facility_scaling,
        min_valid_components=min(SimilarityConfig.min_valid_components, len(active)),
    )

    if args.pair:
        engine = SimilarityEngine(df, cfg)
        def cast(v):
            return int(v) if np.issubdtype(engine.ids.dtype, np.integer) else v
        print(json.dumps(engine.pair(cast(args.pair[0]), cast(args.pair[1])), indent=2, default=str))
        return

    tracemalloc.start()
    started = time.perf_counter()
    engine = SimilarityEngine(df, cfg)
    cand, stats = engine.candidate_neighbours()
    edges = engine.retain_edges(cand)
    counts = engine.neighbour_counts(edges)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out / "similarity_candidates.csv", index=False)
    edges.to_csv(out / "comparable_edges.csv", index=False)
    counts.to_csv(out / "node_neighbour_counts.csv", index=False)
    report = build_report(engine, stats, cand, edges, counts, elapsed, peak / 1e6, args.input)
    (out / "similarity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    g, e = report["graph"], report["efficiency"]
    print("Similarity engine completed.")
    print(f"Config: {cfg.config_id()} | tau={cfg.graph_threshold} k={cfg.neighbour_limit} "
          f"retention={cfg.retention_rule} active={','.join(active)}")
    print(f"Properties: {e['properties']:,} | unique pairs: {e['unique_pairs_evaluated']:,} "
          f"| time: {e['elapsed_seconds']}s | peak mem: {e['peak_traced_memory_mb']} MB")
    print(f"Pairs >= tau: {report['evidence']['pairs_at_or_above_threshold']:,} "
          f"| retained edges: {g['retained_edges']:,} | density: {g['density']}")
    print(f"Isolated nodes: {g['isolated_nodes']:,} ({g['isolated_pct']}%) "
          f"| median neighbours: {g['neighbour_count_distribution']['median']}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
