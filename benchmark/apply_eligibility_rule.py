"""Benchmark candidate-sufficiency rule (thesis Section 3.7).

Run from the project root, BEFORE relevance annotation and before any model output is seen:
    python benchmark/apply_eligibility_rule.py                                   # v2 (default)
    python benchmark/apply_eligibility_rule.py --queries frozen_queries/frozen_queries_set_v1.csv  # v1

For every search profile q in frozen_queries_set.csv, the eligible candidate set C_q is
built from the frozen catalogue with the SAME hard-constraint function the recommender
uses (recommender_service.apply_hard_constraints). Each query then receives one role:

    E_q = |C_q| = 0              -> "empty_result_test"
        Used only to check that every method returns the explicit empty-result state
        without relaxing constraints. Excluded from ranking metrics.
    1 <= E_q < MIN_RANKING_CANDIDATES -> "sparse_candidate"
        Excluded from the primary ranking metrics; reported descriptively.
    E_q >= MIN_RANKING_CANDIDATES    -> "ranking"
        Used for Precision@K, Recall@K, NDCG@K and MRR.

MIN_RANKING_CANDIDATES = 5 = the smallest reported K, so every ranking query can fill a
Top-5 list. Roles are fixed once per (benchmark version, catalogue version) and are never
changed after looking at method results. If either input changes, rerun this script and
record a new manifest version.

Outputs (benchmark/, one pair per benchmark version):
    eligibility_manifest_<version>.csv   query_id, partition, eligible_count, benchmark_role, ...
    eligibility_summary_<version>.json   rule, input hashes, role counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recommender_service import apply_hard_constraints  # noqa: E402  (single source of truth)

DEFAULT_QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
CATALOGUE_PATH = ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
OUT_DIR = ROOT / "benchmark"

RULE_VERSION = "ELIG-v1"
MIN_RANKING_CANDIDATES = 5
ROLE_EMPTY = "empty_result_test"
ROLE_SPARSE = "sparse_candidate"
ROLE_RANKING = "ranking"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def hard_profile(row: pd.Series) -> dict:
    """Hard-constraint part of a benchmark profile (preferred area is always a hard constraint)."""
    tenure = str(row["tenure"]).strip()
    return {
        "max_budget_rm": float(row["max_budget_rm"]),
        "preferred_area": str(row["preferred_area"]).strip(),
        "property_type": str(row["property_type"]).strip(),
        "min_bedrooms": float(row["min_bedrooms"]) or None,
        "min_bathrooms": float(row["min_bathrooms"]) or None,
        "parking_required": int(row["parking_required"]) == 1,
        "furnished_only": int(row["furnished_only"]) == 1,
        "tenure": None if tenure.casefold() in {"no preference", "any", ""} else tenure,
    }


def assign_role(n: int) -> str:
    if n == 0:
        return ROLE_EMPTY
    if n < MIN_RANKING_CANDIDATES:
        return ROLE_SPARSE
    return ROLE_RANKING


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign benchmark roles (rule ELIG-v1).")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    args = ap.parse_args()
    queries_path = args.queries if args.queries.is_absolute() else ROOT / args.queries
    queries = pd.read_csv(queries_path)
    catalogue = pd.read_csv(CATALOGUE_PATH)
    version = (str(queries["benchmark_version"].iloc[0]) if "benchmark_version" in queries
               else "v1")
    manifest_path = OUT_DIR / f"eligibility_manifest_{version}.csv"
    summary_path = OUT_DIR / f"eligibility_summary_{version}.json"

    rows = []
    for _, q in queries.iterrows():
        eligible, audit = apply_hard_constraints(catalogue, hard_profile(q))
        n = len(eligible)
        # The constraint that emptied the set (first one leaving 0), for diagnosis only.
        emptied_by = next((a["constraint"] for a in audit if a["candidates_after"] == 0), None)
        rows.append({
            "query_id": q["query_id"],
            "benchmark_version": version,
            "partition": q.get("partition", ""),
            "eligible_count": n,
            "benchmark_role": assign_role(n),
            "emptied_by_constraint": emptied_by,
            "is_edge_case": pd.notna(q.get("edge_case_difference"))
            and str(q.get("edge_case_difference")).strip() != "",
            "rule_version": RULE_VERSION,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)

    counts = manifest["benchmark_role"].value_counts().reindex(
        [ROLE_RANKING, ROLE_SPARSE, ROLE_EMPTY], fill_value=0)
    summary = {
        "rule_version": RULE_VERSION,
        "benchmark_version": version,
        "min_ranking_candidates": MIN_RANKING_CANDIDATES,
        "roles": {
            ROLE_EMPTY: "E_q = 0; checks explicit empty-result handling only",
            ROLE_SPARSE: f"1 <= E_q < {MIN_RANKING_CANDIDATES}; excluded from primary ranking metrics",
            ROLE_RANKING: f"E_q >= {MIN_RANKING_CANDIDATES}; used for P@K, Recall@K, NDCG@K, MRR",
        },
        "inputs": {
            "queries_file": queries_path.name, "queries_sha256_16": _sha(queries_path),
            "catalogue_file": CATALOGUE_PATH.name, "catalogue_sha256_16": _sha(CATALOGUE_PATH),
            "catalogue_rows": int(len(catalogue)), "queries": int(len(queries)),
        },
        "role_counts": {k: int(v) for k, v in counts.items()},
        "role_counts_edge_case": manifest[manifest["is_edge_case"]]["benchmark_role"]
        .value_counts().to_dict(),
        "role_counts_ordinary": manifest[~manifest["is_edge_case"]]["benchmark_role"]
        .value_counts().to_dict(),
        "role_counts_by_partition": manifest.groupby(["partition", "benchmark_role"]).size()
        .unstack(fill_value=0).to_dict(orient="index") if "partition" in manifest else {},
        "empty_sets_by_first_emptying_constraint": manifest["emptied_by_constraint"]
        .value_counts().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Eligibility rule {RULE_VERSION} on benchmark {version}: {len(queries)} queries "
          f"({len(catalogue):,} catalogue properties).")
    for role, n in counts.items():
        print(f"  {role:18s} {n}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
