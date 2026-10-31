"""Content-based baseline (CBR) for the frozen query set - complete implementation.

Everything this baseline needs is in this file: eligibility, ranking, its Top-5 output, the
shared eligibility summary and the pooled relevance-label file.

    python baseline_models/content_based_baseline.py

Model - weighted attribute match between the query and each listing, i.e. the same content
score the prototype uses as its direct component, with the graph and market switched off
(the frozen "direct_only" ranking variant):

    price_value_fit   how much of the budget the asking price uses
    size_fit          closeness of built-up area to preferred_built_up_sqft
    facility_fit      share of the selected facility types present nearby

Base weights 0.25 / 0.25 / 0.15, multiplied by PRIORITY_MULTIPLIER (2.5) for the query's
main_priority, then normalised over the active weights. No comparable-property graph, no
similarity measure, no neighbour evidence.

Outputs
    outputs/baseline/content_based_top5.csv      Top-5 per query (821 rows: 732 + 89 empty)
    outputs/baseline/eligibility_summary.csv     per query, both models' eligibility check
    frozen_queries/relevance_labels_v2.csv       graded relevance for every pair either model
                                                 returned (rebuilt from whatever Top-5 files
                                                 exist in outputs/baseline/)

The reporting functions are imported from rule_based_baseline.py so the grading rubric is
defined exactly once; they only read outputs/baseline/*_top5.csv, so the shared files are
correct no matter which of the two scripts was run last.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import recommender_service as svc  # noqa: E402
from cpg_ranker import RankingConfig, rank_candidates  # noqa: E402

# Shared reporting: eligibility summary + pooled relevance labels (defined once, in the RB file).
from rule_based_baseline import (  # noqa: E402
    ELIG_COLUMNS,
    FACILITY_COLUMNS,
    LABEL_COLUMNS,
    RELEVANCE_TEXT,
    TOP5_COLUMNS,
    TOP_K,
    _context,
    _int,
    _listing_name,
    build_labels,
    write_eligibility_summary,
)

METHOD = "content_based"
VARIANT = "direct_only"          # direct component only: no graph, no market context
CRITERION = "weighted attribute match (price / size / facility fit)"
QUERIES = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
OUT_DIR = ROOT / "outputs" / "baseline"
OUT = OUT_DIR / "content_based_top5.csv"
OUT_ELIG = OUT_DIR / "eligibility_summary.csv"
OUT_LABELS = ROOT / "frozen_queries" / "relevance_labels_v2.csv"


# --------------------------------------------------------------------------- model
def score(eligible: pd.DataFrame, prof: dict) -> pd.DataFrame:
    """Eligible frame plus the weighted attribute score (and its per-preference evidence)."""
    return svc.score_direct(eligible, svc.build_direct_query(prof))


def rank(eligible: pd.DataFrame, prof: dict, k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    """Top-k by weighted attribute match. Returns (frame, configuration actually used)."""
    cfg = RankingConfig(variant=VARIANT, top_k=k)
    ranked, _ = rank_candidates(score(eligible, prof), cfg)
    return ranked.head(k), {"criterion": CRITERION, "variant": VARIANT,
                            "weights": svc.build_direct_query(prof)["weights"],
                            "ranking_config_id": cfg.config_id()}


# --------------------------------------------------------------------------- run the query set
def run(queries: pd.DataFrame, catalogue: pd.DataFrame,
        k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    """Eligibility + Top-k for every query. Returns (rows, per-query eligibility records)."""
    info = catalogue.set_index("Property_ID")
    rows, eligibility = [], {}

    for _, q in queries.iterrows():
        qid, split = q["query_id"], q.get("split", q.get("partition", ""))
        try:
            prof = svc.parse_profile(q.to_dict())
        except svc.QueryError as exc:
            rows.append({"method": METHOD, "query_id": qid, "split": split,
                         "status": f"invalid_query: {exc}"})
            continue

        eligible, audit = svc.apply_hard_constraints(catalogue, prof)
        emptied_by = next((a["constraint"] for a in audit if a["candidates_after"] == 0), "")
        eligibility[qid] = {"split": split, "eligible_count": int(len(eligible)),
                            "emptied_by": emptied_by, "context": _context(eligible, prof)}

        if eligible.empty:
            rows.append({"method": METHOD, "query_id": qid, "split": split,
                         "eligible_count": 0, "status": "no_eligible_property"})
            continue

        top, _ = rank(eligible, prof, k)
        status = "ok" if len(eligible) >= k else "fewer_than_top_n_eligible"
        for position, (_, r) in enumerate(top.iterrows(), start=1):
            pid = int(r["Property_ID"])
            listing = info.loc[pid]
            rows.append({
                "method": METHOD, "query_id": qid, "split": split,
                "eligible_count": int(len(eligible)), "rank": position, "Property_ID": pid,
                "listing": _listing_name(listing),
                "district": str(listing["district"]).title(),
                "property_type": str(listing["property_type"]),
                "asking_price_rm": _int(listing["price"]),
                "built_up_sqft": _int(listing["built_up_area"]),
                "bedrooms": _int(listing["bedroom"]),
                "bathrooms": _int(listing["bathroom"]),
                "score": round(float(r["final_score"]), 4),
                "criterion": CRITERION, "status": status,
            })

    return pd.DataFrame(rows, columns=TOP5_COLUMNS), eligibility


# --------------------------------------------------------------------------- entry point
def main() -> None:
    queries = pd.read_csv(QUERIES, dtype=str, keep_default_na=False)
    catalogue = pd.read_csv(svc.DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, eligibility = run(queries, catalogue)
    rows.to_csv(OUT, index=False)
    elig = write_eligibility_summary(eligibility, OUT_ELIG)
    labels = build_labels(queries, catalogue, OUT_LABELS)

    total = elig[elig["query_id"] == "TOTAL"].iloc[0]
    dist = labels["relevance"].value_counts().sort_index().to_dict()
    print(f"content_based: {int(rows['status'].isin(['ok', 'fewer_than_top_n_eligible']).sum())} "
          f"recommendations over "
          f"{rows.loc[rows['Property_ID'].notna(), 'query_id'].nunique()} queries "
          f"-> {rows['status'].value_counts().to_dict()}")
    print(f"eligibility: {total['enough_for_top5']} queries can fill a Top-5, "
          f"{int((elig['content_based_check'] == 'empty_eligible_set').sum())} empty")
    print(f"relevance sheet: {len(labels)} pairs over {labels['query_id'].nunique()} queries "
          f"({labels['split'].value_counts().to_dict()}) grades {dist}")
    print("Saved: outputs/baseline/content_based_top5.csv, "
          "outputs/baseline/eligibility_summary.csv, frozen_queries/relevance_labels_v2.csv")


if __name__ == "__main__":
    main()
