"""Generate benchmark v2 (frozen_queries/frozen_queries_set_v2.csv) from the frozen catalogue.

Run from the project root, BEFORE relevance annotation and before any method output is seen:
    python benchmark/generate_benchmark_v2.py

Why v2: in v1, 154 of 250 profiles had no eligible property (67 of them ordinary searches),
leaving only 55 profiles usable for ranking metrics. v2 grounds ordinary profiles in what the
catalogue actually contains. No model score, ranking, graph output or relevance label is used.

Composition (250 profiles, seed 2026):
  * 200 ORDINARY profiles, catalogue-grounded
      - Strata: (district, property_type) combinations with >= MIN_STRATUM_SIZE listings.
      - Profiles per stratum: proportional to sqrt(stratum size), capped at MAX_PER_STRATUM
        (largest-remainder rounding), so large strata do not dominate.
      - Within a stratum, each hard constraint is drawn from the stratum's own distribution:
          max budget        price quantile  ~ U(0.35, 0.95), rounded up to RM10,000
          min bedrooms      bedroom quantile ~ U(0.00, 0.50) among listings within budget
          min bathrooms     bathroom quantile ~ U(0.00, 0.50) among those listings
          parking required  Bernoulli(0.5)
          furnished only    Bernoulli(0.25)
          tenure            "No preference" with p = 0.5, else a tenure present in the subset
      - Soft preferences:
          preferred size    built-up quantile ~ U(0.25, 0.75) of the subset, rounded to 50 sq ft
          main priority     balanced / price / size / facilities, 50 each (shuffled)
          nearby facilities 0-3 of the 4 groups (P = .25/.40/.30/.05); >= 1 if priority = facilities
      - Constraints are NOT forced to leave >= 5 properties; the eligibility rule decides roles.
  * 50 EDGE-CASE profiles carried over from v1 (simple random sample of the 100 v1 edge
    cases, seed 2026), converted to the v2 column format.

Preferred area is always a hard constraint (the property must be in that district).

Partition: train / validation / test = 70 / 10 / 20, assigned within ordinary and edge-case
profiles separately (seeded), so every method evaluates the same test profiles.

Outputs:
    frozen_queries/frozen_queries_set_v2.csv
    benchmark/generation_v2_report.json
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
V1_PATH = ROOT / "frozen_queries" / "frozen_queries_set_v1.csv"
OUT_PATH = ROOT / "frozen_queries" / "frozen_queries_set_v2.csv"
REPORT_PATH = ROOT / "benchmark" / "generation_v2_report.json"

VERSION = "v2"
SEED = 2026
N_ORDINARY = 200
N_EDGE = 50
MIN_STRATUM_SIZE = 10
MAX_PER_STRATUM = 8
PRIORITIES = ["balanced", "price", "size", "facilities"]
FACILITIES = ["Public transport", "Schools", "Retail & groceries", "Parks & recreation"]
FACILITY_COUNT_P = [0.25, 0.40, 0.30, 0.05]   # 0, 1, 2, 3 facility groups
SPLIT = {"train": 0.70, "validation": 0.10, "test": 0.20}
FURNISHED = {"partially furnished", "fully furnished"}

COLUMNS = [
    "query_id", "benchmark_version", "partition", "scenario_source", "source_query_id",
    "generation_stratum", "difficulty", "edge_case_difference",
    "preferred_area", "max_budget_rm", "property_type",
    "min_bedrooms", "min_bathrooms", "parking_required", "furnished_only", "tenure",
    "main_priority", "nearby_facilities", "preferred_built_up_sqft",
    "acceptable_alternatives", "reviewer", "adjudication_status", "generation_note",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def allocate(sizes: pd.Series, total: int, cap: int) -> pd.Series:
    """sqrt-proportional allocation with a cap, largest-remainder rounding (deterministic)."""
    w = np.sqrt(sizes.astype(float))
    alloc = pd.Series(0, index=sizes.index)
    remaining, active = total, list(sizes.index)
    while remaining > 0 and active:
        share = w[active] / w[active].sum() * remaining
        base = np.minimum(np.floor(share), cap - alloc[active])
        alloc[active] += base.astype(int)
        remaining = total - int(alloc.sum())
        if remaining <= 0:
            break
        rema = (share - np.floor(share)).sort_values(ascending=False, kind="mergesort")
        for k in rema.index:
            if remaining == 0:
                break
            if alloc[k] < cap:
                alloc[k] += 1
                remaining -= 1
        active = [k for k in active if alloc[k] < cap]
    return alloc


def q(series: pd.Series, p: float) -> float:
    return float(np.quantile(series.to_numpy(dtype=float), p))


def make_ordinary(cat: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    cat = cat.assign(_d=cat["district"].str.strip().str.lower())
    sizes = cat.groupby(["_d", "property_type"]).size()
    strata = sizes[sizes >= MIN_STRATUM_SIZE].sort_index()
    alloc = allocate(strata, N_ORDINARY, MAX_PER_STRATUM)

    priorities = [p for p in PRIORITIES for _ in range(N_ORDINARY // len(PRIORITIES))]
    rng.shuffle(priorities)

    rows, i = [], 0
    for (district, ptype), n in alloc.items():
        group = cat[(cat["_d"] == district) & (cat["property_type"] == ptype)]
        for _ in range(int(n)):
            budget = math.ceil(q(group["price"], rng.uniform(0.35, 0.95)) / 10_000) * 10_000
            sub = group[group["price"] <= budget]
            min_bed = int(math.floor(q(sub["bedroom"], rng.uniform(0.0, 0.5))))
            sub = sub[sub["bedroom"] >= min_bed]
            min_bath = int(math.floor(q(sub["bathroom"], rng.uniform(0.0, 0.5))))
            sub = sub[sub["bathroom"] >= min_bath]
            parking = int(rng.random() < 0.5)
            furnished = int(rng.random() < 0.25)
            if rng.random() < 0.5:
                tenure = "No preference"
            else:
                counts = sub["tenure"].value_counts().sort_index()
                tenure = str(rng.choice(counts.index.to_numpy(), p=(counts / counts.sum()).to_numpy()))
            size_pref = int(round(q(sub["built_up_area"], rng.uniform(0.25, 0.75)) / 50) * 50)
            priority = priorities[i]
            k = int(rng.choice(len(FACILITY_COUNT_P), p=FACILITY_COUNT_P))
            if priority == "facilities":
                k = max(k, 1)
            fac = sorted(rng.choice(FACILITIES, size=k, replace=False).tolist(), key=FACILITIES.index) if k else []
            rows.append({
                "scenario_source": "researcher_catalogue_grounded",
                "source_query_id": "",
                "generation_stratum": f"{district} | {ptype}",
                "difficulty": "standard",
                "edge_case_difference": "",
                "preferred_area": district,
                "max_budget_rm": budget,
                "property_type": ptype,
                "min_bedrooms": min_bed,
                "min_bathrooms": min_bath,
                "parking_required": parking,
                "furnished_only": furnished,
                "tenure": tenure,
                "main_priority": priority,
                "nearby_facilities": ";".join(fac),
                "preferred_built_up_sqft": max(size_pref, 100),
                "generation_note": "",
            })
            i += 1
    return rows, strata, alloc


def make_edge(v1: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    edge = v1[v1["edge_case_difference"].notna()].sort_values("query_id")
    pick = edge.iloc[np.sort(rng.choice(len(edge), size=N_EDGE, replace=False))]
    rows = []
    for _, r in pick.iterrows():
        fac = "" if str(r["nearby_facilities"]).strip() in {"No preference", "", "nan"} else str(r["nearby_facilities"]).strip()
        priority, note = str(r["main_priority"]).strip(), ""
        if priority == "location":
            # preferred area is always a hard constraint, so "location" has no ranking effect
            priority, note = "balanced", "v1 main_priority 'location' removed -> 'balanced'"
        elif priority == "facilities" and not fac:
            priority, note = "balanced", "v1 main_priority 'facilities' with no facility -> 'balanced'"
        rows.append({
            "scenario_source": "v1_edge_case",
            "source_query_id": r["query_id"],
            "generation_stratum": "",
            "difficulty": "hard",
            "edge_case_difference": r["edge_case_difference"],
            "preferred_area": str(r["preferred_area"]).strip().lower(),
            "max_budget_rm": int(r["max_budget_rm"]),
            "property_type": str(r["property_type"]).strip(),
            "min_bedrooms": int(r["min_bedrooms"]),
            "min_bathrooms": int(r["min_bathrooms"]),
            "parking_required": int(r["parking_required"]),
            "furnished_only": int(r["furnished_only"]),
            "tenure": str(r["tenure"]).strip(),
            "main_priority": priority,
            "nearby_facilities": fac,
            "preferred_built_up_sqft": int(r["preferred_built_up_sqft"]),
            "generation_note": note,
        })
    return rows


def assign_partitions(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    part = pd.Series("", index=df.index)
    for _, idx in df.groupby("difficulty").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * SPLIT["train"]))
        n_val = int(round(n * SPLIT["validation"]))
        part[idx[:n_train]] = "train"
        part[idx[n_train:n_train + n_val]] = "validation"
        part[idx[n_train + n_val:]] = "test"
    return part


def main() -> None:
    rng = np.random.default_rng(SEED)
    cat = pd.read_csv(CATALOGUE_PATH)
    v1 = pd.read_csv(V1_PATH)

    ordinary, strata, alloc = make_ordinary(cat, rng)
    edge = make_edge(v1, rng)
    df = pd.DataFrame(ordinary + edge)
    df.insert(0, "query_id", [f"Q{i:03d}" for i in range(1, len(df) + 1)])
    df.insert(1, "benchmark_version", VERSION)
    df.insert(2, "partition", assign_partitions(df, rng))
    for c in ("acceptable_alternatives", "reviewer"):
        df[c] = ""
    df["adjudication_status"] = "pending"
    df = df[COLUMNS]
    df.to_csv(OUT_PATH, index=False)

    report = {
        "benchmark_version": VERSION,
        "seed": SEED,
        "inputs": {"catalogue": CATALOGUE_PATH.name, "catalogue_sha256_16": _sha(CATALOGUE_PATH),
                   "catalogue_rows": int(len(cat)), "v1_queries": V1_PATH.name,
                   "v1_sha256_16": _sha(V1_PATH)},
        "parameters": {"n_ordinary": N_ORDINARY, "n_edge": N_EDGE,
                       "min_stratum_size": MIN_STRATUM_SIZE, "max_per_stratum": MAX_PER_STRATUM,
                       "split": SPLIT},
        "strata_used": int((alloc > 0).sum()),
        "districts_covered": int(df.loc[df.difficulty == "standard", "preferred_area"].nunique()),
        "property_types_covered": int(df.loc[df.difficulty == "standard", "property_type"].nunique()),
        "profiles_per_stratum": {f"{d} | {t}": int(n) for (d, t), n in alloc.items() if n > 0},
        "main_priority_counts": df["main_priority"].value_counts().to_dict(),
        "partition_counts": df.groupby(["difficulty", "partition"]).size()
        .unstack(fill_value=0).to_dict(orient="index"),
        "notes": [
            "No model output, graph output or relevance label was used to generate profiles.",
            "Ordinary constraints are drawn from each stratum's own listing distribution; "
            "they are not forced to leave a minimum number of eligible properties.",
            "Run benchmark/apply_eligibility_rule.py on this file to assign benchmark roles.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark {VERSION}: {len(df)} profiles "
          f"({(df.difficulty == 'standard').sum()} ordinary, {(df.difficulty == 'hard').sum()} edge case)")
    print(f"Strata used: {report['strata_used']} | districts: {report['districts_covered']} "
          f"| property types: {report['property_types_covered']}")
    print(f"Saved: {OUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
