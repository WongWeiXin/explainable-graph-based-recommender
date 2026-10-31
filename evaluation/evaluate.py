"""Offline ranking evaluation on benchmark v2 (frozen labels, protocol REL-v2-1).

Run from the project root AFTER finalize_labels.py:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --provisional   # pipeline check from the A1 sheet only;
                                                  # written to results_provisional/, NOT for reporting

Queries   validation + test queries with eligibility role "ranking" (E >= 5). The TEST partition is
          the primary result; validation is reported separately (it was used for selection).
Methods   cpg, content_based, rule_based, lightgcn (mean of seeds 42/43/44 per query, plus each seed),
          and every ablation / robustness condition in outputs/ablation/top5_all_conditions.csv.
Metrics   K = 5. Relevance grades 0-3; "relevant" = grade >= 2.
          NDCG@5    gain 2^rel - 1, discount log2(rank + 1); ideal DCG from all labelled pairs of the
                    query (the pool)
          P@5       relevant items in the Top-5 / 5
          Recall@5  relevant items in the Top-5 / relevant items in the pool (pooled recall);
                    queries with no relevant pooled item are excluded from Recall@5
          MRR       1 / rank of the first relevant item in the Top-5 (0 if none)
Tests     paired over test queries: methods vs CPG; ablation conditions vs the full condition.
          Mean difference with 95% bootstrap CI (10,000 resamples of queries, seed 2026), Wilcoxon
          signed-rank p (scipy; exact sign-flip permutation if scipy is missing), Holm correction
          within each family, matched-pairs rank-biserial effect size.

Outputs (evaluation/results/):
    per_query_metrics.csv   method x query metrics
    method_summary.csv      mean, SD, 95% CI per method / partition / metric
    paired_tests.csv        paired comparisons (test partition)
    evaluation_summary.json label source, query counts, unlabelled-item checks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "evaluation" / "labelling"
FINAL_LABELS = LAB_DIR / "relevance_labels_v2_final.csv"
MANIFEST = ROOT / "benchmark" / "eligibility_manifest_v2.csv"
SOURCES = {
    "cpg": ROOT / "outputs" / "top5" / "cpg_top5_recommendations.csv",
    "content_based": ROOT / "outputs" / "baseline" / "content_based_top5.csv",
    "rule_based": ROOT / "outputs" / "baseline" / "rule_based_top5.csv",
}
LIGHTGCN = ROOT / "baseline_models" / "lightgcn_outputs" / "lightgcn_top5_recommendations.csv"
ABLATION = ROOT / "outputs" / "ablation" / "top5_all_conditions.csv"
K = 5
RELEVANT_MIN = 2
METRICS = ["ndcg@5", "p@5", "recall@5", "mrr"]
SEED = 2026
N_BOOT = 10_000
PRIMARY = "cpg"
BASELINES = ["content_based", "rule_based", "lightgcn"]


# ---------------------------------------------------------------------------------- inputs
def load_labels(provisional: bool) -> tuple[pd.DataFrame, str]:
    if not provisional:
        if not FINAL_LABELS.exists():
            raise SystemExit("Final labels not found. Run evaluation/finalize_labels.py first "
                             "(or --provisional for a pipeline check).")
        lab = pd.read_csv(FINAL_LABELS)
        return lab[["query_id", "Property_ID", "relevance"]], FINAL_LABELS.name
    from build_adjudication_sheet import A1, read_sheet
    a1, _ = read_sheet(A1)
    key = pd.read_csv(LAB_DIR / "pool_key_DO_NOT_SHOW_ASSESSOR.csv")[["pair_id", "query_id", "Property_ID"]]
    lab = key.merge(a1[["pair_id", "grade"]], on="pair_id").rename(columns={"grade": "relevance"})
    return lab[["query_id", "Property_ID", "relevance"]], f"PROVISIONAL: {A1.name} (not adjudicated)"


def load_runs() -> pd.DataFrame:
    frames = []
    for name, path in SOURCES.items():
        d = pd.read_csv(path).dropna(subset=["Property_ID"])
        frames.append(d.assign(method=name, family="method")[["method", "family", "query_id", "rank", "Property_ID"]])
    lg = pd.read_csv(LIGHTGCN).dropna(subset=["Property_ID"])
    for seed, d in lg.groupby("seed"):
        frames.append(d.assign(method=f"lightgcn_seed{int(seed)}", family="lightgcn_seed")
                      [["method", "family", "query_id", "rank", "Property_ID"]])
    if ABLATION.exists():
        a = pd.read_csv(ABLATION).dropna(subset=["Property_ID"])
        frames.append(a.assign(method="abl:" + a["condition"], family="ablation")
                      [["method", "family", "query_id", "rank", "Property_ID"]])
    runs = pd.concat(frames, ignore_index=True)
    runs["Property_ID"] = runs["Property_ID"].astype(int)
    runs["rank"] = runs["rank"].astype(int)
    return runs


# ---------------------------------------------------------------------------------- metrics
def dcg(grades) -> float:
    g = np.asarray(grades, dtype=float)
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def query_metrics(top: pd.DataFrame, pool: pd.Series) -> dict:
    """top: one method's Top-K rows for a query (rank, Property_ID); pool: Property_ID -> grade."""
    top = top.sort_values("rank").head(K)
    grades = top["Property_ID"].map(pool)
    unlabelled = int(grades.isna().sum())
    grades = grades.fillna(0).to_numpy()
    rel = grades >= RELEVANT_MIN
    n_rel_pool = int((pool >= RELEVANT_MIN).sum())
    ideal = dcg(np.sort(pool.to_numpy())[::-1][:K])
    first = np.flatnonzero(rel)
    return {
        "ndcg@5": dcg(grades) / ideal if ideal > 0 else np.nan,
        "p@5": rel.sum() / K,
        "recall@5": rel.sum() / n_rel_pool if n_rel_pool else np.nan,
        "mrr": 1.0 / (first[0] + 1) if first.size else 0.0,
        "returned": int(len(top)),
        "unlabelled_items": unlabelled,
        "relevant_in_pool": n_rel_pool,
    }


# ---------------------------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, rng) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d: np.ndarray) -> tuple[float, str]:
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0, "all_zero"
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue), "wilcoxon"
    except ImportError:
        nz = d[d != 0]
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        obs = abs((ranks * np.sign(nz)).sum())
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1, 1], size=(20_000, len(nz)))
        null = np.abs((signs * ranks).sum(axis=1))
        return float((null >= obs).mean()), "signflip_permutation"


def rank_biserial(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    r = pd.Series(np.abs(nz)).rank().to_numpy()
    wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
    return float((wp - wm) / (wp + wm))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(p.index)


# ---------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all methods on benchmark v2.")
    ap.add_argument("--provisional", action="store_true",
                    help="Use the A1 sheet only (pipeline check, not for reporting).")
    args = ap.parse_args()
    out_dir = ROOT / "evaluation" / ("results_provisional" if args.provisional else "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, label_source = load_labels(args.provisional)
    manifest = pd.read_csv(MANIFEST, dtype={"query_id": str})
    evalq = manifest[(manifest["benchmark_role"] == "ranking")
                     & manifest["partition"].isin(["validation", "test"])][["query_id", "partition"]]
    pools = {q: g.set_index("Property_ID")["relevance"].astype(float) for q, g in labels.groupby("query_id")}
    missing_pool = sorted(set(evalq["query_id"]) - set(pools))
    if missing_pool:
        raise SystemExit(f"No labels for evaluation queries: {missing_pool}")

    runs = load_runs()
    runs = runs[runs["query_id"].isin(set(evalq["query_id"]))]

    rows = []
    for (method, family, qid), top in runs.groupby(["method", "family", "query_id"]):
        rows.append({"method": method, "family": family, "query_id": qid, **query_metrics(top, pools[qid])})
    pq = pd.DataFrame(rows).merge(evalq, on="query_id")

    # LightGCN as one method: per-query mean over seeds.
    lg = pq[pq["family"] == "lightgcn_seed"].groupby(["query_id", "partition"], as_index=False)[
        METRICS + ["unlabelled_items"]].mean()
    lg["method"], lg["family"] = "lightgcn", "method"
    pq = pd.concat([pq, lg], ignore_index=True)

    # Every method must cover every evaluation query (otherwise paired tests are invalid).
    coverage = pq.groupby("method")["query_id"].nunique()
    incomplete = coverage[coverage < evalq["query_id"].nunique()].to_dict()

    pq.to_csv(out_dir / "per_query_metrics.csv", index=False)

    rng = np.random.default_rng(SEED)
    summ = []
    for (method, family, part), g in pq.groupby(["method", "family", "partition"]):
        rec = {"method": method, "family": family, "partition": part, "queries": int(g["query_id"].nunique())}
        for m in METRICS:
            x = g[m].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, rng)
            rec.update({m: round(float(np.nanmean(x)), 4), f"{m}_sd": round(float(np.nanstd(x, ddof=1)), 4),
                        f"{m}_ci_low": round(lo, 4), f"{m}_ci_high": round(hi, 4)})
        summ.append(rec)
    summary = pd.DataFrame(summ).sort_values(["partition", "family", "ndcg@5"], ascending=[True, True, False])
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    # Paired tests on the TEST partition.
    test = pq[pq["partition"] == "test"]
    wide = {m: test.pivot_table(index="query_id", columns="method", values=m) for m in METRICS}
    families = {"methods_vs_cpg": (PRIMARY, BASELINES),
                "ablation_vs_full": ("abl:full", sorted(c for c in wide["ndcg@5"].columns
                                                        if c.startswith("abl:") and c != "abl:full"))}
    tests = []
    for fam, (ref, others) in families.items():
        if ref not in wide["ndcg@5"].columns:
            continue
        for m in METRICS:
            block = []
            for o in others:
                if o not in wide[m].columns:
                    continue
                pair = wide[m][[ref, o]].dropna()
                d = (pair[o] - pair[ref]).to_numpy()
                p, test_name = wilcoxon_p(d)
                lo, hi = bootstrap_ci(d, rng)
                block.append({"family": fam, "metric": m, "reference": ref, "compared": o,
                              "queries": int(len(d)), "mean_reference": round(float(pair[ref].mean()), 4),
                              "mean_compared": round(float(pair[o].mean()), 4),
                              "mean_diff_compared_minus_reference": round(float(d.mean()), 4),
                              "diff_ci_low": round(lo, 4), "diff_ci_high": round(hi, 4),
                              "p_value": round(p, 5), "test": test_name,
                              "rank_biserial": round(rank_biserial(d), 4)})
            if block:
                b = pd.DataFrame(block)
                b["p_holm"] = holm(b["p_value"]).round(5)
                tests.append(b)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    tests.to_csv(out_dir / "paired_tests.csv", index=False)

    info = {
        "label_source": label_source,
        "protocol": "REL-v2-1",
        "k": K, "relevant_min_grade": RELEVANT_MIN,
        "queries": evalq["partition"].value_counts().to_dict(),
        "methods_incomplete_query_coverage": incomplete,
        "unlabelled_items_in_top5_total": int(pq.loc[pq["family"] != "lightgcn_seed", "unlabelled_items"].sum()),
        "queries_without_relevant_pooled_item": sorted(
            pq.loc[(pq["method"] == PRIMARY) & (pq["relevant_in_pool"] == 0), "query_id"].unique().tolist()),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    show = summary[(summary["family"] == "method")][["method", "partition", "queries", *METRICS]]
    print(f"Labels: {label_source}")
    print(show.to_string(index=False))
    if not tests.empty:
        t = tests[(tests["family"] == "methods_vs_cpg") & (tests["metric"] == "ndcg@5")]
        print("\nTest NDCG@5, baseline minus CPG:")
        print(t[["compared", "mean_diff_compared_minus_reference", "diff_ci_low", "diff_ci_high",
                 "p_value", "p_holm"]].to_string(index=False))
    print(f"\nChecks: {info}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
