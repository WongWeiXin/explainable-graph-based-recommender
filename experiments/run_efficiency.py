"""Efficiency tests (Sec 4.x, Eq. 4.28-4.29; environment for Sec 4.11).

Run from the project root, on the machine you report in the thesis, with other programs closed:
    python experiments/run_efficiency.py                 # default: 3 repeats
    python experiments/run_efficiency.py --repeats 5
    python experiments/run_efficiency.py --skip-scaling  # faster

What is measured
    1. Graph construction on the full catalogue: time per stage (pairwise similarity + Top-k
       candidates, mutual retention, graph object), pairs per second, peak traced memory.
    2. Scaling test: the same pipeline on random subsets n = 500 ... all (seed 2026), to compare the
       measured growth with the O(n^2 F) bound. Reports the fitted log-log exponent (~2 expected).
    3. Online query latency for all 250 v2 queries, per stage:
           hard-constraint filter -> direct preference score -> graph support + final ranking
       for CPG (full), CBR (direct_only) and CPG + market context (full_market). Median of the
       repeats per query; then mean / median / p95 / max over queries.
    4. LightGCN (if its training log has timing columns): training seconds, seconds per epoch,
       inference ms per query.
    5. Environment: OS, CPU, cores, RAM, Python and package versions, commands.

Outputs (outputs/efficiency/):
    environment.json, graph_build_runs.csv, scaling.csv, query_latency.csv, efficiency_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from importlib import metadata

import numpy as np
import pandas as pd

from common import (COMPONENTS, ROOT, ComparableGraph, NODE_ATTRIBUTES, SimilarityEngine, load_catalogue,
                    load_market, load_queries, prepare_queries, rank_query, similarity_config, svc)

OUT_DIR = ROOT / "outputs" / "efficiency"
LIGHTGCN_LOG = ROOT / "baseline_models" / "lightgcn_outputs" / "lightgcn_training_log.csv"
SCALING_SIZES = (500, 1000, 1500, 2000, 2500)
SEED = 2026
VARIANTS = {"cpg": "full", "cbr": "direct_only", "cpg_market": "full_market"}
PACKAGES = ("numpy", "pandas", "networkx", "scikit-learn", "torch", "flask", "matplotlib")


# ------------------------------------------------------------------ environment
def _total_ram_gb():
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                return round(int(f.readline().split()[1]) / 1e6, 2)
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1e9, 2)
        if sys.platform == "darwin":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None
    return None


def _cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def environment(argv) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"os": platform.platform(), "cpu": _cpu_name(), "logical_cores": os.cpu_count(),
            "ram_gb": _total_ram_gb(), "python": sys.version.split()[0], "packages": versions,
            "command": "python " + " ".join(argv), "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}


# ------------------------------------------------------------------ graph construction
def timed_build(df: pd.DataFrame, cfg, trace_memory: bool = False) -> dict:
    if trace_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    engine = SimilarityEngine(df, cfg)
    t1 = time.perf_counter()
    cand, stats = engine.candidate_neighbours()
    t2 = time.perf_counter()
    edges = engine.retain_edges(cand)
    t3 = time.perf_counter()
    nodes = df[["Property_ID", *[c for c in NODE_ATTRIBUTES if c in df.columns]]]
    cpg = ComparableGraph.from_frames(nodes, edges)
    t4 = time.perf_counter()
    peak = None
    if trace_memory:
        peak = round(tracemalloc.get_traced_memory()[1] / 1e6, 2)
        tracemalloc.stop()
    n = len(df)
    pairs = n * (n - 1) // 2
    return {"n": n, "pairs": pairs, "active_components": len(cfg.active_components),
            "prepare_s": t1 - t0, "pairwise_topk_s": t2 - t1, "retention_s": t3 - t2,
            "graph_object_s": t4 - t3, "total_s": t4 - t0,
            "pairs_per_second": pairs / max(t2 - t1, 1e-9),
            "edges": int(cpg.graph.number_of_edges()), "peak_traced_memory_mb": peak,
            "config_id": cfg.config_id()}


def describe(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    return {"mean": round(float(s.mean()), 4), "sd": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
            "median": round(float(s.median()), 4), "p95": round(float(s.quantile(0.95)), 4),
            "max": round(float(s.max()), 4), "n": int(len(s))}


# ------------------------------------------------------------------ query latency
def query_latency(catalogue, queries, market, cpg, repeats: int) -> pd.DataFrame:
    rows = []
    records = queries.to_dict(orient="records")
    # Warm-up (imports, caches) - not recorded.
    for rec in records[:5]:
        prof = svc.parse_profile(rec)
        el, _ = svc.apply_hard_constraints(catalogue, prof)
        if not el.empty:
            ctx = prepare_queries(catalogue, pd.DataFrame([rec]))[0]
            for v in VARIANTS.values():
                rank_query(ctx, v, cpg, market)

    for rec in records:
        for method, variant in VARIANTS.items():
            runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                prof = svc.parse_profile(rec)
                eligible, _ = svc.apply_hard_constraints(catalogue, prof)
                t1 = time.perf_counter()
                if eligible.empty:
                    runs.append((t1 - t0, 0.0, 0.0, t1 - t0))
                    continue
                direct = svc.score_direct(eligible, svc.build_direct_query(prof))
                t2 = time.perf_counter()
                ctx = type("Ctx", (), {"direct": direct})()
                rank_query(ctx, variant, cpg, market)
                t3 = time.perf_counter()
                runs.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
            med = np.median(np.array(runs), axis=0) * 1000
            rows.append({"query_id": rec["query_id"], "partition": rec.get("partition"),
                         "benchmark_role": rec.get("benchmark_role"), "method": method,
                         "eligible_count": len(eligible),
                         "hard_filter_ms": med[0], "direct_score_ms": med[1],
                         "graph_and_ranking_ms": med[2], "total_ms": med[3]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Efficiency tests for the CPG recommender.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-scaling", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    env = environment(sys.argv)
    (OUT_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Environment: {env['cpu']} | {env['logical_cores']} cores | {env['ram_gb']} GB | "
          f"Python {env['python']} | pandas {env['packages']['pandas']}")

    catalogue = load_catalogue()
    market = load_market()
    cfg = similarity_config()

    # 1. Full-catalogue graph construction (timing runs without tracemalloc; one memory run).
    print(f"\n[1] Graph construction, n={len(catalogue):,}, {args.repeats} repeats ...")
    build_runs = [{**timed_build(catalogue, cfg), "run": i + 1, "kind": "timing"} for i in range(args.repeats)]
    build_runs.append({**timed_build(catalogue, cfg, trace_memory=True), "run": 0, "kind": "memory"})
    build_df = pd.DataFrame(build_runs)
    build_df.to_csv(OUT_DIR / "graph_build_runs.csv", index=False)
    timing = build_df[build_df["kind"] == "timing"]
    print(f"    total {timing['total_s'].mean():.2f} s (sd {timing['total_s'].std(ddof=1) if len(timing) > 1 else 0:.2f}) | "
          f"{timing['pairs_per_second'].mean():,.0f} pairs/s | peak memory "
          f"{build_df.loc[build_df['kind'] == 'memory', 'peak_traced_memory_mb'].iloc[0]} MB")

    # 2. Scaling test.
    scaling_df = pd.DataFrame()
    exponent = None
    if not args.skip_scaling:
        sizes = [n for n in SCALING_SIZES if n < len(catalogue)] + [len(catalogue)]
        print(f"\n[2] Scaling test, n = {sizes} ...")
        rng = np.random.default_rng(SEED)
        rows = []
        for n in sizes:
            sub = catalogue if n == len(catalogue) else \
                catalogue.iloc[np.sort(rng.choice(len(catalogue), n, replace=False))].reset_index(drop=True)
            for i in range(args.repeats):
                rows.append({**timed_build(sub, cfg), "run": i + 1})
            print(f"    n={n:>5}: {np.mean([r['total_s'] for r in rows if r['n'] == n]):.2f} s")
        scaling_df = pd.DataFrame(rows)
        scaling_df.to_csv(OUT_DIR / "scaling.csv", index=False)
        agg = scaling_df.groupby("n")["pairwise_topk_s"].median()
        exponent = round(float(np.polyfit(np.log(agg.index), np.log(agg.values), 1)[0]), 3)
        print(f"    fitted exponent (pairwise stage time ~ n^b): b = {exponent}")

    # 3. Query latency.
    print(f"\n[3] Query latency, {args.repeats} repeats x {len(VARIANTS)} methods ...")
    cpg = ComparableGraph.load()
    queries = load_queries()
    lat = query_latency(catalogue, queries, market, cpg, args.repeats)
    lat.to_csv(OUT_DIR / "query_latency.csv", index=False)
    latency_summary = {}
    for method, g in lat.groupby("method"):
        ne = g[g["eligible_count"] > 0]
        latency_summary[method] = {
            "non_empty_queries": {c: describe(ne[c]) for c in
                                  ("hard_filter_ms", "direct_score_ms", "graph_and_ranking_ms", "total_ms")},
            "empty_queries_total_ms": describe(g.loc[g["eligible_count"] == 0, "total_ms"]),
        }
        t = latency_summary[method]["non_empty_queries"]["total_ms"]
        print(f"    {method:<11} total per query: mean {t['mean']:.1f} ms | median {t['median']:.1f} | "
              f"p95 {t['p95']:.1f} | max {t['max']:.1f}")
    corr = lat[(lat["method"] == "cpg") & (lat["eligible_count"] > 0)][["eligible_count", "total_ms"]].corr().iloc[0, 1]

    # 4. LightGCN timing (from its training log, if recorded).
    lightgcn = None
    if LIGHTGCN_LOG.exists():
        log = pd.read_csv(LIGHTGCN_LOG)
        cols = [c for c in ("training_seconds", "seconds_per_epoch", "inference_ms_per_query") if c in log]
        lightgcn = {c: describe(log[c]) for c in cols} if cols else \
            "no timing columns - rerun baseline_models/lightgcn.py to record training/inference time"

    summary = {
        "environment": env,
        "graph_construction": {
            "properties": int(len(catalogue)), "unique_pairs": int(len(catalogue) * (len(catalogue) - 1) // 2),
            "similarity_components": len(COMPONENTS), "similarity_config_id": cfg.config_id(),
            "edges": int(timing["edges"].iloc[0]),
            "total_seconds": describe(timing["total_s"]),
            "pairwise_topk_seconds": describe(timing["pairwise_topk_s"]),
            "retention_seconds": describe(timing["retention_s"]),
            "graph_object_seconds": describe(timing["graph_object_s"]),
            "pairs_per_second": describe(timing["pairs_per_second"]),
            "peak_traced_memory_mb": build_df.loc[build_df["kind"] == "memory", "peak_traced_memory_mb"].iloc[0],
        },
        "scaling": None if scaling_df.empty else {
            "sizes": sorted(scaling_df["n"].unique().tolist()),
            "median_total_seconds_by_n": scaling_df.groupby("n")["total_s"].median().round(3).to_dict(),
            "fitted_exponent_pairwise_stage": exponent},
        "query_latency_ms": latency_summary,
        "cpg_latency_vs_eligible_count_pearson_r": round(float(corr), 3) if pd.notna(corr) else None,
        "lightgcn": lightgcn,
        "repeats": args.repeats,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT_DIR / "efficiency_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR} (environment.json, graph_build_runs.csv, scaling.csv, "
          f"query_latency.csv, efficiency_summary.json)")


if __name__ == "__main__":
    main()
