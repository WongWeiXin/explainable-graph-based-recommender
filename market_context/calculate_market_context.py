"""Optional market-context evidence for the CPG recommender (thesis Section 4.7.5).

Run from the project root:
    python market_context/calculate_market_context.py
    python market_context/calculate_market_context.py --input outputs/hard_constraint_filter/eligible_properties.csv

What it computes for each property i (Eq. 4.18 - 4.20):
    ppsf_i = price_i / max(built_up_area_i, eps)                  asking price per sq ft
    z_i    = (ppsf_i - mu_ref) / max(sigma_ref, eps)              position in reference distribution
    M_i    = exp(-|z_i|)                                          consistency score in (0, 1]

The reference distribution comes from the dataset's transaction columns:
    transaction_price_per_sqf_mean  -> mu_ref
    transaction_price_per_sqf_std   -> sigma_ref
    transaction_price_per_sqf_q1/q3 -> IQR (eligibility check, robust variant, explanation band)

M_i measures how consistent the asking price per sq ft is with the local transaction
reference group. It is NOT a valuation, NOT proof of under/over-pricing and NOT
investment advice. When the reference evidence is not valid the score is recorded as
UNAVAILABLE (NaN + reason), never as an observed zero, so the ranker can drop the
component and renormalise the remaining weights.

Market context is query-independent, so it is computed once for the whole catalogue and
joined onto eligible candidates at ranking time (see `load_market_context`).

Outputs (outputs/market_context/):
    market_context_scores.csv   one row per property, full evidence record
    market_context_report.json  frozen configuration, availability counts, diagnostics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "market_context"

COL_ID = "Property_ID"
COL_PRICE = "price"
COL_SIZE = "built_up_area"
COL_MEAN = "transaction_price_per_sqf_mean"
COL_STD = "transaction_price_per_sqf_std"
COL_Q1 = "transaction_price_per_sqf_q1"
COL_Q3 = "transaction_price_per_sqf_q3"
REQUIRED_COLUMNS = [COL_ID, COL_PRICE, COL_SIZE, COL_MEAN, COL_STD, COL_Q1, COL_Q3]

# Status / reason codes (stable strings: the explanation engine maps these to text).
STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"
R_PRICE_INVALID = "asking_price_missing_or_non_positive"
R_SIZE_INVALID = "built_up_area_missing_or_implausible"
R_PPSF_IMPLAUSIBLE = "asking_ppsf_outside_plausibility_bounds"
R_REF_MEAN_INVALID = "reference_mean_missing_or_non_positive"
R_REF_STD_MISSING = "reference_std_missing_single_or_unknown_group"
R_REF_STD_TOO_SMALL = "reference_std_below_minimum_degenerate_group"
R_REF_IQR_DEGENERATE = "reference_iqr_zero_or_inverted"


@dataclass(frozen=True)
class MarketContextConfig:
    """Frozen market-context configuration. Change only before held-out evaluation."""

    # "mean_std" = thesis Eq. 4.19.  "robust_iqr" = (ppsf - median_proxy) / (IQR / 1.349),
    # kept as a separately reported sensitivity variant, not the primary method.
    z_method: str = "mean_std"
    eps: float = 1e-9
    # Built-up area plausibility (sq ft). Rows outside are treated as unusable size evidence.
    min_built_up_area: float = 150.0
    max_built_up_area: float = 50_000.0
    # Asking price-per-sq-ft plausibility (RM / sq ft).
    min_ppsf: float = 30.0
    max_ppsf: float = 5_000.0
    # Reference-group eligibility. The dataset has no transaction count column, so a
    # missing std (pandas std of one record) is used as the "group too small" signal and a
    # near-zero std / zero IQR as the "degenerate group" signal.
    min_reference_std: float = 5.0
    require_positive_iqr: bool = True
    # Spatial level the transaction statistics were joined at (documentation only:
    # the columns are already attached to each listing; in this dataset they vary by `area`).
    reference_level: str = "area"
    # Optional cap on |z| before scoring. None = paper formula unchanged.
    z_clip: float | None = None

    def config_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return "MC-" + hashlib.sha1(blob).hexdigest()[:10]


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input is missing market-context columns: " + ", ".join(missing)
            + "\nUse the raw-unit dataset (Malaysia_Property_Cleaned_Raw.csv), not the "
              "normalised Processed dataset, because ppsf needs RM and sq ft."
        )


def _band(ppsf: float, q1: float, q3: float) -> str | None:
    """Position of the asking ppsf relative to the reference IQR (diagnostic label)."""
    if any(not np.isfinite(v) for v in (ppsf, q1, q3)):
        return None
    if ppsf < q1:
        return "below_reference_q1"
    if ppsf > q3:
        return "above_reference_q3"
    return "within_reference_iqr"


def compute_market_context(df: pd.DataFrame, config: MarketContextConfig | None = None) -> pd.DataFrame:
    """Return one evidence row per property. Does not modify `df`."""
    cfg = config or MarketContextConfig()
    _require_columns(df)
    if cfg.z_method not in {"mean_std", "robust_iqr"}:
        raise ValueError("z_method must be 'mean_std' or 'robust_iqr'")

    price = pd.to_numeric(df[COL_PRICE], errors="coerce").to_numpy(dtype=float)
    size = pd.to_numeric(df[COL_SIZE], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(df[COL_MEAN], errors="coerce").to_numpy(dtype=float)
    sd = pd.to_numeric(df[COL_STD], errors="coerce").to_numpy(dtype=float)
    q1 = pd.to_numeric(df[COL_Q1], errors="coerce").to_numpy(dtype=float)
    q3 = pd.to_numeric(df[COL_Q3], errors="coerce").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ppsf = price / np.maximum(size, cfg.eps)

    # Reason codes evaluated in a fixed order; the first failing check is recorded.
    checks = [
        (~(np.isfinite(price) & (price > 0)), R_PRICE_INVALID),
        (~(np.isfinite(size) & (size >= cfg.min_built_up_area) & (size <= cfg.max_built_up_area)), R_SIZE_INVALID),
        (~(np.isfinite(ppsf) & (ppsf >= cfg.min_ppsf) & (ppsf <= cfg.max_ppsf)), R_PPSF_IMPLAUSIBLE),
        (~(np.isfinite(mu) & (mu > 0)), R_REF_MEAN_INVALID),
        (~np.isfinite(sd), R_REF_STD_MISSING),
        (sd < cfg.min_reference_std, R_REF_STD_TOO_SMALL),
    ]
    if cfg.require_positive_iqr:
        checks.append((~(np.isfinite(q1) & np.isfinite(q3) & (q3 > q1)), R_REF_IQR_DEGENERATE))

    reason = np.full(len(df), None, dtype=object)
    for failed, code in checks:
        failed = np.asarray(failed, dtype=bool) & (reason == None)  # noqa: E711 (elementwise)
        reason[failed] = code
    available = reason == None  # noqa: E711

    if cfg.z_method == "mean_std":
        centre, scale = mu, sd
    else:
        centre = (q1 + q3) / 2.0          # dataset has no median column; mid-hinge proxy
        scale = (q3 - q1) / 1.349         # IQR -> sigma for a normal distribution

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (ppsf - centre) / np.maximum(scale, cfg.eps)
    if cfg.z_clip is not None:
        z = np.clip(z, -cfg.z_clip, cfg.z_clip)
    score = np.exp(-np.abs(z))

    z = np.where(available, z, np.nan)
    score = np.where(available, score, np.nan)

    out = pd.DataFrame({
        COL_ID: df[COL_ID].to_numpy(),
        "asking_price": price,
        "built_up_area": size,
        "asking_ppsf": np.round(ppsf, 4),
        "ref_level": cfg.reference_level,
        "ref_ppsf_mean": mu,
        "ref_ppsf_std": sd,
        "ref_ppsf_q1": q1,
        "ref_ppsf_q3": q3,
        "z_method": cfg.z_method,
        "market_z": np.round(z, 6),
        "market_context_score": np.round(score, 6),
        "market_context_status": np.where(available, STATUS_AVAILABLE, STATUS_UNAVAILABLE),
        "market_context_unavailable_reason": reason,
        "ppsf_position_vs_reference": [
            _band(p, a, b) if ok else None for p, a, b, ok in zip(ppsf, q1, q3, available)
        ],
        "ppsf_ratio_to_reference_mean": np.where(available, np.round(ppsf / np.maximum(mu, cfg.eps), 4), np.nan),
        "market_context_config_id": cfg.config_id(),
    })
    return out


def load_market_context(candidates: pd.DataFrame, scores_csv: Path | None = None,
                        config: MarketContextConfig | None = None) -> pd.DataFrame:
    """Attach market-context evidence to an eligible-candidate table (for the ranker).

    Uses the precomputed CSV when given, otherwise computes on the fly from `candidates`.
    """
    if scores_csv is not None and Path(scores_csv).exists():
        mc = pd.read_csv(scores_csv)
    else:
        mc = compute_market_context(candidates, config)
    keep = [c for c in mc.columns if c == COL_ID or c not in candidates.columns]
    return candidates.merge(mc[keep], on=COL_ID, how="left", validate="one_to_one")


def build_report(df: pd.DataFrame, mc: pd.DataFrame, cfg: MarketContextConfig,
                 input_path: Path, elapsed: float) -> dict:
    avail = mc["market_context_status"].eq(STATUS_AVAILABLE)
    z = mc.loc[avail, "market_z"]
    s = mc.loc[avail, "market_context_score"]

    # How many distinct reference groups exist and how they map to locations.
    # ngroup(dropna=False) gives one id per reference distribution and keeps NaN std groups
    # (works the same on pandas 2.x and 3.x, unlike astype(str) + join).
    ref_cols = ["ref_ppsf_mean", "ref_ppsf_std", "ref_ppsf_q1", "ref_ppsf_q3"]
    ref_key = mc[ref_cols].round(6).groupby(ref_cols, dropna=False).ngroup()
    diagnostics = {"distinct_reference_groups": int(ref_key.nunique())}
    if "area" in df.columns:
        per_area = pd.DataFrame({"area": df["area"].to_numpy(), "k": ref_key.to_numpy()}).groupby("area")["k"].nunique()
        diagnostics["areas_with_single_reference_group_pct"] = round(float((per_area == 1).mean() * 100), 2)
    if "property_type" in df.columns:
        per_type = pd.DataFrame({"t": df["property_type"].to_numpy(), "k": ref_key.to_numpy()})
        shared = per_type.groupby("k")["t"].nunique()
        diagnostics["reference_groups_shared_by_multiple_property_types"] = int((shared > 1).sum())

    def q(series, p):
        return None if series.empty else round(float(series.quantile(p)), 4)

    return {
        "component": "market_context",
        "config_id": cfg.config_id(),
        "config": asdict(cfg),
        "input_path": str(input_path),
        "properties": int(len(mc)),
        "available": int(avail.sum()),
        "unavailable": int((~avail).sum()),
        "availability_rate": round(float(avail.mean()), 4) if len(mc) else 0.0,
        "unavailable_reasons": mc.loc[~avail, "market_context_unavailable_reason"].value_counts().to_dict(),
        "position_vs_reference": mc.loc[avail, "ppsf_position_vs_reference"].value_counts().to_dict(),
        "z_summary": {"mean": None if z.empty else round(float(z.mean()), 4), "p05": q(z, .05), "p25": q(z, .25),
                      "median": q(z, .5), "p75": q(z, .75), "p95": q(z, .95)},
        "score_summary": {"mean": None if s.empty else round(float(s.mean()), 4), "p25": q(s, .25),
                          "median": q(s, .5), "p75": q(s, .75)},
        "diagnostics": diagnostics,
        "elapsed_seconds": round(elapsed, 4),
        "limitations": [
            "Listing prices are asking prices; the reference columns summarise completed transactions.",
            "The dataset has no transaction count or period column, so minimum reference-group size and "
            "temporal alignment cannot be verified directly; missing/near-zero std and zero IQR are used as "
            "proxies for an ineligible group.",
            "Reference statistics are attached at area level and may mix property types, so a terrace house "
            "and a condominium in the same area can share one reference distribution.",
            "The candidate's own transaction cannot be excluded because listings are not linked to "
            "individual transaction records.",
            "M_i is a consistency score, not a valuation, under-pricing claim or investment advice.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute optional market-context evidence (Eq. 4.18-4.20).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--z-method", choices=["mean_std", "robust_iqr"], default="mean_std")
    p.add_argument("--min-reference-std", type=float, default=MarketContextConfig.min_reference_std)
    p.add_argument("--z-clip", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    cfg = MarketContextConfig(z_method=args.z_method, min_reference_std=args.min_reference_std,
                              z_clip=args.z_clip)
    df = pd.read_csv(args.input)
    if df[COL_ID].duplicated().any():
        raise ValueError("Property_ID must be unique before computing market context.")

    mc = compute_market_context(df, cfg)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "market_context_scores.csv"
    report_path = args.output_dir / "market_context_report.json"
    mc.to_csv(scores_path, index=False)
    report = build_report(df, mc, cfg, args.input, elapsed)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("Market-context evidence computed.")
    print(f"Config: {cfg.config_id()} ({cfg.z_method})")
    print(f"Properties: {report['properties']:,} | available: {report['available']:,} "
          f"| unavailable: {report['unavailable']:,}")
    for k, v in report["unavailable_reasons"].items():
        print(f"  - {k}: {v}")
    print(f"Median z: {report['z_summary']['median']} | median M: {report['score_summary']['median']}")
    print(f"Scores: {scores_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
