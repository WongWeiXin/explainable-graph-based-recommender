"""Stage-3 near-duplicate handling and state-name standardisation (thesis Section 4.5.3).

Called from data_preprocessing.py AFTER Property_ID is assigned and BEFORE the
train/validation/test split, so:
  * surviving properties keep the Property_ID they already had, and
  * copies of one property cannot land in different partitions.

Stages 1 and 2 (exact duplicate rows, repeated original_link) stay in data_preprocessing.py.

Stage 3 rule (deterministic):
  1. Candidate pair: same property_type, price, built_up_area, bedroom and bathroom AND the
     same location (coordinates rounded to COORD_DECIMALS, or same area + district when
     coordinates are missing).
  2. Confirmed duplicate: a candidate pair that ALSO shares at least one listing signal:
       - same normalised description text
       - same agent phone number found in the description
       - same agent name in the listing URL ("...-for-sale-by-<agent>")
     Attribute similarity alone is NOT treated as duplication, because different units in
     the same development can legitimately share size, rooms and price.
  3. Confirmed pairs are merged into groups (connected components). One record per group is
     kept: the most complete record, then the lowest Property_ID.
  4. Candidate pairs without any signal are kept and recorded as "flagged_unconfirmed".

Every decision is written to an audit CSV; nothing is removed silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COORD_DECIMALS = 4  # ~11 m
MATCH_ATTRIBUTES = ["property_type", "price", "built_up_area", "bedroom", "bathroom"]
COMPLETENESS_COLUMNS = [
    "price", "built_up_area", "bedroom", "bathroom", "car_parks", "land_area", "furnishing",
    "tenure", "land_title", "title_type", "unit_type", "lot_type", "facing_direction",
    "occupancy", "original_address", "road_name", "postcode", "description", "images",
]

# Canonical names follow the frozen benchmark (Johor, Melaka, Pulau Pinang, Selangor,
# Wilayah Persekutuan Kuala Lumpur). Unlisted values are left unchanged.
STATE_MAP = {
    "johor darul ta'zim": "Johor",
    "johor darul takzim": "Johor",
    "johor": "Johor",
    "malacca": "Melaka",
    "melaka": "Melaka",
    "penang": "Pulau Pinang",
    "pulau pinang": "Pulau Pinang",
    "federal territory of kuala lumpur": "Wilayah Persekutuan Kuala Lumpur",
    "kuala lumpur": "Wilayah Persekutuan Kuala Lumpur",
    "wilayah persekutuan kuala lumpur": "Wilayah Persekutuan Kuala Lumpur",
    "putrajaya": "Wilayah Persekutuan Putrajaya",
    "federal territory of putrajaya": "Wilayah Persekutuan Putrajaya",
    "wilayah persekutuan putrajaya": "Wilayah Persekutuan Putrajaya",
    "paka": "Terengganu",  # Paka is a town in Terengganu, mis-recorded as a state
}

_PHONE_RE = re.compile(r"(?:\+?6)?(01\d{8,9})")
_AGENT_RE = re.compile(r"-for-(?:sale|rent)-by-([a-z0-9-]+?)/?$")


# ------------------------------------------------------------------ state names
def standardise_states(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map state spelling variants to one canonical name. Returns (df, change counts)."""
    if "state" not in df.columns:
        return df, {}
    before = df["state"].astype("string")
    key = before.str.strip().str.casefold()
    after = key.map(STATE_MAP).fillna(before.str.strip())
    changed = before.ne(after) & before.notna()
    counts = (pd.DataFrame({"from": before[changed], "to": after[changed]})
              .value_counts().reset_index(name="rows"))
    out = df.copy()
    out["state"] = after.astype(object)
    return out, {f"{r['from']} -> {r['to']}": int(r["rows"]) for _, r in counts.iterrows()}


# ------------------------------------------------------------------ signals
def _norm_description(s: pd.Series) -> pd.Series:
    d = s.fillna("").astype(str).str.casefold().str.replace(r"\W+", "", regex=True)
    return d.mask(d.str.len() < 30)  # very short/empty text is not a usable signal


def _phones(s: pd.Series) -> pd.Series:
    """Set of mobile numbers mentioned in the description (spaces/dashes ignored)."""
    compact = s.fillna("").astype(str).str.replace(r"[\s\-().]", "", regex=True)
    return compact.map(lambda t: frozenset(_PHONE_RE.findall(t)))


def _agent(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.casefold().str.extract(_AGENT_RE)[0]


def _location_key(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df.get("latitude"), errors="coerce").round(COORD_DECIMALS)
    lon = pd.to_numeric(df.get("longitude"), errors="coerce").round(COORD_DECIMALS)
    coord = "xy:" + lat.astype(str) + "," + lon.astype(str)
    area = df.get("area", pd.Series("", index=df.index)).fillna("").astype(str).str.casefold().str.strip()
    dist = df.get("district", pd.Series("", index=df.index)).fillna("").astype(str).str.casefold().str.strip()
    fallback = "ad:" + area + "|" + dist
    return coord.where(lat.notna() & lon.notna(), fallback)


# ------------------------------------------------------------------ main step
def remove_near_duplicates(df: pd.DataFrame, audit_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (deduplicated df, summary). Requires Property_ID."""
    if "Property_ID" not in df.columns:
        raise ValueError("Assign Property_ID before near-duplicate handling.")
    work = df.reset_index(drop=True)
    pid = work["Property_ID"].to_numpy()

    block = work[MATCH_ATTRIBUTES].astype(object).fillna("NA").astype(str).agg("|".join, axis=1) + "#" + _location_key(work)
    desc = _norm_description(work.get("description", pd.Series(index=work.index, dtype=object)))
    phones = _phones(work.get("description", pd.Series(index=work.index, dtype=object)))
    agent = _agent(work.get("original_link", pd.Series(index=work.index, dtype=object)))
    filled = work.reindex(columns=COMPLETENESS_COLUMNS).notna().sum(axis=1).to_numpy()

    # Union-find over confirmed pairs.
    parent = list(range(len(work)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pair_rows = []
    for _, idx in block.groupby(block).groups.items():
        idx = sorted(idx, key=lambda k: pid[k])
        if len(idx) < 2:
            continue
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                signals = []
                if pd.notna(desc[a]) and desc[a] == desc[b]:
                    signals.append("same_description")
                if phones[a] and phones[a] & phones[b]:
                    signals.append("same_agent_phone")
                if pd.notna(agent[a]) and agent[a] == agent[b]:
                    signals.append("same_agent_in_url")
                confirmed = bool(signals)
                if confirmed:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                pair_rows.append({
                    "Property_ID_a": pid[a], "Property_ID_b": pid[b],
                    "match_block": block[a], "signals": "|".join(signals),
                    "decision": "confirmed_duplicate" if confirmed else "flagged_unconfirmed",
                })

    roots = np.array([find(k) for k in range(len(work))])
    group_size = pd.Series(roots).map(pd.Series(roots).value_counts()).to_numpy()
    in_group = group_size > 1

    # Keeper per group: most complete, then lowest Property_ID.
    order = pd.DataFrame({"root": roots, "filled": filled, "pid": pid})
    keepers = (order[in_group].sort_values(["root", "filled", "pid"], ascending=[True, False, True])
               .drop_duplicates("root").set_index("root")["pid"])
    keep_pid = np.where(in_group, pd.Series(roots).map(keepers).to_numpy(), pid)
    remove = in_group & (keep_pid != pid)

    group_id = np.where(in_group, pd.Series(roots).map(
        {r: f"DG{n:04d}" for n, r in enumerate(sorted(set(roots[in_group]), key=lambda r: pid[r]), 1)}
    ).to_numpy(), None)

    flagged_pids = set()
    for r in pair_rows:
        if r["decision"] == "flagged_unconfirmed":
            flagged_pids.update([r["Property_ID_a"], r["Property_ID_b"]])

    audit = pd.DataFrame({
        "Property_ID": pid,
        "duplicate_group": group_id,
        "group_size": np.where(in_group, group_size, 1),
        "kept_Property_ID": keep_pid,
        "decision": np.where(remove, "removed_duplicate",
                    np.where(in_group, "kept_group_representative",
                    np.where(np.isin(pid, list(flagged_pids)), "kept_flagged_unconfirmed", "kept"))),
        "completeness": filled,
        "original_link": work.get("original_link"),
    })
    audit = audit[audit["decision"] != "kept"]
    pairs = pd.DataFrame(pair_rows)

    summary = {
        "rule": "same type/price/size/bed/bath + same location (coords rounded to "
                f"{COORD_DECIMALS} dp, else area+district) + >=1 signal "
                "(same description | same agent phone | same agent in URL)",
        "rows_in": int(len(work)),
        "candidate_pairs": int(len(pairs)),
        "confirmed_pairs": int((pairs["decision"] == "confirmed_duplicate").sum()) if len(pairs) else 0,
        "unconfirmed_pairs_kept": int((pairs["decision"] == "flagged_unconfirmed").sum()) if len(pairs) else 0,
        "duplicate_groups": int(len(keepers)),
        "rows_removed": int(remove.sum()),
        "rows_out": int((~remove).sum()),
        "signal_counts": {} if pairs.empty else
            pairs.loc[pairs["signals"] != "", "signals"].str.split("|").explode().value_counts().to_dict(),
    }

    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_dir / "duplicate_audit_properties.csv", index=False)
        pairs.to_csv(audit_dir / "duplicate_audit_pairs.csv", index=False)
        (audit_dir / "duplicate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return work.loc[~remove].reset_index(drop=True), summary
