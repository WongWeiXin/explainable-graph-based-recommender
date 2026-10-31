"""Freeze the final relevance labels (protocol REL-v2-1).

Run from the project root AFTER the adjudication sheet is filled in:
    python evaluation/finalize_labels.py

Rule
    double-labelled pairs:  A1 == A2          -> that grade
                            A1 != A2          -> "Agreed grade" from adjudication_sheet.xlsx
                                                 ("none" = no consensus -> the lower grade)
    all other pairs:        A1 grade

Output (evaluation/labelling/):
    relevance_labels_v2_final.csv   pair_id, query_id, Property_ID, partition, relevance,
                                    grade_A1, grade_A2, resolution, protocol
Refuses to overwrite an existing final file unless --force (labels are frozen once written).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from build_adjudication_sheet import A1, A1_ORIGINAL, A2, ADJ, LAB_DIR, read_sheet

KEY = LAB_DIR / "pool_key_DO_NOT_SHOW_ASSESSOR.csv"
OUT = LAB_DIR / "relevance_labels_v2_final.csv"


def read_adjudication() -> dict:
    ws = load_workbook(ADJ, data_only=True)["Disagreements"]
    header = [c.value for c in ws[1]]
    pi, ai = header.index("pair_id"), header.index("Agreed grade")
    out, missing, invalid = {}, [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[pi]:
            continue
        v = r[ai]
        if v is None or str(v).strip() == "":
            missing.append(r[pi])
        elif str(v).strip().lower() == "none":
            out[r[pi]] = "none"
        else:
            try:
                g = int(float(v))
            except ValueError:
                invalid.append(r[pi])
                continue
            if g not in (0, 1, 2, 3):
                invalid.append(r[pi])
            out[r[pi]] = g
    if missing or invalid:
        raise SystemExit(f"Adjudication incomplete. Missing: {missing[:10]}{'...' if len(missing) > 10 else ''} "
                         f"({len(missing)}); invalid: {invalid}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze final relevance labels.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.name} already exists (labels are frozen). Use --force only to rebuild it.")

    a1, _ = read_sheet(A1)
    a2, _ = read_sheet(A2)
    adj = read_adjudication()
    key = pd.read_csv(KEY)[["pair_id", "query_id", "Property_ID", "partition"]]

    lab = key.merge(a1[["pair_id", "grade"]].rename(columns={"grade": "grade_A1"}), on="pair_id", how="left") \
             .merge(a2[["pair_id", "grade"]].rename(columns={"grade": "grade_A2"}), on="pair_id", how="left")
    # Keep A1's original grade (before seeing A2) on record; resolution uses the latest A1 sheet.
    if not A1_ORIGINAL.exists():
        raise SystemExit("Run build_adjudication_sheet.py first (it freezes A1's original grades).")
    orig = pd.read_csv(A1_ORIGINAL).set_index("pair_id")["grade_A1_original"]
    lab["grade_A1_original"] = lab["pair_id"].map(orig).astype("Int64")
    if lab["grade_A1"].isna().any():
        raise SystemExit(f"{int(lab['grade_A1'].isna().sum())} pool pairs have no A1 grade.")

    rel, how = [], []
    for _, r in lab.iterrows():
        if pd.isna(r["grade_A2"]):
            rel.append(int(r["grade_A1"])); how.append("A1_only")
        elif r["grade_A1"] == r["grade_A2"]:
            rel.append(int(r["grade_A1"]))
            revised = pd.notna(r["grade_A1_original"]) and r["grade_A1_original"] != r["grade_A1"]
            how.append("A1_revised_to_A2" if revised else "A1_A2_agree")
        else:
            if r["pair_id"] not in adj:
                raise SystemExit(f"Disagreement {r['pair_id']} is not in the adjudication sheet. "
                                 "Rebuild it with build_adjudication_sheet.py.")
            v = adj[r["pair_id"]]
            if v == "none":
                rel.append(int(min(r["grade_A1"], r["grade_A2"]))); how.append("no_consensus_lower")
            else:
                rel.append(int(v)); how.append("consensus")
    lab["relevance"] = rel
    lab["resolution"] = how
    lab["protocol"] = "REL-v2-1"
    lab["grade_A1"] = lab["grade_A1"].astype(int)
    lab["grade_A2"] = lab["grade_A2"].astype("Int64")
    lab = lab[["pair_id", "query_id", "Property_ID", "partition", "relevance", "grade_A1",
               "grade_A1_original", "grade_A2", "resolution", "protocol"]]
    lab.to_csv(OUT, index=False)
    print(f"Frozen: {OUT} ({len(lab)} pairs)")
    print("Resolution:", lab["resolution"].value_counts().to_dict())
    print("Final grades:", lab["relevance"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
