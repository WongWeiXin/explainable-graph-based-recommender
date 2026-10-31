"""Agreement report + adjudication sheet for the double-labelled pairs (protocol REL-v2-1).

Run from the project root once both assessors have finished:
    python evaluation/build_adjudication_sheet.py

Inputs (evaluation/labelling/):
    relevance_labelling_sheet_A1_completed.xlsx   primary assessor, all pairs
    relevance_labelling_sheet_A2_completed.xlsx   second assessor, agreement subset

Outputs (evaluation/labelling/):
    agreement_report.json      linear-weighted Cohen's kappa (original labels, before adjudication),
                               exact agreement, binary (>=2) kappa, confusion matrix
    adjudication_sheet.xlsx    one row per disagreement: profile, property, A1 and A2 grades and an
                               empty "Agreed grade" column. Fill it by discussion using only the
                               information in the sheet. Write "none" if no consensus is reached
                               (the lower grade is then used, per the protocol).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "evaluation" / "labelling"
A1 = LAB_DIR / "relevance_labelling_sheet_A1_completed.xlsx"
A2 = LAB_DIR / "relevance_labelling_sheet_A2_completed.xlsx"
# Frozen A1 grades for the double-labelled pairs as they were BEFORE A1 saw A2's grades.
# Agreement is always computed from this file (protocol: kappa on the original labels).
A1_ORIGINAL = LAB_DIR / "A1_original_double_labelled.csv"
ADJ = LAB_DIR / "adjudication_sheet.xlsx"
LABEL_COL = "Relevance (0-3)"
RELEVANT_MIN = 2


def confusion_matrix(a, b, labels):
    idx = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    return cm


def cohen_kappa_score(a, b, weights=None, labels=None):
    """Cohen's kappa; weights=None (unweighted) or 'linear'. Same definition as scikit-learn."""
    a, b = list(a), list(b)
    labels = labels if labels is not None else sorted(set(a) | set(b))
    cm = confusion_matrix(a, b, labels).astype(float)
    n = cm.sum()
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    k = len(labels)
    if weights == "linear":
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
    else:
        w = 1.0 - np.eye(k)
    denom = (w * expected).sum()
    return 1.0 - (w * cm).sum() / denom if denom else 1.0


def read_sheet(path: Path) -> tuple[pd.DataFrame, dict]:
    """Rows of the Labels tab -> DataFrame (pair_id, profile, property columns, grade)."""
    ws = load_workbook(path, data_only=True)["Labels"]
    header = [c.value for c in ws[1]]
    rows, profile = [], None
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        if str(r[0]).startswith("SEARCH PROFILE"):
            profile = r[0]
            continue
        rows.append({"profile": profile, **dict(zip(header, r))})
    df = pd.DataFrame(rows)
    df["grade"] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    problems = {
        "blank": df.loc[df["grade"].isna(), "pair_id"].tolist(),
        "out_of_range": df.loc[df["grade"].notna() & ~df["grade"].isin([0, 1, 2, 3]), "pair_id"].tolist(),
    }
    return df, problems


def main() -> None:
    a1, p1 = read_sheet(A1)
    a2, p2 = read_sheet(A2)
    for name, p in (("A1", p1), ("A2", p2)):
        if p["blank"] or p["out_of_range"]:
            raise SystemExit(f"{name} sheet is incomplete or invalid: {p}")

    m = a2[["pair_id", "grade"]].merge(a1, on="pair_id", suffixes=("_A2", "_A1"))
    if len(m) != len(a2):
        raise SystemExit("Some A2 pairs are missing from the A1 sheet.")
    m = m.rename(columns={"grade_A1": "grade_A1_current"})
    if A1_ORIGINAL.exists():
        orig = pd.read_csv(A1_ORIGINAL)[["pair_id", "grade_A1_original"]]
        m = m.merge(orig, on="pair_id", how="left")
        if m["grade_A1_original"].isna().any():
            raise SystemExit("A1_original_double_labelled.csv does not cover every A2 pair.")
        m["grade_A1"] = m["grade_A1_original"].astype(int)
        a1_basis = A1_ORIGINAL.name
    else:
        # First run: freeze the current A1 grades of the double-labelled pairs as the originals.
        m["grade_A1"] = m["grade_A1_current"].astype(int)
        m[["pair_id", "grade_A1"]].rename(columns={"grade_A1": "grade_A1_original"}) \
            .assign(source="A1 sheet at first agreement run").to_csv(A1_ORIGINAL, index=False)
        a1_basis = "current A1 sheet (now frozen as " + A1_ORIGINAL.name + ")"
    revised = m["grade_A1_current"].astype(int) != m["grade_A1"]
    g1, g2 = m["grade_A1"].astype(int), m["grade_A2"].astype(int)

    labels = [0, 1, 2, 3]
    report = {
        "protocol": "REL-v2-1",
        "a1_grades_used_for_agreement": a1_basis,
        "a1_revised_after_seeing_a2": int(revised.sum()),
        "double_labelled_pairs": int(len(m)),
        "queries": int(m["profile"].nunique()),
        "kappa_linear_weighted": round(float(cohen_kappa_score(g1, g2, weights="linear", labels=labels)), 4),
        "kappa_unweighted": round(float(cohen_kappa_score(g1, g2, labels=labels)), 4),
        "kappa_binary_relevant_ge2": round(float(cohen_kappa_score(g1 >= RELEVANT_MIN, g2 >= RELEVANT_MIN)), 4),
        "exact_agreement_pct": round(float((g1 == g2).mean() * 100), 2),
        "within_one_grade_pct": round(float(((g1 - g2).abs() <= 1).mean() * 100), 2),
        "disagreements": int((g1 != g2).sum()),
        "disagreements_after_a1_revision": int((m["grade_A1_current"].astype(int) != g2).sum()),
        "after_a1_revision_NOT_independent_agreement": {
            "note": "A1's latest grades, which include revisions made after A1 saw A2's grades. Not an "
                    "inter-assessor agreement statistic; report the original-label figures above.",
            "kappa_linear_weighted": round(float(cohen_kappa_score(
                m["grade_A1_current"].astype(int), g2, weights="linear", labels=labels)), 4),
            "exact_agreement_pct": round(float((m["grade_A1_current"].astype(int) == g2).mean() * 100), 2),
        },
        "disagreements_by_gap": (g1 - g2).abs()[g1 != g2].value_counts().sort_index().astype(int).to_dict(),
        "confusion_matrix_rows_A1_cols_A2": confusion_matrix(g1, g2, labels=labels).tolist(),
        "grade_counts_A1_all_pairs": a1["grade"].astype(int).value_counts().sort_index().to_dict(),
        "grade_counts_A2": g2.value_counts().sort_index().to_dict(),
        "target_kappa": 0.60,
    }
    report["meets_target"] = report["kappa_linear_weighted"] >= report["target_kappa"]
    (LAB_DIR / "agreement_report.json").write_text(json.dumps(report, indent=2, default=int), encoding="utf-8")

    # ---- adjudication sheet (keeps anything already entered; A1 revisions that now equal A2 are
    #      pre-filled as the agreed grade, since they record the same decision)
    previous = {}
    if ADJ.exists():
        old = load_workbook(ADJ, data_only=True)["Disagreements"]
        hh = [c.value for c in old[1]]
        for r in old.iter_rows(min_row=2, values_only=True):
            if r[0]:
                previous[r[0]] = (r[hh.index("Agreed grade")], r[hh.index("Discussion note")])
    # Disagreements to adjudicate = latest A1 sheet vs A2 (A1 revisions count as already settled).
    cur = m["grade_A1_current"].astype(int)
    dis = m[cur != g2].sort_values("pair_id")
    detail_cols = [c for c in a1.columns if c not in {"profile", "pair_id", LABEL_COL, "grade",
                                                       "Comment (optional)", "Listing link"}]
    detail_cols = [c for c in detail_cols if c in m.columns]
    wb = Workbook()
    ins = wb.active
    ins.title = "How to adjudicate"
    for line in [
        "Adjudication – protocol REL-v2-1",
        "",
        "For each row, A1 and A2 gave different grades. Discuss the pair using only the profile and "
        "property information in this sheet (no recommender output).",
        "Enter the agreed grade (0-3) in the yellow column. If no consensus is reached, write 'none': "
        "the lower of the two grades is then used.",
        "Do not change A1/A2 grades; both originals are kept on record.",
        f"Linear-weighted kappa before adjudication: {report['kappa_linear_weighted']} "
        f"(target 0.60); exact agreement {report['exact_agreement_pct']}% (A1 original grades).",
        "A1 grades shown here are the latest A1 sheet; pairs A1 already revised to A2's grade are not listed.",
    ]:
        ins.append([line])
    ins["A1"].font = Font(bold=True, size=13)
    ins.column_dimensions["A"].width = 120
    for row in ins.iter_rows():
        row[0].alignment = Alignment(wrap_text=True)

    ws = wb.create_sheet("Disagreements")
    cols = ["pair_id", "Search profile", *detail_cols, "A1 grade", "A2 grade", "Agreed grade",
            "Discussion note"]
    ws.append(cols)
    widths = {"pair_id": 9, "Search profile": 60, "Property": 26, "Area": 20, "Agreed grade": 10,
              "Discussion note": 34}
    for i, c in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(c, 11)
        cell = ws.cell(1, i)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3A5F")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    yellow = PatternFill("solid", fgColor="FFF4CC")
    agreed_col = cols.index("Agreed grade") + 1
    for _, r in dis.iterrows():
        agreed, note = previous.get(r["pair_id"], (None, None))
        ws.append([r["pair_id"], str(r["profile"]).replace("SEARCH PROFILE ", ""),
                   *[r[c] for c in detail_cols], int(r["grade_A1_current"]), int(r["grade_A2"]), agreed, note])
        ws.cell(ws.max_row, agreed_col).fill = yellow
        for i in range(1, len(cols) + 1):
            ws.cell(ws.max_row, i).alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    wb.save(LAB_DIR / "adjudication_sheet.xlsx")

    print(json.dumps({k: v for k, v in report.items() if not k.startswith("confusion")}, indent=2, default=int))
    filled = sum(1 for r in wb["Disagreements"].iter_rows(min_row=2, values_only=True)
                 if r[0] and r[cols.index("Agreed grade")] not in (None, ""))
    print(f"\nAdjudication sheet: {ADJ} ({len(dis)} disagreements, {filled} already resolved, "
          f"{len(dis) - filled} to discuss)")


if __name__ == "__main__":
    main()
