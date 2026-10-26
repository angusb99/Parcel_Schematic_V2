"""Score duplicate groups against a truth table [ADDR_ID, ENTITY_ID, (SCENARIO)].

Two addresses are true duplicates when they share an ENTITY_ID. Predictions are
scored on the pairs implied by the duplicate groups (transitive closure), which
is how the groups are consumed. Review candidates are scored separately.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from .reconcile import ParcelResult

Pair = Tuple[str, str]


def _closure(groups: Iterable[List[str]]) -> Set[Pair]:
    out = set()
    for pids in groups:
        out.update(combinations(sorted(pids), 2))
    return out


def _prf(tp: int, fp: int, fn: int) -> Dict[str, object]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"PRECISION": round(p, 4), "RECALL": round(r, 4), "F1": round(f, 4),
            "TP": tp, "FP": fp, "FN": fn}


def evaluate_use(use: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, object]:
    """Score the use reconciliation against truth [ADDR_ID, ENTITY_ID, IS_DWELLING,
    (BASE_ERROR, OTHER_ERROR)].

    * use accuracy: reconciled use of each ADDR_ID's premise vs whether it truly is a dwelling
    * counting: a source is right for an entity (physical place) when it flags exactly
      one of its ADDR_IDs if it is a dwelling, none otherwise - before and after
      applying the suggested flags
    * detection: share of injected source errors the suggestions would fix
    """
    t = truth.copy()
    t.columns = [c.upper() for c in t.columns]
    m = use.merge(t[["ADDR_ID", "ENTITY_ID", "IS_DWELLING"] +
                    [c for c in ("BASE_ERROR", "OTHER_ERROR") if c in t.columns]], on="ADDR_ID")
    decided = m[m["RECONCILED_USE"] != "UNCERTAIN"]
    acc = ((decided["RECONCILED_USE"] == "RESIDENTIAL") == decided["IS_DWELLING"]).mean()
    out = {"USE_ACCURACY_DECIDED": round(float(acc), 4) if len(decided) else None,
           "UNCERTAIN_SHARE": round(float((m["RECONCILED_USE"] == "UNCERTAIN").mean()), 4),
           "TRUE_DWELLINGS": int(m.loc[m["IS_DWELLING"], "ENTITY_ID"].nunique())}
    rows = []
    for src, col, ch in (("BASE", "BASE_RESIDENTIAL", "CHANGE_BASE"), ("OTHER", "OTHER_RESIDENTIAL", "CHANGE_OTHER")):
        before = m[col].fillna(False).astype(bool)
        after = m["SUGGESTED_RESIDENTIAL"].where(m["SUGGESTED_RESIDENTIAL"].notna(), before).astype(bool)
        ent = m.assign(_b=before, _a=after).groupby("ENTITY_ID").agg(
            dw=("IS_DWELLING", "first"), b=("_b", "sum"), a=("_a", "sum"))
        want = ent["dw"].astype(int)
        row = {"SOURCE": src, "COUNT_BEFORE": int(before.sum()), "COUNT_AFTER": int(after.sum()),
               "ENTITIES_RIGHT_BEFORE": round(float((ent["b"] == want).mean()), 4),
               "ENTITIES_RIGHT_AFTER": round(float((ent["a"] == want).mean()), 4),
               "CHANGES_SUGGESTED": int(m[ch].sum())}
        err_col = src + "_ERROR"
        if err_col in m.columns:
            # per injected error type: places (entities) carrying it that are counted right after
            right_after = (ent["a"] == want)
            for kind, sub in m[m[err_col] != ""].groupby(err_col):
                ents = sub["ENTITY_ID"].unique()
                row["RIGHT_AFTER_" + kind.upper()] = "%d/%d" % (int(right_after.loc[ents].sum()), len(ents))
        rows.append(row)
    out["by_source"] = pd.DataFrame(rows)
    out["categories"] = m.groupby("CATEGORY").agg(
        ADDR_IDS=("ADDR_ID", "size"), TRULY_DWELLING_PREMISE=("IS_DWELLING", "mean")).round(3).reset_index()
    return out


def evaluate(results: Iterable[ParcelResult], truth: pd.DataFrame) -> Dict[str, object]:
    results = list(results)
    t = truth.copy()
    t.columns = [c.upper() for c in t.columns]
    ent = dict(zip(t["ADDR_ID"].astype(str), t["ENTITY_ID"].astype(str)))
    scen = dict(zip(t["ADDR_ID"].astype(str), t["SCENARIO"])) if "SCENARIO" in t else {}
    analysed = {f.pid for r in results for f in r.feats}

    by_entity: Dict[str, List[str]] = defaultdict(list)
    for pid in analysed:
        if pid in ent:
            by_entity[ent[pid]].append(pid)
    truth_pairs = _closure(by_entity.values())
    pred_pairs = _closure([f.pid for f in g.members] for r in results for g in r.groups)
    review = {p.key for r in results for p in r.review_pairs}

    tp, fp, fn = pred_pairs & truth_pairs, pred_pairs - truth_pairs, truth_pairs - pred_pairs
    rows = []
    for k in sorted({scen.get(p[0], "?") for p in truth_pairs | pred_pairs | review}, key=str):
        sel = lambda S: {p for p in S if scen.get(p[0], "?") == k}
        m = _prf(len(sel(tp)), len(sel(fp)), len(sel(fn)))
        m["REVIEW_PAIRS"] = len(sel(review))
        m["REVIEW_TRUE"] = len(sel(review) & truth_pairs)
        m["SCENARIO"] = k
        rows.append(m)
    flagged = fn & review
    return {
        "overall": _prf(len(tp), len(fp), len(fn)),
        "review": {"PAIRS": len(review), "TRUE": len(review & truth_pairs),
                   "MISSED_TRUE_PAIRS_IN_REVIEW": len(flagged),
                   "RECALL_IF_REVIEWED": round((len(tp) + len(flagged)) / len(truth_pairs), 4)
                   if truth_pairs else None},
        "by_scenario": pd.DataFrame(rows, columns=["SCENARIO", "PRECISION", "RECALL", "F1", "TP", "FP",
                                                   "FN", "REVIEW_PAIRS", "REVIEW_TRUE"]),
        "false_positives": sorted(fp)[:50],
        "false_negatives": sorted(fn - review)[:50],
    }
