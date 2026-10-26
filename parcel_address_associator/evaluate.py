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
