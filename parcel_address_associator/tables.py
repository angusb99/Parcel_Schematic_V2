"""Turn per-parcel results into tidy DataFrames (one builder per output)."""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from . import codes
from .config import Config
from .reconcile import ParcelResult


def _lvl(v) -> str:
    return codes.level_label(v) if v is not None else ""


def _n(s) -> tuple:
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return (int(d) if d else -1, str(s))


def link_status(l, kept_ids) -> str:
    if l.id in kept_ids:
        return "AMBIGUOUS" if l.ambiguous else "KEPT"
    return "REVIEW" if l.review_logit is not None else "REJECTED"


def parcel_summary(results: Iterable[ParcelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        in_groups = sum(len(g.members) for g in r.groups)
        rows.append({
            "CADASTRE_POLYGON_PID": r.parcel, "N_ADDRESSES": len(r.feats), "N_SUB_ADDRESSES": r.n_sub,
            "N_PARENTS": len(r.parents), "N_STYLES": len(r.styles), "N_BLOCKS": len(r.blocks),
            "N_STRUCTURES": len(r.structures), "N_DUPLICATE_GROUPS": len(r.groups),
            "N_ADDRESSES_IN_GROUPS": in_groups,
            "N_REDUNDANT_ADDRESSES": in_groups - len(r.groups),
            "N_REVIEW_PAIRS": len(r.review_pairs), "REJECTED_CONFLICTS": r.rejected_conflicts,
        })
    cols = ["CADASTRE_POLYGON_PID", "N_ADDRESSES", "N_SUB_ADDRESSES", "N_PARENTS", "N_STYLES",
            "N_BLOCKS", "N_STRUCTURES", "N_DUPLICATE_GROUPS", "N_ADDRESSES_IN_GROUPS",
            "N_REDUNDANT_ADDRESSES", "N_REVIEW_PAIRS", "REJECTED_CONFLICTS"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["N_DUPLICATE_GROUPS", "N_ADDRESSES"], ascending=False).reset_index(drop=True)


def styles(results: Iterable[ParcelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for s in r.styles:
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "STYLE_ID": s.id, "TEMPLATE": s.template,
                "SIGNATURE": s.signature, "N_ADDRESSES": s.size, "N_BLOCKS": len(s.blocks),
                "BLOCKS": "; ".join("%s: %s [%s] n=%d" % (b.id, b.context_label(), b.model, b.size)
                                    for b in s.blocks),
                "EXAMPLES": " | ".join(f.label for f in s.feats[:3]),
            })
    return pd.DataFrame(rows)


def blocks(results: Iterable[ParcelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for b in r.blocks:
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "BLOCK_ID": b.id, "STYLE_ID": b.style_id,
                "CONTEXT": b.context_label(), "BUILDING_REF": b.ref[0], "FLAT_TYPE": b.ftype,
                "NUMBERING_MODEL": b.model, "MODEL_NOTE": b.describe_model(), "N_ADDRESSES": b.size,
                "LEVELS": ("%s-%s" % (_lvl(min(b.levels)), _lvl(max(b.levels)))) if b.levels else "",
                "UNITS_PER_LEVEL": b.units_per_level or None, "STRUCTURE_ID": b.structure_id,
                "SPANS_BUILDINGS": b.spanning,
                "KEY_FORMS": ",".join(k for k, ok in b.valid.items() if ok),
            })
    return pd.DataFrame(rows)


def mappings(results: Iterable[ParcelResult], cfg: Config, include_rejected: bool = False) -> pd.DataFrame:
    rows = []
    for r in results:
        kept = {l.id for l in r.links}
        source = r.scored_links if include_rejected else r.links
        for l in sorted(source, key=lambda l: -l.logit):
            status = link_status(l, kept)
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "LINK_ID": l.id,
                "BLOCK_A": l.x.id, "CONTEXT_A": l.x.context_label(),
                "BLOCK_B": l.y.id, "CONTEXT_B": l.y.context_label(),
                "TRANSFORM": l.t.describe(), "MATCHED_KEYS": l.t.m, "KEYS_A": l.t.nx, "KEYS_B": l.t.ny,
                "QUALITY": round(l.t.quality, 3), "P": round(l.p, 4), "BAND": cfg.band(l.p),
                "STATUS": status, "EVIDENCE": "; ".join(l.evidence),
            })
    return pd.DataFrame(rows)


def structures(results: Iterable[ParcelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for st in r.structures:
            grid = r.cells.get(st.id, {})
            lv = sorted({k[0] for k in grid})
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "STRUCTURE_ID": st.id, "LABEL": st.label,
                "BLOCKS": ",".join(b.id for b in st.blocks), "CLASSES": ",".join(sorted(st.classes)),
                "LEVELS": ("%s-%s" % (_lvl(lv[0]), _lvl(lv[-1]))) if lv else "",
                "N_SLOTS": len(grid), "N_SLOTS_WITH_DUPLICATES": sum(1 for fs in grid.values() if len(fs) > 1),
                "N_UNPLACED": len(r.unplaced.get(st.id, [])),
            })
    return pd.DataFrame(rows)


def addresses(results: Iterable[ParcelResult], cfg: Config) -> pd.DataFrame:
    """Every analysed address with its style, block, structure, slot and group."""
    rows = []
    for r in results:
        style_of = {s.id: s for s in r.styles}
        block_of = {b.id: b for b in r.blocks}
        role_of = {p.feat.pid: p for p in r.parents}
        group_of, pref_of = {}, {}
        for g in r.groups:
            for f in g.members:
                group_of[f.pid] = g
        cell_of = {}
        for sid, grid in r.cells.items():
            for (lv, un), fs in grid.items():
                for f in fs:
                    cell_of[f.pid] = (sid, lv, un)
        for f in r.feats:
            b = block_of.get(f.block_id)
            g = group_of.get(f.pid)
            pr = role_of.get(f.pid)
            sid, lv, un = cell_of.get(f.pid, (b.structure_id if b else "", None, ""))
            rows.append({
                "ADDR_ID": f.pid, "SOURCE_ID": f.a.source_id, "CADASTRE_POLYGON_PID": r.parcel,
                "LABEL": f.label, "KIND": "SUB" if f.is_sub else "PARENT",
                "STYLE_ID": f.style_id, "STYLE_TEMPLATE": style_of[f.style_id].template if f.style_id else "",
                "BLOCK_ID": f.block_id, "NUMBERING_MODEL": b.model if b else "",
                "STRUCTURE_ID": sid or (",".join(pr.structures) if pr else ""),
                "LEVEL": _lvl(lv), "UNIT": un, "PARENT_ROLE": pr.role if pr else "",
                "GROUP_ID": g.id if g else "",
                "IS_PREFERRED": (g.preferred.pid == f.pid) if g else None,
                "DUPLICATE_OF": g.preferred.pid if (g and g.preferred.pid != f.pid) else "",
                "GROUP_P_MIN": round(g.p_min, 4) if g else None,
                "GROUP_BAND": cfg.band(g.p_min) if g else "",
            })
    return pd.DataFrame(rows)


def duplicate_pairs(results: Iterable[ParcelResult], cfg: Config, include_review: bool = False,
                    review_only: bool = False) -> pd.DataFrame:
    rows = []
    for r in results:
        for p in sorted(r.pairs, key=lambda p: (-p.p, p.a.pid)):
            if (p.is_review and not include_review and not review_only) or (review_only and not p.is_review):
                continue
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "GROUP_ID": p.group_id,
                "ADDR_ID_A": p.a.pid, "SOURCE_ID_A": p.a.a.source_id, "LABEL_A": p.a.label,
                "STYLE_A": p.a.style_id, "BLOCK_A": p.a.block_id,
                "ADDR_ID_B": p.b.pid, "SOURCE_ID_B": p.b.a.source_id, "LABEL_B": p.b.label,
                "STYLE_B": p.b.style_id, "BLOCK_B": p.b.block_id,
                "RELATION": p.relation, "METHOD": p.method, "P": round(p.p, 4),
                "BAND": "REVIEW" if p.is_review else cfg.band(p.p),
                "EVIDENCE": "; ".join(p.evidence),
            })
    return pd.DataFrame(rows)


def duplicate_groups(results: Iterable[ParcelResult], cfg: Config) -> pd.DataFrame:
    rows = []
    for r in results:
        for g in r.groups:
            rows.append({
                "CADASTRE_POLYGON_PID": r.parcel, "GROUP_ID": g.id, "SIZE": len(g.members),
                "RELATION": g.relation, "BAND": cfg.band(g.p_min),
                "P_MIN": round(g.p_min, 4), "P_MAX": round(g.p_max, 4),
                "PREFERRED_ADDR_ID": g.preferred.pid, "PREFERRED_LABEL": g.preferred.label,
                "MEMBER_ADDR_IDS": ";".join(f.pid for f in g.members),
                "MEMBER_SOURCE_IDS": ";".join(f.a.source_id for f in g.members),
                "MEMBER_LABELS": " | ".join(f.label for f in g.members),
                "MEMBER_STYLES": ";".join(f.style_id or "parent" for f in g.members),
            })
    return pd.DataFrame(rows)
