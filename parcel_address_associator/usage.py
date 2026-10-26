"""Residential-use reconciliation.

Two sources say whether an ADDR_ID is a residential dwelling: the base
ADDRESS_TYPE (RESIDENTIAL_DWELLING / SHELL_CHILD count as residential) and a
second dataset's RESIDENTIAL_DWELLING flag. Neither is trusted outright.

The decision is made per *premise*: a duplicate group (one physical place with
several ADDR_IDs) or a lone address. For each premise:

    log-odds(dwelling) = structural prior   (flat type; building/site vs standalone parent)
                       + peer consensus      (other premises in the same block)
                       + floor consensus     (other premises on the same floor of the structure)
                       + base evidence       (does the base flag any member as residential?)
                       + other evidence      (does the second dataset?)

Peer and floor terms use the current estimates of neighbouring premises, and each
source's evidence weight comes from its sensitivity/specificity, re-estimated per
context from the data (Dawid-Skene style EM). The fitted reliabilities quantify how
far each source can be trusted for units, shops, building addresses and houses.

Each ADDR_ID then gets a category explaining any disagreement, and a suggested
flag per source: a residential premise is counted once, on one representative
ADDR_ID; building/site addresses and non-residential premises are not counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import codes
from .alignment import sigmoid
from .config import Config
from .features import Feat, build_feat
from .model import Address
from .reconcile import ParcelResult

SOURCES = ("BASE", "OTHER")
CATEGORY_NOTES = {
    "OK": "sources agree with the reconciled use",
    "DUPLICATE_OF_COUNTED_DWELLING": "same dwelling as another ADDR_ID that is (or should be) the one counted",
    "BUILDING_OR_SITE_ADDRESS": "address of a building or site that holds units; not itself a dwelling",
    "MISSED_BY_OTHER": "residential premise the second dataset does not count",
    "MISSED_BY_BASE": "residential premise the base ADDRESS_TYPE does not count",
    "MISSED_BY_BOTH": "structure says residential although neither source does (check)",
    "LIKELY_NOT_RESIDENTIAL": "flagged residential, but the premise looks non-residential",
    "POSSIBLE_DUPLICATE": "may be the same dwelling as a review-candidate address; no change "
                          "suggested until the duplicate is confirmed",
    "UNRESOLVED": "sources disagree and the evidence is balanced",
}
_TRUE = {"Y", "YES", "TRUE", "T", "1", "RESIDENTIAL", "RESIDENTIAL_DWELLING"}
_FALSE = {"N", "NO", "FALSE", "F", "0", "NON_RESIDENTIAL", "NOT_RESIDENTIAL", ""}


def parse_flag(v) -> Optional[bool]:
    """Y/N, True/False, 1/0, yes/no -> bool; anything else -> None."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()):
        return {1: True, 0: False}.get(int(v))
    t = str(v).strip().upper()
    if t in _TRUE:
        return True
    if t in _FALSE and t != "":
        return False
    return None


def prepare_use(use_df: pd.DataFrame, id_col: str = "ADDR_ID",
                flag_col: str = "RESIDENTIAL_DWELLING", known_ids: Optional[Set[str]] = None
                ) -> Tuple[Dict[str, Optional[bool]], pd.DataFrame]:
    """Validate the second dataset. Returns ({ADDR_ID: flag}, issues)."""
    cols = {str(c).strip().upper(): c for c in use_df.columns}
    for c in (id_col, flag_col):
        if c.upper() not in cols:
            raise ValueError("Use dataset is missing column %s (have %s)" % (c, list(use_df.columns)))
    ids = use_df[cols[id_col.upper()]].map(lambda v: "" if v is None or (isinstance(v, float) and v != v)
                                         else (str(int(v)) if isinstance(v, float) and v.is_integer()
                                               else str(v).strip()))
    raw = use_df[cols[flag_col.upper()]]
    flags = raw.map(parse_flag)
    issues = []

    def issue(check, severity, n, detail, examples=()):
        if n:
            issues.append({"CHECK": check, "SEVERITY": severity, "ROWS": int(n), "DETAIL": detail,
                           "EXAMPLE_ADDR_IDS": ", ".join(list(examples)[:5])})

    bad = flags.isna() & raw.notna()
    issue("unparsed_flag", "warning", bad.sum(), "%s values not recognised as yes/no: %s"
          % (flag_col, ", ".join(sorted({str(x) for x in raw[bad]})[:6])), ids[bad])
    dup = ids.duplicated(keep="first")
    issue("duplicate_addr_id", "error", dup.sum(), "repeated ADDR_ID; first occurrence kept", ids[dup])
    out = {i: f for i, f, d in zip(ids, flags, dup) if i and not d}
    if known_ids is not None:
        missing = sorted(known_ids - set(out))
        extra = sorted(set(out) - known_ids)
        issue("not_in_use_dataset", "warning", len(missing),
              "ADDR_IDs in the address data but not in the use dataset (treated as unknown)", missing)
        issue("not_in_address_data", "info", len(extra),
              "ADDR_IDs in the use dataset but not in the address data (ignored)", extra)
    issue("flags_loaded", "info", len(out), "%d flags loaded (%d residential)"
          % (len(out), sum(1 for f in out.values() if f)))
    return out, pd.DataFrame(issues, columns=["CHECK", "SEVERITY", "ROWS", "DETAIL", "EXAMPLE_ADDR_IDS"])


@dataclass
class Premise:
    idx: int
    parcel: str
    group_id: str
    members: List[Feat]
    context: str
    preferred: str
    blocks: Set[Tuple[str, str]] = field(default_factory=set)
    floor: Optional[Tuple[str, str, float]] = None
    base: Optional[float] = None      # 1 if the base flags any member residential, 0 if none, None unknown
    other: Optional[float] = None
    p: float = 0.5
    terms: Dict[str, float] = field(default_factory=dict)
    n_peers: int = 0
    n_floor: int = 0
    maybe_dups: List[str] = field(default_factory=list)   # review-candidate partners outside the premise
    area: str = ""             # area pattern of its block(s): OTHER_COVERAGE_GAP, BLOCK_DISAGREEMENT...
    area_block: str = ""


@dataclass
class BlockInfo:
    parcel: str
    block_id: str
    style_id: str
    structure_id: str
    template: str
    context: str = ""
    premises: List[int] = field(default_factory=list)
    verdict: str = ""


def _context(members: List[Feat], role: str) -> str:
    subs = [f for f in members if f.is_sub]
    if not subs:
        return "building" if role in ("BUILDING", "SITE") else "standalone"
    classes = {f.fclass for f in subs}
    if classes & {codes.RESIDENTIAL, codes.ATTACHED}:
        return "unit"
    if codes.COMMERCIAL in classes:
        return "commercial"
    if codes.ANCILLARY in classes:
        return "ancillary"
    return "other_sub"


def build_premises(results: Iterable[ParcelResult], extra: List[Address], base: Dict[str, Optional[bool]],
                   other: Dict[str, Optional[bool]],
                   partners: Optional[Dict[str, Set[str]]] = None
                   ) -> Tuple[List[Premise], Dict[Tuple[str, str], BlockInfo]]:
    premises: List[Premise] = []
    blocks: Dict[Tuple[str, str], BlockInfo] = {}
    partners = partners or {}

    def vote(flags, members) -> Optional[float]:
        known = [flags.get(f.pid) for f in members if flags.get(f.pid) is not None]
        return None if not known else float(any(known))

    for r in results:
        style_tmpl = {s.id: s.template for s in r.styles}
        for b in r.blocks:
            blocks[(r.parcel, b.id)] = BlockInfo(r.parcel, b.id, b.style_id, b.structure_id,
                                                 style_tmpl.get(b.style_id, ""))
        role = {p.feat.pid: p.role for p in r.parents}
        cell = {}
        for sid, grid in r.cells.items():
            for (lv, _u), fs in grid.items():
                for f in fs:
                    cell[f.pid] = (r.parcel, sid, lv)
        grouped = set()
        units: List[Tuple[str, List[Feat], str]] = []
        for g in r.groups:
            units.append((g.id, list(g.members), g.preferred.pid))
            grouped.update(f.pid for f in g.members)
        for f in r.feats:
            if f.pid not in grouped:
                units.append(("", [f], f.pid))
        for gid, members, pref in units:
            roles = [role.get(f.pid, "") for f in members if not f.is_sub]
            r_role = "BUILDING" if any(x in ("BUILDING", "SITE") for x in roles) else "STANDALONE"
            pr = Premise(len(premises), r.parcel, gid, members, _context(members, r_role), pref)
            pr.blocks = {(r.parcel, f.block_id) for f in members if f.block_id}
            pr.floor = next((cell[f.pid] for f in members if f.pid in cell), None)
            pr.base, pr.other = vote(base, members), vote(other, members)
            pr.maybe_dups = sorted(set().union(*(partners.get(f.pid, set()) for f in members))
                                   - {f.pid for f in members})
            for key in pr.blocks:
                blocks[key].premises.append(pr.idx)
                blocks[key].context = pr.context
            premises.append(pr)
    for a in extra:                                   # parcels too small to analyse, or no parcel
        f = build_feat(a)
        pr = Premise(len(premises), a.parcel, "", [f], _context([f], "STANDALONE"), f.pid)
        pr.base, pr.other = vote(base, [f]), vote(other, [f])
        premises.append(pr)
    return premises, blocks


def find_area_patterns(premises: List[Premise], blocks: Dict[Tuple[str, str], BlockInfo],
                       cfg: Config) -> None:
    """Mark blocks where one source says residential and the other does not for
    (nearly) every premise.

    OTHER_COVERAGE_GAP / BASE_COVERAGE_GAP: the same style or structure elsewhere
    on the parcel is residential in both sources, so the silent source is most
    likely missing this block; its flags are set aside there.
    BLOCK_DISAGREEMENT: same pattern without support on the parcel; review the area.
    """
    def stats(info: BlockInfo):
        ps = [premises[i] for i in info.premises if premises[i].base is not None
              and premises[i].other is not None and not premises[i].maybe_dups]
        n = len(ps)
        both = sum(1 for p in ps if p.base == 1 and p.other == 1)
        b_only = sum(1 for p in ps if p.base == 1 and p.other == 0)
        o_only = sum(1 for p in ps if p.base == 0 and p.other == 1)
        return n, both, b_only, o_only

    st = {k: stats(v) for k, v in blocks.items()}
    for key, info in blocks.items():
        n, both, b_only, o_only = st[key]
        if n < cfg.use_gap_min_premises:
            info.verdict = "AGREE" if n and both == n else ("MIXED" if n else "")
            continue
        if b_only >= cfg.use_gap_share * n:
            silent = "OTHER"
        elif o_only >= cfg.use_gap_share * n:
            silent = "BASE"
        else:
            info.verdict = "AGREE" if both >= cfg.use_gap_share * n else "MIXED"
            continue
        support = False
        if info.context == "unit":
            for k2, i2 in blocks.items():
                if k2 == key or i2.parcel != info.parcel:
                    continue
                if not (i2.style_id == info.style_id or (i2.structure_id and i2.structure_id == info.structure_id)):
                    continue
                n2, both2, _, _ = st[k2]
                if n2 >= cfg.use_gap_min_premises and both2 >= cfg.use_gap_support_share * n2:
                    support = True
                    break
        info.verdict = "%s_COVERAGE_GAP" % silent if support else "BLOCK_DISAGREEMENT"
        for i in info.premises:
            p = premises[i]
            if not p.area or p.area == "BLOCK_DISAGREEMENT":
                p.area, p.area_block = info.verdict, info.block_id


def fit(premises: List[Premise], cfg: Config) -> pd.DataFrame:
    """Iterate premise estimates and per-context source reliabilities. Returns the
    reliability table."""
    n = len(premises)
    if not n:
        return pd.DataFrame(columns=["SOURCE", "CONTEXT", "SENSITIVITY", "SPECIFICITY",
                                     "EST_DWELLINGS", "EST_NON_DWELLINGS"])
    prior = np.array([cfg.use_prior.get(p.context, 0.0) for p in premises])
    # neighbour lists (leave-one-out), built once
    by_block: Dict[Tuple[str, str], List[int]] = {}
    by_floor: Dict[Tuple[str, str, float], List[int]] = {}
    for p in premises:
        for b in p.blocks:
            by_block.setdefault(b, []).append(p.idx)
        if p.floor is not None:
            by_floor.setdefault(p.floor, []).append(p.idx)
    peers, floors = [], []
    for p in premises:
        s = set()
        for b in p.blocks:
            s.update(by_block[b])
        s.discard(p.idx)
        peers.append(np.fromiter(s, dtype=np.int64, count=len(s)))
        fl = [i for i in by_floor.get(p.floor, []) if i != p.idx and i not in s] if p.floor else []
        floors.append(np.array(fl, dtype=np.int64))
        p.n_peers, p.n_floor = len(s), len(fl)
    contexts = sorted({p.context for p in premises})
    ctx_idx = {c: np.array([p.idx for p in premises if p.context == c]) for c in contexts}
    # premises whose disagreement may just be an unconfirmed duplicate say nothing
    # about a source's reliability: keep them out of the M-step
    learn = np.array([not p.maybe_dups for p in premises])
    def vote_of(p: Premise, s: str):
        v = p.base if s == "BASE" else p.other
        if v is None or p.area == "%s_COVERAGE_GAP" % s:     # a source's gap says nothing
            return np.nan
        return v
    votes = {s: np.array([vote_of(p, s) for p in premises]) for s in SOURCES}
    a0 = cfg.use_reliability_prior * cfg.use_reliability_strength
    b0 = (1 - cfg.use_reliability_prior) * cfg.use_reliability_strength
    rel = {(s, c): (cfg.use_reliability_prior, cfg.use_reliability_prior) for s in SOURCES for c in contexts}

    def smoothed_logit(sum_p: float, k: int, cap: float) -> float:
        if k == 0:
            return 0.0
        share = (sum_p + 1.0) / (k + 2.0)
        return max(-cap, min(cap, math.log(share / (1 - share))))

    lo_r, hi_r = cfg.use_reliability_bounds
    p_est = 1.0 / (1.0 + np.exp(-prior))
    for _ in range(cfg.use_iterations):
        # E-step. Source evidence first, then neighbours: peers contribute their own
        # local evidence (structure + sources), never their peer-boosted estimate, so a
        # block cannot reinforce itself into a wrong answer.
        src_t = {}
        for s in SOURCES:
            t = np.zeros(n)
            for c in contexts:
                sens, spec = rel[(s, c)]
                ix = ctx_idx[c]
                v = votes[s][ix]
                llr = np.where(v == 1, math.log(sens / (1 - spec)), math.log((1 - sens) / spec))
                llr = np.clip(llr, -cfg.use_source_cap, cfg.use_source_cap)
                t[ix] = np.where(np.isnan(v), 0.0, llr)
            src_t[s] = t
        local = prior + src_t["BASE"] + src_t["OTHER"]
        p_local = 1.0 / (1.0 + np.exp(-local))
        peer_t = np.array([cfg.w_use_peers * smoothed_logit(p_local[ix].sum(), len(ix), cfg.use_peers_cap)
                           for ix in peers])
        floor_t = np.array([cfg.w_use_floor * smoothed_logit(p_local[ix].sum(), len(ix), cfg.use_floor_cap)
                            for ix in floors])
        p_new = 1.0 / (1.0 + np.exp(-(local + peer_t + floor_t)))
        # M-step: re-estimate each source's reliability per context, bounded so no
        # source is fitted as worse than chance (unidentifiable where one class is rare)
        for s in SOURCES:
            for c in contexts:
                ix = ctx_idx[c]
                v = votes[s][ix]
                known = ~np.isnan(v) & learn[ix]
                pp, vv = p_new[ix][known], v[known]
                sens = (np.sum(pp * vv) + a0) / (np.sum(pp) + a0 + b0)
                spec = (np.sum((1 - pp) * (1 - vv)) + a0) / (np.sum(1 - pp) + a0 + b0)
                rel[(s, c)] = (float(min(max(sens, lo_r), hi_r)), float(min(max(spec, lo_r), hi_r)))
        converged = np.max(np.abs(p_new - p_est)) < 1e-4
        p_est = p_new
        if converged:
            break
    for p in premises:
        i = p.idx
        p.p = float(p_est[i])
        p.terms = {"prior (%s)" % p.context: float(prior[i]), "block peers": float(peer_t[i]),
                   "floor peers": float(floor_t[i]), "base": float(src_t["BASE"][i]),
                   "other": float(src_t["OTHER"][i])}
    rows = []
    for (s, c), (sens, spec) in sorted(rel.items()):
        ix = ctx_idx[c]
        rows.append({"SOURCE": s, "CONTEXT": c, "SENSITIVITY": round(sens, 3), "SPECIFICITY": round(spec, 3),
                     "EST_DWELLINGS": round(float(p_est[ix].sum()), 1),
                     "EST_NON_DWELLINGS": round(float((1 - p_est[ix]).sum()), 1)})
    return pd.DataFrame(rows)


def review_partners(results: Iterable[ParcelResult]) -> Dict[str, Set[str]]:
    """ADDR_ID -> ADDR_IDs it may duplicate (review candidates, not grouped)."""
    out: Dict[str, Set[str]] = {}
    for r in results:
        for p in r.review_pairs:
            out.setdefault(p.a.pid, set()).add(p.b.pid)
            out.setdefault(p.b.pid, set()).add(p.a.pid)
    return out


def decide(premises: List[Premise], base_type: Dict[str, str], base: Dict[str, Optional[bool]],
           other: Dict[str, Optional[bool]], cfg: Config) -> pd.DataFrame:
    """One row per ADDR_ID: flags, reconciled use, category and suggested flags."""
    rows = []
    for pr in premises:
        maybe_dups = pr.maybe_dups
        if pr.p >= cfg.use_res_threshold:
            use = "RESIDENTIAL"
        elif pr.p <= cfg.use_non_threshold:
            use = "NOT_RESIDENTIAL"
        else:
            use = "UNCERTAIN"
        # count a residential premise once, on the member the sources already count
        rep = sorted(pr.members, key=lambda f: (-(int(bool(base.get(f.pid))) + int(bool(other.get(f.pid)))),
                                                f.pid != pr.preferred, not f.is_sub, f.pid))[0]
        evidence = "; ".join("%s %+.1f" % (k, v) for k, v in pr.terms.items() if abs(v) >= 0.05)
        if pr.area.endswith("COVERAGE_GAP"):
            evidence += "; %s flags set aside (coverage gap in block %s)" % (
                pr.area.split("_")[0].lower(), pr.area_block)
        for f in pr.members:
            b, o = base.get(f.pid), other.get(f.pid)
            should = None if use == "UNCERTAIN" else (use == "RESIDENTIAL" and f.pid == rep.pid)
            ch_b = should is not None and b is not None and b != should
            ch_o = should is not None and o is not None and o != should
            flagged = bool(b) or bool(o)
            if maybe_dups and (ch_b or ch_o or (use == "UNCERTAIN" and pr.base != pr.other)):
                # the disagreement may just be the other half of an unconfirmed duplicate
                should, ch_b, ch_o = None, False, False
                cat = "POSSIBLE_DUPLICATE"
            elif use == "UNCERTAIN":
                cat = "UNRESOLVED" if (b is not None and o is not None and b != o) else "OK"
            elif not (ch_b or ch_o):
                cat = "OK"
            elif pr.context == "building" and flagged:
                cat = "BUILDING_OR_SITE_ADDRESS"
            elif use == "RESIDENTIAL" and f.pid != rep.pid and flagged:
                cat = "DUPLICATE_OF_COUNTED_DWELLING"
            elif use == "RESIDENTIAL":
                cat = "MISSED_BY_BOTH" if (ch_b and ch_o) else ("MISSED_BY_OTHER" if ch_o else "MISSED_BY_BASE")
            else:
                cat = "LIKELY_NOT_RESIDENTIAL"
            fix = []
            if ch_b:
                fix.append("base -> %s" % ("residential" if should else "not residential"))
            if ch_o:
                fix.append("other -> %s" % ("residential" if should else "not residential"))
            if cat == "DUPLICATE_OF_COUNTED_DWELLING":
                fix.append("(dwelling counted on %s)" % rep.pid)
            if cat == "UNRESOLVED":
                fix = ["review"]
            if cat == "POSSIBLE_DUPLICATE":
                fix = ["review with %s" % ", ".join(maybe_dups[:3])]
            if pr.area.endswith("COVERAGE_GAP") and (ch_b or ch_o):
                fix.append("(%s missing block %s)" % (pr.area.split("_")[0].lower(), pr.area_block))
            if pr.area == "BLOCK_DISAGREEMENT" and cat in ("UNRESOLVED", "OK"):
                fix.append("(whole block %s disagrees: review the area)" % pr.area_block)
            rows.append({
                "ADDR_ID": f.pid, "SOURCE_ID": f.a.source_id, "CADASTRE_POLYGON_PID": pr.parcel,
                "LABEL": f.label, "PREMISE_GROUP": pr.group_id, "PREMISE_SIZE": len(pr.members),
                "CONTEXT": pr.context, "BASE_ADDRESS_TYPE": base_type.get(f.pid, ""),
                "BASE_RESIDENTIAL": b, "OTHER_RESIDENTIAL": o,
                "DIRECTION": ("BOTH" if (b and o) else "BASE_ONLY" if b and not o else
                              "OTHER_ONLY" if o and not b else "NEITHER"),
                "P_DWELLING": round(pr.p, 4), "RECONCILED_USE": use,
                "COUNTED_DWELLING_ADDR_ID": rep.pid if use == "RESIDENTIAL" else "",
                "SUGGESTED_RESIDENTIAL": should,
                "CHANGE_BASE": ch_b, "CHANGE_OTHER": ch_o, "CATEGORY": cat,
                "SUGGESTED_ACTION": "; ".join(fix), "EVIDENCE": evidence,
                "AREA_PATTERN": pr.area, "AREA_BLOCK": pr.area_block,
                "N_BLOCK_PEERS": pr.n_peers, "N_FLOOR_PEERS": pr.n_floor,
            })
    return pd.DataFrame(rows)


def block_use_table(premises: List[Premise], blocks: Dict[Tuple[str, str], BlockInfo]) -> pd.DataFrame:
    """Per block (one scheme in one building): how the two sources compare, and the
    area pattern. Rows with a COVERAGE_GAP or BLOCK_DISAGREEMENT verdict are the
    areas to rectify."""
    rows = []
    for (parcel, bid), info in blocks.items():
        ps = [premises[i] for i in info.premises]
        if not ps:
            continue
        rows.append({
            "CADASTRE_POLYGON_PID": parcel, "BLOCK_ID": bid, "STRUCTURE_ID": info.structure_id,
            "STYLE_TEMPLATE": info.template, "CONTEXT": info.context, "N_PREMISES": len(ps),
            "BOTH_RESIDENTIAL": sum(1 for p in ps if p.base == 1 and p.other == 1),
            "BASE_ONLY": sum(1 for p in ps if p.base == 1 and p.other == 0),
            "OTHER_ONLY": sum(1 for p in ps if p.base == 0 and p.other == 1),
            "NEITHER": sum(1 for p in ps if p.base == 0 and p.other == 0),
            "RECONCILED_RESIDENTIAL": sum(1 for p in ps if p.p >= 0.8),
            "VERDICT": info.verdict,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rank = df["VERDICT"].map({"OTHER_COVERAGE_GAP": 0, "BASE_COVERAGE_GAP": 1, "BLOCK_DISAGREEMENT": 2,
                              "MIXED": 3}).fillna(4)
    return df.assign(_r=rank).sort_values(["_r", "N_PREMISES"], ascending=[True, False]).drop(
        columns="_r").reset_index(drop=True)


def parcel_use_summary(use: pd.DataFrame) -> pd.DataFrame:
    """Per parcel: residential counts by source vs reconciled dwellings, and categories."""
    if use.empty:
        return pd.DataFrame()
    u = use.assign(_b=use["BASE_RESIDENTIAL"] == True, _o=use["OTHER_RESIDENTIAL"] == True,  # noqa: E712
                   _d=use["COUNTED_DWELLING_ADDR_ID"] == use["ADDR_ID"],
                   _bo=use["DIRECTION"] == "BASE_ONLY", _ob=use["DIRECTION"] == "OTHER_ONLY",
                   _u=(use["RECONCILED_USE"] == "UNCERTAIN"))
    g = u.groupby("CADASTRE_POLYGON_PID")
    out = pd.DataFrame({
        "N_ADDRESSES": g.size(), "BASE_RESIDENTIAL": g["_b"].sum(), "OTHER_RESIDENTIAL": g["_o"].sum(),
        "RECONCILED_DWELLINGS": g["_d"].sum(), "UNCERTAIN_ADDRESSES": g["_u"].sum(),
        "BASE_ONLY": g["_bo"].sum(), "OTHER_ONLY": g["_ob"].sum(),
    })
    cats = pd.crosstab(u["CADASTRE_POLYGON_PID"], u["CATEGORY"])
    for c in CATEGORY_NOTES:
        if c != "OK":
            out[c] = cats[c] if c in cats.columns else 0
    out["BASE_MINUS_RECONCILED"] = out["BASE_RESIDENTIAL"] - out["RECONCILED_DWELLINGS"]
    out["OTHER_MINUS_RECONCILED"] = out["OTHER_RESIDENTIAL"] - out["RECONCILED_DWELLINGS"]
    out = out.reset_index()
    order = (out["BASE_MINUS_RECONCILED"].abs() + out["OTHER_MINUS_RECONCILED"].abs() + out["UNCERTAIN_ADDRESSES"])
    return out.loc[order.sort_values(ascending=False).index].reset_index(drop=True)
