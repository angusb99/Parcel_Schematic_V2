"""Learn how blocks (style x context) on a parcel map onto each other.

For every pair of class-compatible blocks we search a small family of
transforms (same identifier, level+unit decomposition with an optional level
offset, position within level, position through the building, strata-lot
order) and keep the one that explains the most records. The resulting *block
link* is scored in log-odds from contextual evidence:

  * how much of the two blocks the transform explains (coverage + layout extent)
  * building references (TOWER A vs prefix A vs BUILDING 2)
  * base-address relationship (same, range contains number, other street)
  * a prior against two blocks of the same style describing one building
  * rank-based mappings cannot be confirmed from components alone

Links that compete for the same records (e.g. two identical towers) are resolved
by exclusivity, and surviving links cluster blocks into inferred structures.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from . import codes
from .config import Config
from .features import Feat, base_relation, compare_brefs
from .schemes import Block

FORM_NAMES = {
    ("raw", "raw"): "same unit identifier",
    ("lu", "lu"): "level + unit-on-level",
    ("lr", "lr"): "level + position within level",
    ("seq", "seq"): "position through the building",
    ("seq", "num"): "unit number = position in the other scheme (e.g. strata lot order)",
    ("num", "seq"): "unit number = position in the other scheme (e.g. strata lot order)",
}
RANK_FORMS = {"seq", "num"}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass
class Transform:
    fx: str
    fy: str
    offset: float
    matched: List[Tuple[object, object]]
    nx: int
    ny: int
    quality: float
    extent_frac: float = 1.0     # share of the larger block's keys inside the smaller block's extent
    n_small: int = 0             # keys in the smaller block

    @property
    def m(self) -> int:
        return len(self.matched)

    @property
    def containment(self) -> float:
        return self.m / min(self.nx, self.ny) if min(self.nx, self.ny) else 0.0

    @property
    def rank_based(self) -> bool:
        return self.fx in RANK_FORMS or self.fy in RANK_FORMS

    def describe(self) -> str:
        s = FORM_NAMES.get((self.fx, self.fy), "%s->%s" % (self.fx, self.fy))
        if self.offset:
            s += " with level offset %+d" % self.offset
        return s


@dataclass
class BlockLink:
    x: Block
    y: Block
    t: Transform
    logit: float
    evidence: List[str]
    ambiguous: bool = False
    review_logit: Optional[float] = None   # rank mappings: score before the unconfirmed penalty
    pid_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    @property
    def p(self) -> float:
        return sigmoid(self.logit)

    @property
    def id(self) -> str:
        return "%s~%s" % (self.x.id, self.y.id)

    def other(self, b: Block) -> Block:
        return self.y if b is self.x else self.x

    def matched_pids(self, b: Block) -> Set[str]:
        side = 0 if b is self.x else 1
        return {pp[side] for pp in self.pid_pairs}

    def feat_pairs(self) -> List[Tuple[Feat, Feat]]:
        out = []
        kx, ky = self.x.keys[self.t.fx], self.y.keys[self.t.fy]
        for a, b in self.t.matched:
            for fa in kx.get(a, ()):
                for fb in ky.get(b, ()):
                    out.append((fa, fb))
        return out


@dataclass
class Structure:
    id: str
    blocks: List[Block]
    label: str = ""
    classes: Set[str] = field(default_factory=set)
    reference: Optional[Block] = None


# --------------------------------------------------------------------------
# Transform search
# --------------------------------------------------------------------------

def _numeric(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return float(digits) if digits else 0.0


def _points(keys, two_d: bool, off: float = 0.0) -> List[Tuple[float, float]]:
    """Keys as (level, unit) points, or (number, 0) for one-dimensional forms."""
    if two_d:
        return [(k[0] + off, _numeric(k[1])) for k in keys]
    return [(_numeric(k), 0.0) for k in keys]


def _bounds(pts):
    return (min(p[0] for p in pts), max(p[0] for p in pts),
            min(p[1] for p in pts), max(p[1] for p in pts))


def _iou(a, b) -> float:
    """Overlap of two inclusive integer boxes (single points have width 1)."""
    def span(lo, hi):
        return hi - lo + 1.0

    def inter(l1, h1, l2, h2):
        return max(0.0, min(h1, h2) - max(l1, l2) + 1.0)

    ix = inter(a[0], a[1], b[0], b[1]) * inter(a[2], a[3], b[2], b[3])
    union = span(a[0], a[1]) * span(a[2], a[3]) + span(b[0], b[1]) * span(b[2], b[3]) - ix
    return ix / union if union > 0 else 0.0


def best_transform(X: Block, Y: Block, cfg: Config) -> Optional[Transform]:
    """Search the transform family; quality blends how fully the smaller block is
    explained (containment) with how well the two layouts' extents agree."""
    cands: List[Transform] = []
    both_lu = X.valid["lu"] and Y.valid["lu"]

    def consider(fx: str, fy: str, off: float, matched):
        nx, ny = len(X.keys[fx]), len(Y.keys[fy])
        if not matched or not nx or not ny:
            return
        pen_key = "seqnum" if {fx, fy} == {"seq", "num"} else fx
        pen = cfg.form_penalty.get(pen_key, 0.0) + (cfg.offset_penalty if off else 0.0)
        t = Transform(fx, fy, off, matched, nx, ny, 0.0)
        if fx in ("lu", "lr"):
            px, py = _points(X.keys[fx], True, off), _points(Y.keys[fy], True)
        elif fx == "raw" and both_lu:
            px, py = _points(X.keys["lu"], True), _points(Y.keys["lu"], True)
        else:
            px, py = _points(X.keys[fx], False), _points(Y.keys[fy], False)
        ext = _iou(_bounds(px), _bounds(py))
        small, large = (px, py) if len(px) <= len(py) else (py, px)
        sb = _bounds(small)
        inside = sum(1 for p in large if sb[0] <= p[0] <= sb[1] and sb[2] <= p[1] <= sb[3])
        t.extent_frac, t.n_small = inside / len(large), len(small)
        w = cfg.w_containment
        t.quality = w * t.containment + (1 - w) * ext - pen
        cands.append(t)

    if X.valid["raw"] and Y.valid["raw"]:
        ky = Y.keys["raw"]
        consider("raw", "raw", 0, [(k, k) for k in X.keys["raw"] if k in ky])
    if both_lu:
        ky = Y.keys["lu"]
        for off in (0, -1, 1):
            m = [(k, (k[0] + off, k[1])) for k in X.keys["lu"] if (k[0] + off, k[1]) in ky]
            if off and len({k[0] for k, _ in m}) < 2:
                continue        # a level offset must hold across several levels
            consider("lu", "lu", off, m)
    if X.valid["lr"] and Y.valid["lr"]:
        # position-within-level only means something where both blocks hold the
        # same number of units on the level (complete, comparable floors)
        cx = Counter(k[0] for k in X.keys["lr"])
        cy = Counter(k[0] for k in Y.keys["lr"])
        for off in (0, -1, 1):
            ok_levels = {lv for lv, n in cx.items() if cy.get(lv + off) == n}
            if len(ok_levels) < 2:
                continue
            m = [(k, (k[0] + off, k[1])) for k in X.keys["lr"] if k[0] in ok_levels]
            consider("lr", "lr", off, m)
    # Rank-based mappings exist to tie a level-less scheme to a level-aware one;
    # they carry no information on tiny blocks or when both sides know levels.
    rank_ok = (not both_lu and min(len(X.keys["raw"]), len(Y.keys["raw"])) >= cfg.rank_min_keys)
    if rank_ok and X.valid["seq"] and Y.valid["seq"]:
        nx, ny = len(X.keys["seq"]), len(Y.keys["seq"])
        if abs(nx - ny) <= max(1, cfg.seq_size_tolerance * max(nx, ny)):
            consider("seq", "seq", 0, [(i, i) for i in X.keys["seq"] if i in Y.keys["seq"]])
    if rank_ok and X.valid["seq"] and Y.valid["num"]:
        consider("seq", "num", 0, [(i, i) for i in X.keys["seq"] if i in Y.keys["num"]])
    if rank_ok and X.valid["num"] and Y.valid["seq"]:
        consider("num", "seq", 0, [(n, n) for n in X.keys["num"] if n in Y.keys["seq"]])
    if not cands:
        return None
    return max(cands, key=lambda t: (t.quality, -cfg.form_penalty.get(t.fx, 0)))


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def score_link(X: Block, Y: Block, t: Transform, cfg: Config) -> BlockLink:
    lo = 0.0
    ev: List[str] = []

    def add(w: float, why: str):
        nonlocal lo
        lo += w
        ev.append("%s (%+.1f)" % (why, w))

    link = BlockLink(X, Y, t, 0.0, ev)
    link.pid_pairs = {(a.pid, b.pid) for a, b in link.feat_pairs()}

    if X.style_id == Y.style_id:
        add(cfg.prior_same_style, "same style on two contexts")
    q_term = cfg.w_quality * (t.quality - cfg.quality_pivot)
    add(q_term, "%s explains %d of %d/%d keys" % (t.describe(), t.m, t.nx, t.ny))
    if t.m < cfg.min_support:
        add(cfg.w_low_support * (cfg.min_support - t.m), "only %d matched" % t.m)
    if not t.rank_based and t.n_small >= cfg.extent_min_keys and t.extent_frac < 1.0:
        # If the smaller block were a sample of the larger, its keys would spread over
        # the larger one's whole layout; P(all inside a sub-extent) ~ frac ** n.
        llr = cfg.w_extent_llr * min(t.n_small, 40) * math.log(max(t.extent_frac, 1e-3))
        llr = max(-cfg.extent_llr_cap, llr)
        if llr < -0.2:
            add(llr, "smaller block spans only %.0f%% of the larger one's layout"
                % (100 * t.extent_frac))
    cc = codes.class_compatibility(X.ftype, Y.ftype)
    if cc:
        add(cc, "flat types %s/%s" % (X.ftype or "-", Y.ftype or "-"))

    br = compare_brefs(X.ref, Y.ref)
    if br == "equal":
        add(cfg.w_bref_equal, "building ref %s = %s" % (X.ref[0], Y.ref[0]))
    elif br == "ordinal":
        add(cfg.w_bref_ordinal, "building ref %s ~ %s" % (X.ref[0], Y.ref[0]))
    elif br == "conflict":
        add(cfg.w_bref_conflict, "building ref %s != %s" % (X.ref[0], Y.ref[0]))

    rel = base_relation(X.feats[0], Y.feats[0])
    if rel == "same":
        add(cfg.w_same_base, "same base address")
    elif rel == "contains":
        add(cfg.w_range_contains, "range contains number")
    elif rel == "same_street":
        add(cfg.w_same_street_other_number, "different number, same street")

    if t.rank_based:
        # rank alignment manufactures its own coverage and nothing in the
        # components can confirm it: keep it out of the groups, and offer it for
        # review only between similar-sized blocks of the same kind of premises
        sizes = sorted((len(X.keys["raw"]), len(Y.keys["raw"])))
        if (sizes[0] >= cfg.review_size_ratio * sizes[1]
                and (cc is None or cc >= -0.3) and codes.class_compatibility(X.ftype, Y.ftype) is not None):
            link.review_logit = lo
        if q_term > 0:
            add(-q_term, "rank coverage discounted")
        add(cfg.w_rank_unconfirmed, "rank mapping cannot be confirmed from components")
    link.logit = lo
    return link


# --------------------------------------------------------------------------
# Parcel-level resolution
# --------------------------------------------------------------------------

def compatible(X: Block, Y: Block) -> bool:
    if X is Y:
        return False
    if (X.fclass == codes.LEVEL_ONLY) != (Y.fclass == codes.LEVEL_ONLY):
        return False
    return codes.class_compatibility(X.ftype, Y.ftype) is not None


def link_blocks(blocks: List[Block], cfg: Config) -> Tuple[List[BlockLink], List[BlockLink]]:
    """Return (kept_links, all_scored_links)."""
    scored: List[BlockLink] = []
    for i, X in enumerate(blocks):
        for Y in blocks[i + 1:]:
            if not compatible(X, Y):
                continue
            t = best_transform(X, Y, cfg)
            if t is not None:
                scored.append(score_link(X, Y, t, cfg))
    kept = [l for l in scored if l.p >= cfg.link_threshold]

    # Exclusivity: one block's records cannot belong to two different blocks of
    # another style unless those links cover disjoint parts of it (spanning).
    by_side: Dict[Tuple[str, str], List[BlockLink]] = defaultdict(list)
    for l in kept:
        by_side[(l.x.id, l.y.style_id)].append(l)
        by_side[(l.y.id, l.x.style_id)].append(l)
    dropped: Set[str] = set()
    ambiguous: Set[str] = set()
    span_count: Dict[str, int] = defaultdict(int)
    block_by_id = {b.id: b for b in blocks}
    for (bid, _sty), links in by_side.items():
        if len(links) < 2:
            continue
        b = block_by_id[bid]
        links = sorted(links, key=lambda l: -l.logit)
        accepted = [links[0]]
        for other in links[1:]:
            if other.id in dropped:
                continue
            mine = other.matched_pids(b)
            clash = None
            for acc in accepted:
                theirs = acc.matched_pids(b)
                if mine and theirs and len(mine & theirs) / min(len(mine), len(theirs)) >= 0.2:
                    clash = acc
                    break
            if clash is None:
                accepted.append(other)
                span_count[bid] += 1
            elif clash.logit - other.logit < cfg.ambiguity_margin:
                ambiguous.update({clash.id, other.id})
                accepted.append(other)
            else:
                dropped.add(other.id)
    out = []
    for l in kept:
        if l.id in dropped:
            continue
        if l.id in ambiguous:
            l.ambiguous = True
            l.logit = logit(l.p * cfg.ambiguity_factor)
            l.evidence.append("ambiguous: competes with another block (x%.2f)" % cfg.ambiguity_factor)
        out.append(l)
    for bid, n in span_count.items():
        if n >= 1:
            block_by_id[bid].spanning = True
    return out, scored


def build_structures(blocks: List[Block], links: List[BlockLink], cfg: Config) -> List[Structure]:
    parent = {b.id: b.id for b in blocks}
    members: Dict[str, List[Block]] = {b.id: [b] for b in blocks}
    direct = {(l.x.id, l.y.id) for l in links} | {(l.y.id, l.x.id) for l in links}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_merge(r1, r2) -> bool:
        for b1 in members[r1]:
            for b2 in members[r2]:
                if b1.style_id == b2.style_id and (b1.id, b2.id) not in direct:
                    return False
        return True

    for l in sorted(links, key=lambda l: -l.logit):
        if l.p < cfg.link_threshold or l.x.spanning or l.y.spanning or l.ambiguous:
            continue
        r1, r2 = find(l.x.id), find(l.y.id)
        if r1 != r2 and can_merge(r1, r2):
            parent[r2] = r1
            members[r1].extend(members.pop(r2))

    # Commercial/level-only series sitting under a residential tower on the same
    # base and building reference belong to the same physical building.
    groups = {r: m for r, m in members.items() if not all(b.spanning for b in m) or len(m) > 1}
    sig_owner: Dict[Tuple, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for r, m in groups.items():
        for b in m:
            sig_owner[b.context_signature()][_class_family(b.fclass)].add(r)
    com_homes: Dict[str, Set[str]] = defaultdict(set)   # commercial root -> residential roots
    for sig, fam in sig_owner.items():
        res = fam.get("RES", set())
        for c in fam.get("COM", set()):
            com_homes[c] |= res
    for c, homes in com_homes.items():
        if len(homes) != 1:
            continue                                     # shared or ambiguous podium
        r1, r2 = find(next(iter(homes))), find(c)
        if r1 != r2 and r1 in members and r2 in members:
            parent[r2] = r1
            members[r1].extend(members.pop(r2))

    structures = []
    roots = [r for r in members if not (len(members[r]) == 1 and members[r][0].spanning)]
    roots.sort(key=lambda r: (-sum(b.size for b in members[r]), r))
    for i, r in enumerate(roots, 1):
        bl = members[r]
        st = Structure(id="B%d" % i, blocks=bl, classes={b.fclass for b in bl})
        res = [b for b in bl if b.fclass in (codes.RESIDENTIAL, codes.ATTACHED)] or bl
        st.reference = max(res, key=lambda b: (b.valid.get("lu", False), b.size))
        st.label = _structure_label(bl)
        for b in bl:
            b.structure_id = st.id
        structures.append(st)
    _level_shifts(structures, links)
    return structures


def _class_family(fclass: str) -> str:
    if fclass == codes.RESIDENTIAL:
        return "RES"
    if fclass in (codes.COMMERCIAL, codes.LEVEL_ONLY):
        return "COM"
    return "OTHER"


def _structure_label(blocks: List[Block]) -> str:
    refs = sorted({b.ref[0] for b in blocks if b.ref[0]})
    bases = sorted({b.base_label for b in blocks if b.base_shape == "single"})
    bits = []
    if refs:
        bits.append("/".join(refs))
    if bases:
        bits.append("; ".join(bases))
    if not bits:
        bits.append(sorted({b.base_label for b in blocks})[0])
    return " | ".join(bits)


def _level_shifts(structures: List[Structure], links: List[BlockLink]) -> None:
    """Express every block's levels in its structure's reference block frame."""
    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for l in links:
        if l.ambiguous:
            continue
        off = l.t.offset if l.t.fx in ("lu", "lr") else 0.0
        adj[l.x.id].append((l.y.id, -off))   # y level = x level + off
        adj[l.y.id].append((l.x.id, off))
    for st in structures:
        ids = {b.id for b in st.blocks}
        shift = {st.reference.id: 0.0}
        queue = [st.reference.id]
        while queue:
            cur = queue.pop(0)
            for nxt, off in adj[cur]:
                if nxt in ids and nxt not in shift:
                    shift[nxt] = shift[cur] + off
                    queue.append(nxt)
        for b in st.blocks:
            b.level_shift = shift.get(b.id, 0.0)
