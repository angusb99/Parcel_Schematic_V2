"""Record-level duplicate pairs, duplicate groups, parent (site/building)
relationships and stacking-plan placement for one parcel."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .alignment import BlockLink, Structure, build_structures, link_blocks, sigmoid
from .config import Config
from .features import Feat, base_relation, build_feat, compare_brefs, is_common_name
from .model import Address
from .schemes import Block, Style, build_styles

# Relations reported for review and never grouped
REVIEW_RELATIONS = {"AMBIGUOUS_BUILDING", "REVIEW_RENUMBERING"}


@dataclass
class PairResult:
    a: Feat
    b: Feat
    logit: float
    relation: str
    method: str
    source: str            # block link id, block id or 'parents'
    evidence: List[str]
    group_id: str = ""

    @property
    def p(self) -> float:
        return sigmoid(self.logit)

    @property
    def key(self) -> Tuple[str, str]:
        return tuple(sorted((self.a.pid, self.b.pid)))

    @property
    def is_review(self) -> bool:
        return self.relation in REVIEW_RELATIONS


@dataclass
class Group:
    id: str
    members: List[Feat]
    preferred: Feat
    relation: str
    p_min: float
    p_max: float


@dataclass
class ParentInfo:
    feat: Feat
    role: str                      # SITE | BUILDING | STANDALONE
    structures: Tuple[str, ...]


@dataclass
class ParcelResult:
    parcel: str
    feats: List[Feat]
    styles: List[Style]
    blocks: List[Block]
    links: List[BlockLink]
    scored_links: List[BlockLink]
    structures: List[Structure]
    pairs: List[PairResult]
    groups: List[Group]
    parents: List[ParentInfo]
    cells: Dict[str, Dict[Tuple[float, str], List[Feat]]] = field(default_factory=dict)
    unplaced: Dict[str, List[Feat]] = field(default_factory=dict)
    rejected_conflicts: int = 0

    @property
    def n_sub(self) -> int:
        return sum(1 for f in self.feats if f.is_sub)

    @property
    def review_pairs(self) -> List[PairResult]:
        return [p for p in self.pairs if p.is_review]


# --------------------------------------------------------------------------
# Record pairs
# --------------------------------------------------------------------------

def record_ok(fa: Feat, fb: Feat, t_form: str = "", offset: float = 0.0) -> bool:
    """Record-level veto: explicit levels that contradict the mapping, or two
    level-encoded door numbers that differ (UNIT 405 is never UNIT 505)."""
    if fa.pid == fb.pid:
        return False
    if fa.encoded_number and fb.encoded_number and fa.keys.get("raw") != fb.keys.get("raw"):
        return False
    if fa.lvl_ord is not None and fb.lvl_ord is not None and t_form in ("raw", "lu", "exact"):
        if fb.lvl_ord != fa.lvl_ord + offset:
            return False
    return True


def _add(cands: Dict[Tuple[str, str], PairResult], pr: PairResult) -> None:
    k = pr.key
    if k not in cands or pr.logit > cands[k].logit:
        cands[k] = pr


def record_pairs(blocks: List[Block], links: List[BlockLink], scored: List[BlockLink],
                 cfg: Config) -> List[PairResult]:
    cands: Dict[Tuple[str, str], PairResult] = {}
    for l in links:
        desc = l.t.describe()
        # a block that fits two buildings equally well yields review candidates
        rel = "AMBIGUOUS_BUILDING" if l.ambiguous else "SAME_DWELLING"
        for fa, fb in l.feat_pairs():
            if record_ok(fa, fb, l.t.fx, l.t.offset):
                _add(cands, PairResult(fa, fb, l.logit, rel, desc, l.id, list(l.evidence)))
    # rank mappings (e.g. strata-lot order vs level-encoded numbers) that look right
    # but cannot be confirmed from components: offer for review
    kept = {l.id for l in links}
    for l in scored:
        if l.id in kept or l.review_logit is None or sigmoid(l.review_logit) < cfg.review_min_p:
            continue
        for fa, fb in l.feat_pairs():
            if record_ok(fa, fb) and (tuple(sorted((fa.pid, fb.pid))) not in cands):
                cands[tuple(sorted((fa.pid, fb.pid)))] = PairResult(
                    fa, fb, l.review_logit, "REVIEW_RENUMBERING", l.t.describe(), l.id,
                    list(l.evidence))
    # exact duplicates inside one block (same key, different ADDR_ID)
    for b in blocks:
        form = "raw" if b.valid.get("raw") else ("lu" if b.valid.get("lu") else "raw")
        for k, fs in b.keys.get(form, {}).items():
            for i, fa in enumerate(fs):
                for fb in fs[i + 1:]:
                    if record_ok(fa, fb, "exact"):
                        _add(cands, PairResult(fa, fb, cfg.exact_duplicate_logit, "EXACT_DUPLICATE",
                                               "identical components", b.id, []))
    return list(cands.values())


# --------------------------------------------------------------------------
# Parents (addresses without a flat or level)
# --------------------------------------------------------------------------

def parent_roles(feats: List[Feat], blocks: List[Block], links: List[BlockLink]) -> List[ParentInfo]:
    """Attach each parent address to the structures whose blocks hang off its base
    address (and building name, if it has one)."""
    span_structs: Dict[str, Set[str]] = defaultdict(set)
    for l in links:
        for b in (l.x, l.y):
            o = l.other(b)
            if b.spanning and o.structure_id:
                span_structs[b.id].add(o.structure_id)

    def structs_of(b: Block) -> Set[str]:
        return {b.structure_id} if b.structure_id else set(span_structs.get(b.id, ()))

    out = []
    for p in (f for f in feats if not f.is_sub):
        cands = [b for b in blocks if b.base_key == p.base_key]
        if p.bref[0]:
            named = [b for b in cands if compare_brefs(p.bref, b.ref) in ("equal", "ordinal")]
            cands = named or [b for b in cands if compare_brefs(p.bref, b.ref) != "conflict"]
        S: Set[str] = set()
        for b in cands:
            S |= structs_of(b)
        role = "SITE" if len(S) > 1 else ("BUILDING" if S else "STANDALONE")
        out.append(ParentInfo(p, role, tuple(sorted(S))))
    return out


def parent_pairs(parents: List[ParentInfo], cfg: Config) -> List[PairResult]:
    out = []
    for i, pa in enumerate(parents):
        for pb in parents[i + 1:]:
            fa, fb = pa.feat, pb.feat
            if compare_brefs(fa.bref, fb.bref) == "conflict":
                continue
            if is_common_name(fa.bname) != is_common_name(fb.bname):
                continue                     # 'COMMON PROPERTY, 10 X ST' is not the building itself
            rel = base_relation(fa, fb)
            sa, sb = set(pa.structures), set(pb.structures)
            overlap = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            if pa.role != "STANDALONE" and pa.role == pb.role and overlap >= 0.5:
                lo = cfg.parent_same_building_logit - (0.0 if sa == sb else 0.8)
                ev = ["both %s of %s" % (pa.role.lower(), ",".join(pa.structures))]
                if rel == "same":
                    lo += cfg.parent_same_base_bonus
                    ev.append("same base")
                relation = "SAME_" + pa.role
            elif pa.role == "STANDALONE" and pb.role == "STANDALONE":
                same_no = fa.nf is not None and fa.nf == fb.nf
                lot_vs_no = "lot" in (fa.base_shape, fb.base_shape)
                if fa.street_key == fb.street_key and (same_no or lot_vs_no or rel == "contains"):
                    lo = cfg.parent_standalone_same_street_logit
                else:
                    lo = cfg.parent_standalone_other_logit
                ev = ["standalone addresses on one parcel (%s)" % rel]
                relation = "SAME_PROPERTY"
            else:
                continue
            out.append(PairResult(fa, fb, lo, relation, "parent mapping", "parents", ev))
    return out


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

def group_pairs(pairs: List[PairResult], links: List[BlockLink], styles: List[Style],
                cfg: Config) -> Tuple[List[PairResult], List[Group], int]:
    direct = {(l.x.id, l.y.id) for l in links} | {(l.y.id, l.x.id) for l in links}
    parent: Dict[str, str] = {}
    members: Dict[str, List[Feat]] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def ensure(f: Feat):
        if f.pid not in parent:
            parent[f.pid] = f.pid
            members[f.pid] = [f]

    def conflict(g1: List[Feat], g2: List[Feat]) -> bool:
        """No group may hold two different units of one scheme."""
        for u in g1:
            for v in g2:
                if u.is_sub != v.is_sub:
                    return True
                if not u.is_sub:
                    continue
                if u.block_id == v.block_id and u.keys.get("raw") != v.keys.get("raw"):
                    return True
                if (u.style_id == v.style_id and u.block_id != v.block_id
                        and (u.block_id, v.block_id) not in direct):
                    return True
        return False

    accepted: List[PairResult] = []
    rejected = 0
    for pr in sorted(pairs, key=lambda p: -p.logit):
        if pr.is_review:
            if pr.p >= cfg.band_low or pr.relation == "REVIEW_RENUMBERING":
                accepted.append(pr)          # reported, never grouped
            continue
        if pr.p < cfg.band_low:
            continue
        ensure(pr.a)
        ensure(pr.b)
        ra, rb = find(pr.a.pid), find(pr.b.pid)
        if ra == rb:
            accepted.append(pr)
            continue
        if conflict(members[ra], members[rb]):
            rejected += 1
            continue
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        accepted.append(pr)

    style_size = {s.id: s.size for s in styles}
    roots = sorted((r for r in {find(x) for x in parent} if len(members[r]) > 1),
                   key=lambda r: min(f.pid for f in members[r]))
    by_root: Dict[str, List[PairResult]] = defaultdict(list)
    for pr in accepted:
        if not pr.is_review and pr.a.pid in parent:
            by_root[find(pr.a.pid)].append(pr)
    groups: List[Group] = []
    root_gid: Dict[str, str] = {}
    for i, r in enumerate(roots, 1):
        mem = sorted(members[r], key=lambda f: f.pid)
        prs = by_root[r]
        gid = "G%d" % i
        root_gid[r] = gid
        rel = Counter(p.relation for p in prs).most_common(1)[0][0] if prs else ""
        groups.append(Group(gid, mem, preferred(mem, style_size), rel,
                            min(p.p for p in prs) if prs else 0.0,
                            max(p.p for p in prs) if prs else 0.0))
    for pr in accepted:
        if not pr.is_review and pr.a.pid in parent:
            pr.group_id = root_gid.get(find(pr.a.pid), "")
    return accepted, groups, rejected


def preferred(members: List[Feat], style_size: Dict[str, int]) -> Feat:
    """Suggest the record to keep: from the parcel's dominant scheme, then the most
    complete (explicit level, building name, single street number), then lowest id."""
    def score(f: Feat):
        richness = int(f.has_level) + int(bool(f.bname)) + int(f.base_shape == "single")
        return (style_size.get(f.style_id, 0), richness, -len(f.label))
    best = max(score(f) for f in members)
    return min((f for f in members if score(f) == best), key=lambda f: f.pid)


# --------------------------------------------------------------------------
# Stacking-plan placement
# --------------------------------------------------------------------------

def place(structures: List[Structure], groups: List[Group], blocks: List[Block]):
    by_block = {b.id: b for b in blocks}
    group_of: Dict[str, Group] = {}
    for g in groups:
        for f in g.members:
            group_of[f.pid] = g
    cells: Dict[str, Dict[Tuple[float, str], List[Feat]]] = {}
    unplaced: Dict[str, List[Feat]] = {}
    for st in structures:
        grid: Dict[Tuple[float, str], List[Feat]] = defaultdict(list)
        loose: List[Feat] = []
        ids = {b.id for b in st.blocks}
        done: Set[str] = set()
        for b in st.blocks:
            for f in b.feats:
                if f.pid in done:
                    continue
                g = group_of.get(f.pid)
                mem = [m for m in (g.members if g else [f]) if m.is_sub and m.block_id in ids]
                anchor = _anchor(mem, st, by_block)
                cell = None
                if anchor is not None and anchor.position is not None:
                    lv = anchor.position[0] + by_block[anchor.block_id].level_shift
                    cell = (lv, anchor.position[1])
                for m in mem:
                    done.add(m.pid)
                    (grid[cell] if cell is not None else loose).append(m)
        cells[st.id] = dict(grid)
        unplaced[st.id] = loose
    return cells, unplaced


def _anchor(mem: List[Feat], st: Structure, by_block) -> Optional[Feat]:
    good = ("encoded", "explicit+encoded", "explicit+per-level", "prefix-level", "level-only")
    ranked = sorted(
        [m for m in mem if m.position is not None],
        key=lambda m: (m.block_id != st.reference.id,
                       by_block[m.block_id].model not in good, m.pid))
    return ranked[0] if ranked else None


# --------------------------------------------------------------------------
# Driver for one parcel
# --------------------------------------------------------------------------

def analyse_parcel(parcel: str, addresses: List[Address], cfg: Config) -> ParcelResult:
    feats = [build_feat(a) for a in addresses]
    styles, blocks = build_styles(feats)
    links, scored = link_blocks(blocks, cfg)
    structures = build_structures(blocks, links, cfg)
    pairs = record_pairs(blocks, links, scored, cfg)
    parents = parent_roles(feats, blocks, links)
    pairs += parent_pairs(parents, cfg)
    accepted, groups, rejected = group_pairs(pairs, links, styles, cfg)
    cells, unplaced = place(structures, groups, blocks)
    return ParcelResult(parcel, feats, styles, blocks, links, scored, structures, accepted,
                        groups, parents, cells, unplaced, rejected)
