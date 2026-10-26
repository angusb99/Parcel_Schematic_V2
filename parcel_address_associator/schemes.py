"""Address styles, blocks and numbering models for one parcel.

Vocabulary
----------
style   A formatting scheme: flat type, whether a building name / level / building
        prefix is present, the numbering model, and the base-number shape
        (single, range, lot). E.g. ``UNIT {B}{L}{UU}, {N}-{N} {STREET}``.
block   The addresses of one style that hang off one context: a base address
        plus (optionally) a building reference from the name or the unit prefix.
        A block is what the scheme says is in one building.
model   How a block's unit identifiers encode position:
          encoded               1103 = level 11, unit 03 (G01 / small numbers = ground)
          explicit+encoded      'UNIT 1103, LEVEL 11'
          explicit+per-level    'UNIT 3, LEVEL 11' (numbers restart per level)
          explicit+sequential   'UNIT 78, LEVEL 11' (numbers run on through the building)
          plain                 1..N with no level information
          level-only            'LEVEL 3' with no flat
Key forms (how a block's records can be compared with another block's):
          raw    the unit identifier as written (building prefix removed)
          lu     (level, unit-on-level)
          lr     (level, position within level)
          seq    position within the whole block
          num    the bare number (plain blocks)
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import codes
from .features import Feat, building_ref

MODEL_TEMPLATES = {
    "encoded": "{L}{UU}", "explicit+encoded": "{L}{UU}", "explicit+per-level": "{U}",
    "explicit+sequential": "{N}", "plain": "{N}", "prefix-level": "{N}", "level-only": "",
}
MODEL_NOTES = {
    "encoded": "unit number encodes level (1103 = level 11, unit 3)",
    "explicit+encoded": "explicit level and level-encoded unit number",
    "explicit+per-level": "explicit level; unit numbers restart on each level",
    "explicit+sequential": "explicit level; unit numbers run on through the building",
    "plain": "plain sequence, no level information",
    "prefix-level": "level given only by a unit prefix (G01, B02)",
    "level-only": "level with no flat number",
}


@dataclass
class Block:
    id: str
    style_id: str
    feats: List[Feat]
    base_key: Tuple
    base_label: str
    base_shape: str
    street_key: str
    nf: Optional[int]
    nl: Optional[int]
    bref: Tuple[str, str]
    bname: str
    bpfx: str
    ftype: str
    fclass: str
    model: str = "plain"
    keys: Dict[str, Dict[object, List[Feat]]] = field(default_factory=dict)
    valid: Dict[str, bool] = field(default_factory=dict)
    levels: List[float] = field(default_factory=list)
    units_per_level: float = 0.0
    split_note: str = ""
    spanning: bool = False
    structure_id: str = ""
    level_shift: float = 0.0

    @property
    def size(self) -> int:
        return len(self.feats)

    @property
    def ref(self) -> Tuple[str, str]:
        """Building reference for cross-style comparison (name first, then prefix)."""
        if self.bref[0]:
            return self.bref
        if self.bpfx:
            return (self.bpfx, "number" if self.bpfx.isdigit() else "letter")
        return ("", "")

    def context_label(self) -> str:
        bits = [self.base_label]
        if self.bname:
            bits.append(self.bname)
        if self.bpfx:
            bits.append("prefix %s" % self.bpfx)
        return " | ".join(bits)

    def context_signature(self) -> Tuple:
        return (self.base_key, self.ref[0])

    def describe_model(self) -> str:
        note = MODEL_NOTES.get(self.model, self.model)
        if self.levels:
            lo, hi = codes.level_label(min(self.levels)), codes.level_label(max(self.levels))
            note += "; levels %s-%s" % (lo, hi) if lo != hi else "; level %s" % lo
            if self.units_per_level:
                note += ", ~%g per level" % round(self.units_per_level, 1)
        if self.split_note:
            note += "; " + self.split_note
        return note


@dataclass
class Style:
    id: str
    key: Tuple
    feats: List[Feat]
    blocks: List[Block] = field(default_factory=list)
    template: str = ""
    signature: str = ""

    @property
    def size(self) -> int:
        return len(self.feats)


# --------------------------------------------------------------------------
# Numbering-model detection
# --------------------------------------------------------------------------

def detect_encoded(nums: List[int]) -> bool:
    """True when numbers >= 100 look like level*100 + unit."""
    hi = [n for n in nums if n >= 100]
    lo = [n for n in nums if n < 100]
    if len(hi) < 3:
        return False
    units = [n % 100 for n in hi]
    if max(units) > 40:
        return False
    if sum(1 for u in units if u == 0) > 0.05 * len(hi):
        return False
    if lo and max(lo) >= 95 and min(hi) <= 105:          # a sequence running through 100
        return False
    groups = {n // 100 for n in hi}
    if len(groups) == 1:
        return len(lo) <= max(units)
    return True


def _encodes(num: Optional[int], lvl: Optional[float]) -> bool:
    if num is None or lvl is None:
        return False
    if lvl >= 1 and lvl == int(lvl):
        return num // 100 == int(lvl)
    if lvl == 0:
        return num < 100
    return False


def _split_mixture(feats: List[Feat]) -> Optional[Tuple[List[Feat], List[Feat], str]]:
    """Separate two numbering schemes sharing one format and base, e.g. level-encoded
    unit numbers (101..907) alongside strata-lot numbers (1..63)."""
    plain = [f for f in feats if f.num is not None and f.pfx_role != "level" and f.lvl_ord is None]
    hi = [f for f in plain if f.num >= 100]
    lo = [f for f in plain if f.num < 100]
    if not hi or not lo or not detect_encoded([f.num for f in hi]):
        return None
    per_level = Counter(f.num // 100 for f in hi)
    upl = statistics.median(per_level.values())
    lo_nums = sorted({f.num for f in lo})
    span = lo_nums[-1] - lo_nums[0] + 1
    dense = lo_nums[0] <= 0.15 * lo_nums[-1] + 2 and len(lo_nums) >= 0.5 * span
    if len(lo) > max(1.5 * upl, 8) and dense:
        lo_ids = {id(f) for f in lo}
        return ([f for f in feats if id(f) not in lo_ids], lo,
                "split from %d low numbers (1-%d) forming a second scheme" % (len(lo), lo_nums[-1]))
    return None


def _assign_levels(block: Block) -> Tuple[Dict[int, Optional[float]], Dict[int, Optional[int]]]:
    fs = block.feats
    lvl: Dict[int, Optional[float]] = {}
    unit: Dict[int, Optional[int]] = {}
    explicit = [f for f in fs if f.lvl_ord is not None]
    if all(f.num is None and f.has_level for f in fs):
        block.model = "level-only"
        for f in fs:
            lvl[id(f)], unit[id(f)] = f.lvl_ord, 0
        return lvl, unit
    if len(explicit) >= 0.6 * len(fs):
        ex_num = [f for f in explicit if f.num is not None]
        hits = sum(1 for f in ex_num if _encodes(f.num, f.lvl_ord))
        if ex_num and hits >= 0.8 * len(ex_num):
            block.model = "explicit+encoded"
            for f in fs:
                if f.num is None:
                    lvl[id(f)], unit[id(f)] = f.lvl_ord, None
                    continue
                lv = f.lvl_ord if f.lvl_ord is not None else (float(f.num // 100) if f.num >= 100 else None)
                lvl[id(f)], unit[id(f)] = lv, (f.num % 100 if f.num >= 100 else f.num)
        else:
            seen = defaultdict(set)
            for f in ex_num:
                seen[f.num].add(f.lvl_ord)
            repeats = sum(1 for s in seen.values() if len(s) > 1)
            block.model = ("explicit+per-level" if repeats >= max(1, 0.2 * len(seen))
                           else "explicit+sequential")
            for f in fs:
                lvl[id(f)], unit[id(f)] = f.lvl_ord, f.num
        return lvl, unit
    plain_nums = [f.num for f in fs if f.num is not None and f.pfx_role != "level"]
    has_level_pfx = any(f.pfx_role == "level" for f in fs)
    if detect_encoded(plain_nums):
        block.model = "encoded"
    elif has_level_pfx:
        block.model = "prefix-level"
    else:
        block.model = "plain"
    for f in fs:
        if f.pfx_role == "level" and f.num is not None:
            lvl[id(f)], unit[id(f)] = codes.LEVEL_PREFIX_ORDINAL.get(f.pfx), f.num
        elif f.lvl_ord is not None:
            lvl[id(f)], unit[id(f)] = f.lvl_ord, f.num
        elif block.model == "encoded" and f.num is not None:
            if f.num >= 100:
                lvl[id(f)], unit[id(f)] = float(f.num // 100), f.num % 100
            else:
                lvl[id(f)], unit[id(f)] = 0.0, f.num
        else:
            lvl[id(f)], unit[id(f)] = None, f.num
    return lvl, unit


def _raw_key(f: Feat) -> str:
    pfx = f.pfx if f.pfx_role != "building" else ""
    raw = pfx + (str(f.num) if f.num is not None else "") + f.sfx
    if not raw and f.has_level:
        raw = "L:" + codes.level_label(f.lvl_ord)
    return raw


def build_keys(block: Block) -> None:
    lvl, unit = _assign_levels(block)
    keys: Dict[str, Dict[object, List[Feat]]] = {k: defaultdict(list) for k in
                                                 ("raw", "lu", "lr", "seq", "num")}
    for f in block.feats:
        f.keys = {}
        raw = _raw_key(f)
        if raw:
            keys["raw"][raw].append(f)
            f.keys["raw"] = raw
        lv, un = lvl[id(f)], unit[id(f)]
        if lv is not None and (un is not None or block.model == "level-only"):
            lu = (lv, "%s%s" % ("" if un is None else un, f.sfx))
            keys["lu"][lu].append(f)
            f.keys["lu"] = lu
            f.position = (lv, lu[1])
        if block.model == "plain" and f.num is not None and not f.pfx and not f.sfx:
            keys["num"][f.num].append(f)
            f.keys["num"] = f.num

    # lr: rank of each lu key within its level
    by_level = defaultdict(list)
    for lu in keys["lu"]:
        by_level[lu[0]].append(lu)
    for lv, lus in by_level.items():
        lus.sort(key=lambda k: (_num_part(k[1]), k[1]))
        for rank, lu in enumerate(lus, 1):
            for f in keys["lu"][lu]:
                keys["lr"][(lv, rank)].append(f)
                f.keys["lr"] = (lv, rank)
    # seq: rank of each raw key through the whole block
    def seq_order(raw):
        f = keys["raw"][raw][0]
        lv = lvl[id(f)]
        return (lv if lv is not None else -1e9, f.num if f.num is not None else -1, f.sfx, raw)
    for rank, raw in enumerate(sorted(keys["raw"], key=seq_order), 1):
        for f in keys["raw"][raw]:
            keys["seq"][rank].append(f)
            f.keys["seq"] = rank

    n = len(block.feats)
    raw_levels = all(len({f.lvl_ord for f in fs if f.lvl_ord is not None}) <= 1
                     for fs in keys["raw"].values())
    raw_cov = sum(len(v) for v in keys["raw"].values()) / n
    lu_cov = sum(len(v) for v in keys["lu"].values()) / n
    block.valid = {
        "raw": raw_levels and raw_cov >= 0.9 and bool(keys["raw"]),
        "lu": lu_cov >= 0.9,
        "lr": lu_cov >= 0.9 and block.model != "level-only",
        "seq": len(keys["raw"]) >= 3,
        "num": len(keys["num"]) >= 3,
    }
    block.keys = {k: dict(v) for k, v in keys.items()}
    levels = sorted({lu[0] for lu in keys["lu"]})
    block.levels = levels
    if levels:
        counts = Counter(lu[0] for lu in keys["lu"])
        block.units_per_level = statistics.median(counts.values())


def _num_part(s: str) -> int:
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else -1


# --------------------------------------------------------------------------
# Styles and blocks for a parcel
# --------------------------------------------------------------------------

def _prefix_roles(group: List[Feat]) -> Dict[str, str]:
    by_pfx = defaultdict(list)
    for f in group:
        if f.pfx and f.num is not None:
            by_pfx[f.pfx].append(f.num)
    roles = {}
    for p, nums in by_pfx.items():
        roles[p] = "level" if (p in codes.LEVEL_PREFIX_ORDINAL and max(nums) < 100) else "building"
    return roles


def _style_key(f: Feat) -> Tuple:
    return (f.ftype or "-", "LVL" if f.has_level else "", "BN" if f.bname else "", f.base_shape)


def _name_pattern(name: str) -> str:
    ref, kind = building_ref(name)
    if not ref:
        return "{BUILDING NAME}"
    if kind in ("letter", "number"):
        return name[:-len(ref)] + "{B}" if name.endswith(ref) else name.replace(ref, "{B}", 1)
    if name.endswith(" " + ref) or name == ref or name == "THE " + ref:
        return "{BUILDING NAME}"
    return name.replace(ref, "{B}", 1)


def _template(style: Style) -> Tuple[str, str]:
    fs = style.feats
    f0 = fs[0]
    parts, sig = [], [f0.ftype or "-"]
    if f0.bname:
        pats = {_name_pattern(f.bname) for f in fs}
        parts.append(pats.pop() if len(pats) == 1 else "{BUILDING NAME}")
    model = Counter(b.model for b in style.blocks).most_common(1)[0][0]
    has_flat = f0.ftype or any(f.num is not None for f in fs)
    if has_flat:
        idp = MODEL_TEMPLATES.get(model, "{N}")
        if any(f.pfx_role == "building" for f in fs):
            idp = "{B}" + idp
            sig.append("BPFX")
        level_pfx = sorted({f.pfx for f in fs if f.pfx_role == "level"})
        if level_pfx:
            idp += " | " + " | ".join("%s{UU}" % p for p in level_pfx)
        name = codes.FLAT_TYPE_AUT.get(f0.ftype, f0.ftype) if f0.ftype else ""
        parts.append(("%s %s" % (name, idp)).strip())
    sig.append(model)
    if f0.has_level:
        names = []
        for lt, _ in Counter(f.lvl_type for f in fs).most_common():
            nm = codes.LEVEL_TYPE_AUT.get(lt, lt or "LEVEL")
            names.append(nm if lt in codes.LEVEL_TYPES_WITHOUT_NUMBER else nm + " {L}")
        parts.append("|".join(names))
        sig.append("LVL")
    if f0.bname:
        sig.append("BN")
    base = {"range": "{N}-{N} {STREET}", "single": "{N} {STREET}",
            "lot": "LOT {N} {STREET}", "none": "{STREET}"}[f0.base_shape]
    parts.append(base)
    sig.append(f0.base_shape.upper())
    return ", ".join(parts), "|".join(sig)


def build_styles(feats: List[Feat]) -> Tuple[List[Style], List[Block]]:
    subs = [f for f in feats if f.is_sub]
    coarse = defaultdict(list)
    for f in subs:
        coarse[_style_key(f)].append(f)
    for group in coarse.values():
        roles = _prefix_roles(group)
        for f in group:
            f.pfx_role = roles.get(f.pfx, "") if f.pfx else ""
    refined = defaultdict(list)
    for ck, group in coarse.items():
        for f in group:
            refined[ck + (("BPFX",) if f.pfx_role == "building" else ())].append(f)

    ordered = sorted(refined.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    styles: List[Style] = []
    blocks: List[Block] = []
    for si, (skey, sfeats) in enumerate(ordered, 1):
        style = Style(id="S%d" % si, key=skey, feats=sfeats)
        groups = defaultdict(list)
        for f in sfeats:
            groups[(f.base_key, f.bname, f.pfx if f.pfx_role == "building" else "")].append(f)
        gorder = sorted(groups.items(), key=lambda kv: (kv[1][0].base_label, kv[0][1], kv[0][2]))
        for bi, ((bkey, bname, bpfx), bfeats) in enumerate(gorder, 1):
            parts = [(bfeats, "")]
            split = _split_mixture(bfeats)
            if split:
                parts = [(split[0], ""), (split[1], split[2])]
            for pi, (pfeats, note) in enumerate(parts):
                f0 = pfeats[0]
                bid = "%s.%d%s" % (style.id, bi, "" if len(parts) == 1 else "ab"[pi])
                b = Block(id=bid, style_id=style.id, feats=pfeats, base_key=bkey,
                          base_label=f0.base_label, base_shape=f0.base_shape,
                          street_key=f0.street_key, nf=f0.nf, nl=f0.nl,
                          bref=building_ref(bname), bname=bname, bpfx=bpfx, ftype=f0.ftype,
                          fclass=f0.fclass, split_note=note)
                build_keys(b)
                for f in pfeats:
                    f.style_id, f.block_id = style.id, bid
                style.blocks.append(b)
                blocks.append(b)
        style.template, style.signature = _template(style)
        styles.append(style)
    return styles, blocks
