"""Per-address features derived from the address components."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from . import codes
from .model import Address

_BWORD = r"(?:TOWER|TWR|BUILDING|BLDG|BLOCK|BLK|WING|STAGE|PHASE|CORE|RESIDENCES?|HOUSE|COURT)"
_BREF_AFTER = re.compile(r"^(?:THE\s+)?%s\s+([A-Z]{1,2}|\d{1,3})$" % _BWORD)
_BREF_BEFORE = re.compile(r"^(?:THE\s+)?([A-Z]{1,2}|\d{1,3})\s+%s$" % _BWORD)
_BREF_NAMED = re.compile(r"^(?:THE\s+)?([A-Z][A-Z ]*?)\s+%s$" % _BWORD)
_INT = re.compile(r"^\d+$")
# building names that denote common property or a shared facility, not a building
COMMON_WORDS = re.compile(
    r"\b(COMMON (PROPERTY|AREAS?)|ASSOCIATION PROPERTY|CAR ?PARK|PARKING|PLANT ROOM|SUBSTATION|"
    r"SWIMMING POOL|POOL HOUSE|GYM(NASIUM)?|RECREATION|COMMUNITY (CENTRE|HALL|ROOM)|CLUB ?HOUSE|"
    r"BUILDING MANAGEMENT|STRATA|OWNERS CORP\w*|BODY CORP\w*|CONCIERGE|RECEPTION|LOBBY|BASEMENT|"
    r"MAIL ?ROOM|BIN (ROOM|STORE)|PUMP (ROOM|STATION))\b")


def is_common_name(name: str) -> bool:
    return bool(name) and COMMON_WORDS.search(name) is not None


def to_int(v: str) -> Optional[int]:
    v = (v or "").strip()
    return int(v) if _INT.match(v) else None


def building_ref(name: str) -> Tuple[str, str]:
    """Reduce a building name to a comparable reference and its kind.

    'TOWER A' -> ('A', 'letter'); 'BUILDING 2' -> ('2', 'number');
    'EAST TOWER' -> ('EAST', 'name'); 'THE ARCADIA' -> ('ARCADIA', 'name').
    """
    n = re.sub(r"\s+", " ", (name or "").upper()).strip()
    if not n:
        return "", ""
    for rx in (_BREF_AFTER, _BREF_BEFORE):
        m = rx.match(n)
        if m:
            ref = m.group(1)
            return ref, ("number" if ref.isdigit() else "letter")
    m = _BREF_NAMED.match(n)
    if m:
        return m.group(1), "name"
    return re.sub(r"^THE\s+", "", n), "name"


def compare_brefs(a: Tuple[str, str], b: Tuple[str, str]) -> str:
    """'equal' | 'ordinal' | 'conflict' | 'unknown'."""
    (ra, ka), (rb, kb) = a, b
    if not ra or not rb:
        return "unknown"
    if ka == kb:
        return "equal" if ra == rb else "conflict"
    if {ka, kb} == {"letter", "number"}:
        letter = ra if ka == "letter" else rb
        number = rb if ka == "letter" else ra
        if len(letter) == 1 and str(ord(letter) - 64) == number:
            return "ordinal"
        return "unknown"
    return "unknown"


@dataclass
class Feat:
    a: Address
    pid: str
    is_sub: bool
    ftype: str
    fclass: str
    pfx: str
    num: Optional[int]
    sfx: str
    lvl_type: str
    lvl_num: Optional[int]
    lvl_ord: Optional[float]
    bname: str
    bref: Tuple[str, str]
    street_key: str
    street_label: str
    nf: Optional[int]
    nf_sfx: str
    nl: Optional[int]
    lot: str
    base_shape: str
    base_key: Tuple
    base_label: str
    # filled in by scheme analysis
    pfx_role: str = ""            # '' | 'building' | 'level'
    style_id: str = ""
    block_id: str = ""
    keys: Dict[str, object] = field(default_factory=dict)
    position: Optional[Tuple[float, str]] = None   # (level ordinal, unit label) for schematics
    encoded_number: bool = False  # level-encoded door number (e.g. 405) in a numbered scheme

    @property
    def label(self) -> str:
        return self.a.short_label()

    @property
    def has_level(self) -> bool:
        return bool(self.lvl_type or self.lvl_num is not None)


def build_feat(a: Address) -> Feat:
    num = to_int(a.flat_number)
    lvl_num = to_int(a.level_number)
    lvl_ord = None
    if a.level_type or lvl_num is not None:
        lvl_ord = codes.level_ordinal(a.level_type, lvl_num)
    nf, nl = to_int(a.number_first), to_int(a.number_last)
    if nl is not None and nf is not None and nl == nf:
        nl = None
    lot = a.lot_number_prefix + a.lot_number + a.lot_number_suffix
    if nf is not None and nl is not None:
        shape = "range"
    elif nf is not None:
        shape = "single"
    elif lot:
        shape = "lot"
    else:
        shape = "none"
    street_key = "|".join((a.street_name, a.street_type, a.street_suffix, a.locality_name))
    street_label = a.street_label()
    if shape == "range":
        base_label = "%s%d%s-%d %s" % (a.number_first_prefix, nf, a.number_first_suffix, nl, street_label)
    elif shape == "single":
        base_label = "%s%d%s %s" % (a.number_first_prefix, nf, a.number_first_suffix, street_label)
    elif shape == "lot":
        base_label = "LOT %s %s" % (lot, street_label)
    else:
        base_label = street_label
    is_sub = bool(a.flat_type or a.flat_number or a.flat_prefix or a.flat_suffix
                  or a.level_type or a.level_number)
    return Feat(
        a=a, pid=a.addr_id, is_sub=is_sub, ftype=a.flat_type, fclass=codes.flat_class(a.flat_type),
        pfx=a.flat_prefix, num=num, sfx=a.flat_suffix, lvl_type=a.level_type,
        lvl_num=lvl_num, lvl_ord=lvl_ord, bname=a.building_name,
        bref=building_ref(a.building_name), street_key=street_key, street_label=street_label,
        nf=nf, nf_sfx=a.number_first_suffix, nl=nl, lot=lot, base_shape=shape,
        base_key=(street_key, a.number_first_prefix, nf, a.number_first_suffix, nl,
                  lot if shape == "lot" else ""),
        base_label=base_label,
    )


def base_relation(fa: Feat, fb: Feat) -> str:
    """'same' | 'contains' | 'same_street' | 'other_street'."""
    if fa.base_key == fb.base_key:
        return "same"
    if fa.street_key != fb.street_key:
        return "other_street"
    for x, y in ((fa, fb), (fb, fa)):
        if x.base_shape == "range" and y.nf is not None and x.nf is not None:
            hi = y.nl if y.nl is not None else y.nf
            if x.nf <= y.nf and hi <= x.nl:
                return "contains"
    return "same_street"
