"""Synthetic input in the engine's DataFrame schema, with ground truth.

Each scenario is one cadastre parcel whose dwellings are addressed by several
"sources" that number them differently - the duplicate patterns seen in real
apartment precincts. The town, streets and IDs are invented.

    df, truth = make_synthetic(seed=42)
    # df: INPUT_COLUMNS; truth: ADDR_ID, ENTITY_ID, ENTITY_TYPE, SCENARIO, STYLE
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .model import INPUT_COLUMNS

LOCALITIES = [("PORT ALDER", "2989"), ("WATTLE QUAY", "2988"), ("BANKSIA HEIGHTS", "2987"),
              ("PELICAN POINT", "2986")]

SCENARIOS = {
    "single_tower": "One tower on a range address: level-encoded units (UNIT 1103) vs "
                    "'UNIT 3, LEVEL 11' on one street number, APARTMENT synonyms, shops, exact duplicates.",
    "multi_tower": "Three towers of different heights: building as unit prefix (UNIT A1101), as "
                   "building name (TOWER A) and as individual street numbers; retail confounders.",
    "corner_frontage": "One building addressed from two streets with identical numbering, plus an "
                       "explicit-level variant.",
    "strata_lot_numbering": "Level-encoded unit numbers vs strata-lot order (UNIT 1..63) on the same "
                            "base. Not confirmable from components alone: expect REVIEW candidates.",
    "level_convention": "Ground floor as G-prefix (G01) vs a source that counts ground as LEVEL 1.",
    "twin_towers": "Two identical towers, prefix A/B vs TOWER A/B names: the letters resolve them.",
    "twin_towers_unresolved": "Identical towers, street numbers vs EAST/WEST TOWER names: nothing "
                              "separates them, so expect AMBIGUOUS review candidates, not guesses.",
    "townhouse_mix": "Sequentially numbered units plus rear townhouses re-addressed as 'UNIT n' off "
                     "a rear street (a trap for text matching).",
    "mixed_use": "Shops, level-1 suites and apartments: SUITE/OFFICE, SHOP/TENANCY and "
                 "APARTMENT/UNIT synonyms; shops vs suites must not match.",
    "background": "Houses and small strata blocks: FLAT/UNIT synonyms and lot-number duplicates.",
}


class _Builder:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.rows: List[dict] = []
        self.truth: List[dict] = []
        self.seen: Dict[tuple, str] = {}
        self.n_addr = 0
        self.n_par = 0

    def parcel(self) -> str:
        self.n_par += 1
        return "cpl%012x" % (0x5A1700000000 + self.n_par * 7919)

    def maybe(self, p: float) -> bool:
        return self.rng.random() < p

    def add(self, parcel: str, loc: Tuple[str, str], scenario: str, entity: str, etype: str,
            style: str, street: Tuple[str, str], n1: Optional[int] = None, n2: Optional[int] = None,
            ft: str = "", fpfx: str = "", fnum: Optional[int] = None, lt: str = "",
            lnum: Optional[int] = None, bname: str = "", lot: str = "", force: bool = False):
        sig = (parcel, bname, ft, fpfx, fnum, lt, lnum, n1, n2, street, lot if n1 is None else "")
        if sig in self.seen and not force:
            return                               # one record per address string
        self.n_addr += 1
        aid = "A%07d" % self.n_addr
        self.seen.setdefault(sig, aid)
        s = lambda v: "" if v is None else str(v)
        self.rows.append({
            "ADDR_ID": aid, "SOURCE_ID": "TXT%08d" % (40000000 + self.n_addr * 13),
            "BUILDING_NAME": bname, "LOT_NUMBER_PREFIX": "", "LOT_NUMBER": lot,
            "LOT_NUMBER_SUFFIX": "", "FLAT_TYPE_CODE": ft, "FLAT_NUMBER_PREFIX": fpfx,
            "FLAT_NUMBER": s(fnum), "FLAT_NUMBER_SUFFIX": "", "LEVEL_TYPE_CODE": lt,
            "LEVEL_NUMBER_PREFIX": "", "LEVEL_NUMBER": s(lnum), "LEVEL_NUMBER_SUFFIX": "",
            "NUMBER_FIRST_PREFIX": "", "NUMBER_FIRST": s(n1), "NUMBER_FIRST_SUFFIX": "",
            "NUMBER_LAST_PREFIX": "", "NUMBER_LAST": s(n2), "NUMBER_LAST_SUFFIX": "",
            "STREET_NAME": street[0], "STREET_TYPE_CODE": street[1], "STREET_SUFFIX_CODE": "",
            "LOCALITY_NAME": loc[0], "STATE_ABBREVIATION": "NSW", "POSTCODE": loc[1],
            "CADASTRE_POLYGON_PID": parcel,
        })
        self.truth.append({"ADDR_ID": aid, "ENTITY_ID": entity, "ENTITY_TYPE": etype,
                           "SCENARIO": scenario, "STYLE": style, "CADASTRE_POLYGON_PID": parcel})

    # ------------------------------------------------------------------
    def single_tower(self, loc, tag):
        P, sc, st = self.parcel(), "single_tower", ("HARBOURVIEW", "PARADE")
        n0, n1 = 2, 8
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        bldg = e("BLDG")
        self.add(P, loc, sc, bldg, "BUILDING", "P:range", st, n0, n1)
        self.add(P, loc, sc, bldg, "BUILDING", "P:single", st, n0)
        self.add(P, loc, sc, bldg, "BUILDING", "P:named-range", st, n0, n1, bname="ALDER TOWER")
        dwell = []
        for L in range(1, 15):
            for u in range(1, 9):
                d = e(L, u)
                dwell.append((d, L, u))
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU} range", st, n0, n1, "UNIT", fnum=L * 100 + u)
                if self.maybe(0.55):
                    self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {U} LEVEL {L} single", st, n0, None,
                             "UNIT", fnum=u, lt="L", lnum=L)
                if self.maybe(0.12):
                    self.add(P, loc, sc, d, "DWELLING", "S3 APARTMENT {L}{UU} range", st, n0, n1, "APT",
                             fnum=L * 100 + u)
        for k in range(1, 5):
            d = e("SHOP", k)
            self.add(P, loc, sc, d, "TENANCY", "S4 SHOP {N} range", st, n0, n1, "SHOP", fnum=k)
            if self.maybe(0.6):
                self.add(P, loc, sc, d, "TENANCY", "S5 SHOP {N} GROUND single", st, n0, None, "SHOP",
                         fnum=k, lt="G")
        for d, L, u in self.rng.sample(dwell, 3):          # exact-component duplicates
            self.add(P, loc, sc, d, "DWELLING", "S1 exact duplicate", st, n0, n1, "UNIT",
                     fnum=L * 100 + u, force=True)

    def multi_tower(self, loc, tag):
        P, sc, st = self.parcel(), "multi_tower", ("MERIDIAN", "AVENUE")
        towers = {"A": (1, 18, 10, [1, 2, 3, 4]), "B": (7, 12, 8, [5, 6, 7]), "C": (15, 8, 6, [])}
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("SITE"), "SITE", "P:range", st, 1, 15)
        for key, (num, levels, upl, shops) in towers.items():
            bldg = e("BLDG", key)
            self.add(P, loc, sc, bldg, "BUILDING", "P:named-range", st, 1, 15, bname="TOWER " + key)
            self.add(P, loc, sc, bldg, "BUILDING", "P:single", st, num)
            for L in range(1, levels + 1):
                for u in range(1, upl + 1):
                    d = e(key, L, u)
                    self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {B}{L}{UU} range", st, 1, 15, "UNIT",
                             fpfx=key, fnum=L * 100 + u)
                    if self.maybe(0.8):
                        self.add(P, loc, sc, d, "DWELLING", "S2 TOWER {B}, UNIT {L}{UU} range", st, 1, 15,
                                 "UNIT", fnum=L * 100 + u, bname="TOWER " + key)
                    if self.maybe(0.6):
                        self.add(P, loc, sc, d, "DWELLING", "S3 UNIT {L}{UU} single", st, num, None,
                                 "UNIT", fnum=L * 100 + u)
            for k in shops:
                d = e(key, "SHOP", k)
                self.add(P, loc, sc, d, "TENANCY", "S4 SHOP {N} range", st, 1, 15, "SHOP", fnum=k)
                if self.maybe(0.5):
                    self.add(P, loc, sc, d, "TENANCY", "S5 SHOP {N} single", st, num, None, "SHOP", fnum=k)

    def corner_frontage(self, loc, tag):
        P, sc = self.parcel(), "corner_frontage"
        south, west = ("QUARRY", "STREET"), ("WATTLE", "LANE")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        bldg = e("BLDG")
        self.add(P, loc, sc, bldg, "BUILDING", "P:primary street", south, 40)
        self.add(P, loc, sc, bldg, "BUILDING", "P:secondary street", west, 2)
        for L, upl in [(0, 4)] + [(L, 6) for L in range(1, 11)]:
            for u in range(1, upl + 1):
                d = e(L, u)
                enc = dict(fpfx="G", fnum=u) if L == 0 else dict(fnum=L * 100 + u)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {G|L}{UU} primary", south, 40, None, "UNIT", **enc)
                if self.maybe(0.5):
                    self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {G|L}{UU} secondary", west, 2, None,
                             "UNIT", **enc)
                if self.maybe(0.4):
                    lvl = dict(lt="G") if L == 0 else dict(lt="L", lnum=L)
                    self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {U}, LEVEL {L}|GROUND", south, 40, None,
                             "UNIT", fnum=u, **lvl)

    def strata_lot_numbering(self, loc, tag):
        P, sc, st = self.parcel(), "strata_lot_numbering", ("CANAL", "ROAD")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:single", st, 120)
        lot = 0
        for L in range(1, 10):
            for u in range(1, 8):
                lot += 1
                d = e(L, u)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU}", st, 120, None, "UNIT", fnum=L * 100 + u)
                if self.maybe(0.85):
                    self.add(P, loc, sc, d, "DWELLING", "S1b UNIT {strata lot}", st, 120, None, "UNIT", fnum=lot)

    def level_convention(self, loc, tag):
        P, sc, st = self.parcel(), "level_convention", ("JETTY", "ROAD")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:single", st, 30)
        for L, upl in [(0, 4)] + [(L, 5) for L in range(1, 11)]:
            for u in range(1, upl + 1):
                d = e(L, u)
                enc = dict(fpfx="G", fnum=u) if L == 0 else dict(fnum=L * 100 + u)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {G|L}{UU}", st, 30, None, "UNIT", **enc)
                if self.maybe(0.7):
                    self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {U}, LEVEL {L+1}", st, 30, None, "UNIT",
                             fnum=u, lt="L", lnum=L + 1)

    def twin_towers(self, loc, tag, resolvable: bool = True):
        sc = "twin_towers" if resolvable else "twin_towers_unresolved"
        P, st = self.parcel(), ("SKYLINE", "BOULEVARD")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("SITE"), "SITE", "P:range", st, 50, 52)
        for key, num, name in (("A", 50, "TOWER A" if resolvable else "WEST TOWER"),
                               ("B", 52, "TOWER B" if resolvable else "EAST TOWER")):
            bldg = e("BLDG", key)
            self.add(P, loc, sc, bldg, "BUILDING", "P:named-range", st, 50, 52, bname=name)
            if not resolvable:
                self.add(P, loc, sc, bldg, "BUILDING", "P:single", st, num)
            for L in range(1, 11):
                for u in range(1, 7):
                    d = e(key, L, u)
                    if resolvable:
                        self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {B}{L}{UU} range", st, 50, 52, "UNIT",
                                 fpfx=key, fnum=L * 100 + u)
                    else:
                        self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU} single", st, num, None, "UNIT",
                                 fnum=L * 100 + u)
                    if self.maybe(0.75):
                        self.add(P, loc, sc, d, "DWELLING", "S2 {NAME}, UNIT {L}{UU} range", st, 50, 52,
                                 "UNIT", fnum=L * 100 + u, bname=name)

    def townhouse_mix(self, loc, tag):
        P, sc = self.parcel(), "townhouse_mix"
        front, rear = ("ORCHARD", "STREET"), ("ORCHARD", "LANE")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("SITE"), "SITE", "P:front street", front, 88)
        self.add(P, loc, sc, e("BLDG", "TH"), "BUILDING", "P:rear street", rear, 5)
        seq = 0
        for L in range(0, 4):
            for p in range(1, 7):
                seq += 1
                d = e("U", L, p)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {seq} front", front, 88, None, "UNIT", fnum=seq)
                if self.maybe(0.3):
                    self.add(P, loc, sc, d, "DWELLING", "S3 FLAT {seq} front", front, 88, None, "FLAT", fnum=seq)
        for k in range(1, 9):
            d = e("TH", k)
            self.add(P, loc, sc, d, "DWELLING", "S1 TOWNHOUSE {N} front", front, 88, None, "TNHS", fnum=k)
            if self.maybe(0.65):
                self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {N} rear street", rear, 5, None, "UNIT", fnum=k)

    def mixed_use(self, loc, tag):
        P, sc, st = self.parcel(), "mixed_use", ("BANKSIA", "STREET")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        bldg = e("BLDG")
        self.add(P, loc, sc, bldg, "BUILDING", "P:range", st, 64, 68)
        self.add(P, loc, sc, bldg, "BUILDING", "P:single", st, 64)
        for k in range(1, 7):
            d = e("SHOP", k)
            self.add(P, loc, sc, d, "TENANCY", "S1 SHOP {N} range", st, 64, 68, "SHOP", fnum=k)
            if self.maybe(0.5):
                self.add(P, loc, sc, d, "TENANCY", "S3 TENANCY {N}, GROUND single", st, 64, None, "TNCY",
                         fnum=k, lt="G")
        for u in range(1, 11):
            d = e("SUITE", u)
            self.add(P, loc, sc, d, "TENANCY", "S1 SUITE 1{UU} range", st, 64, 68, "SE", fnum=100 + u)
            if self.maybe(0.6):
                self.add(P, loc, sc, d, "TENANCY", "S2 OFFICE 1{UU} single", st, 64, None, "OFFC", fnum=100 + u)
        for L in range(2, 13):
            for u in range(1, 9):
                d = e(L, u)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU} range", st, 64, 68, "UNIT", fnum=L * 100 + u)
                if self.maybe(0.6):
                    self.add(P, loc, sc, d, "DWELLING", "S2 APARTMENT {L}{UU} single", st, 64, None, "APT",
                             fnum=L * 100 + u)

    def background(self, loc, tag, n_lots: int = 40):
        sc = "background"
        streets = [("TIDEWATER", "DRIVE"), ("SALTBUSH", "TERRACE"), ("MARINER", "WAY")]
        for i in range(n_lots):
            st = streets[i % len(streets)]
            num = 2 * (i // len(streets)) + 1
            P = self.parcel()
            e = lambda *k: ":".join((tag, sc, st[0], str(num)) + tuple(str(x) for x in k))
            if self.maybe(0.3):                       # small strata block
                self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:small strata", st, num)
                synonyms = self.maybe(0.5)
                for u in range(1, self.rng.randint(2, 6) + 1):
                    self.add(P, loc, sc, e(u), "DWELLING", "S1 UNIT {N}", st, num, None, "UNIT", fnum=u)
                    if synonyms and self.maybe(0.6):
                        self.add(P, loc, sc, e(u), "DWELLING", "S2 FLAT {N}", st, num, None, "FLAT", fnum=u)
            else:                                     # house, sometimes with an old lot address
                self.add(P, loc, sc, e("HOUSE"), "HOUSE", "P:house", st, num)
                if self.maybe(0.15):
                    self.add(P, loc, sc, e("HOUSE"), "HOUSE", "P:house LOT number", st,
                             lot=str(self.rng.randint(10, 400)))


def make_synthetic(seed: int = 42, replicates: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (input DataFrame in INPUT_COLUMNS order, truth DataFrame)."""
    b = _Builder(seed)
    for r in range(replicates):
        loc = LOCALITIES[r % len(LOCALITIES)]
        if r >= len(LOCALITIES):
            loc = ("%s %d" % (loc[0], r // len(LOCALITIES) + 1), loc[1])
        tag = "R%d" % r
        b.single_tower(loc, tag)
        b.multi_tower(loc, tag)
        b.corner_frontage(loc, tag)
        b.strata_lot_numbering(loc, tag)
        b.level_convention(loc, tag)
        b.twin_towers(loc, tag, resolvable=True)
        b.twin_towers(loc, tag, resolvable=False)
        b.townhouse_mix(loc, tag)
        b.mixed_use(loc, tag)
        b.background(loc, tag)
    df = pd.DataFrame(b.rows, columns=INPUT_COLUMNS)
    truth = pd.DataFrame(b.truth)
    # shuffle so nothing relies on input order
    order = list(range(len(df)))
    random.Random(seed + 1).shuffle(order)
    return df.iloc[order].reset_index(drop=True), truth.iloc[order].reset_index(drop=True)
