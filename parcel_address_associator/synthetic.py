"""Synthetic input in the engine's DataFrame schema, with ground truth.

Each scenario is one cadastre parcel whose dwellings are addressed by several
"sources" that number them differently - the duplicate patterns seen in real
apartment precincts. The town, streets and IDs are invented.

    df, truth = make_synthetic(seed=42)
    # df: INPUT_COLUMNS; truth: ADDR_ID, ENTITY_ID, ENTITY_TYPE, SCENARIO, STYLE, SHELL_CLASS

    df, truth, geo = make_synthetic(seed=42, with_geometry=True)
    # df also has LATITUDE/LONGITUDE; geo = {"cadastre": [CADASTRE_POLYGON_PID, WKT],
    #                                        "buildings": [BUILDING_ID, CADASTRE_POLYGON_PID, WKT]}

Most geocodes sit at the parcel centroid (``centroid_share``, default 0.85), as most
G-NAF geocodes do. The rest follow building-level habits: units at their building's
centroid, building addresses at the building or property centroid or the frontage,
site addresses at the property centroid or frontage.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

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
    "staggered_coverage": "Sources covering different floor ranges of one building (levels 1-8 vs "
                          "3-10 vs 2-7), plus one whose LEVEL field sometimes contradicts its door "
                          "number. Must match same numbers only, never UNIT 405 with UNIT 505.",
    "twin_towers": "Two identical towers, prefix A/B vs TOWER A/B names: the letters resolve them.",
    "twin_towers_unresolved": "Identical towers, street numbers vs EAST/WEST TOWER names: nothing "
                              "separates them, so expect AMBIGUOUS review candidates, not guesses.",
    "townhouse_mix": "Sequentially numbered units plus rear townhouses re-addressed as 'UNIT n' off "
                     "a rear street (a trap for text matching).",
    "mixed_use": "Shops, level-1 suites and apartments: SUITE/OFFICE, SHOP/TENANCY and "
                 "APARTMENT/UNIT synonyms; shops vs suites must not match.",
    "common_property": "A tower with common-property addresses (LOBBY, CARPARK 1, BASEMENT 1, ROOFTOP, "
                       "SUBSTATION in a kiosk footprint, a COMMON PROPERTY building name) beside private "
                       "car spaces and stores that are neither dwellings nor shells.",
    "community_title": "An association lot (internal driveway, no building) holding '1 WARATAH CLOSE', "
                       "whose townhouses 'UNIT n, 1 WARATAH CLOSE' each sit on their own parcel.",
    "background": "Houses and small strata blocks: FLAT/UNIT synonyms and lot-number duplicates; front "
                  "houses with rear units (the base address is the house, not a shell), vacant lots "
                  "and garden sheds.",
}
SHELL_CLASS_OF = {"BUILDING": "BUILDING", "SITE": "SITE", "COMMON": "COMMON_PROPERTY"}
TOWN_ORIGIN = (150.9, -33.1)          # lon, lat of the fictional town's local grid


class _Builder:
    def __init__(self, seed: int, centroid_share: float = 0.85):
        self.rng = random.Random(seed)
        self.centroid_share = centroid_share    # geocodes at the parcel centroid, as most G-NAF ones are
        self.rows: List[dict] = []
        self.truth: List[dict] = []
        self.seen: Dict[tuple, str] = {}
        self.first_of: Dict[str, str] = {}      # entity -> first ADDR_ID (the one a register counts)
        self.n_addr = 0
        self.n_par = 0
        # geometry in town metres; its own RNG so the layout never shifts the addresses
        self.grng = random.Random(seed + 99)
        self.parcel_geo: Dict[str, List[Tuple[float, float]]] = {}
        self.footprints: List[Tuple[str, str, List[Tuple[float, float]]]] = []
        self.points: Dict[str, Tuple[float, float]] = {}
        self.x = self.y = 0.0                    # street cursor

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
        self.first_of.setdefault(entity, aid)
        self.truth.append({"ADDR_ID": aid, "ENTITY_ID": entity, "ENTITY_TYPE": etype,
                           "SCENARIO": scenario, "STYLE": style, "CADASTRE_POLYGON_PID": parcel})

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def lot(self, P: str, w: float, d: float, gap: float = 12.0) -> Tuple[float, float]:
        """Lay parcel P out as a w x d rectangle fronting the street at the cursor."""
        o = (self.x, self.y)
        self.x += w + gap
        self.parcel_geo[P] = self.rect(o[0], o[1], o[0] + w, o[1] + d)
        return o

    def place(self, i0: int, P: str, o: Tuple[float, float], w: float, d: float,
              rects: Dict[str, Tuple[float, float, float, float]],
              building_of: Callable[[str], str] = lambda ent: "",
              fixed: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """Footprints (key -> rect relative to the parcel origin) and G-NAF-like geocodes
        for every address added on P since row i0."""
        for r in rects.values():
            self.footprints.append(("BLD%06d" % (len(self.footprints) + 1), P,
                                    self.rect(o[0] + r[0], o[1] + r[1], o[0] + r[2], o[1] + r[3])))
        pc, fc = (o[0] + w / 2, o[1] + d / 2), (o[0] + w / 2, o[1] + 1.5)
        bc = {k: (o[0] + (r[0] + r[2]) / 2, o[1] + (r[1] + r[3]) / 2) for k, r in rects.items()}
        for row, t in zip(self.rows[i0:], self.truth[i0:]):
            if row["CADASTRE_POLYGON_PID"] != P:
                continue
            ent, et = t["ENTITY_ID"], t["ENTITY_TYPE"]
            b, r = bc.get(building_of(ent)), self.grng.random()
            if fixed and ent in fixed:
                pt = (o[0] + fixed[ent][0], o[1] + fixed[ent][1])
            elif self.grng.random() < self.centroid_share:
                pt = pc
            elif et == "SITE":
                pt = pc if r < 0.5 else fc
            elif et == "BUILDING":
                pt = b if (b and r < 0.55) else (pc if r < 0.85 else fc)
            elif et == "HOUSE":
                pt = b if (b and r < 0.75) else (pc if r < 0.9 else fc)
            else:
                pt = b if (b and r < 0.8) else pc
            self.points[row["ADDR_ID"]] = pt

    def geometry_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """Town metres -> lon/lat WKT; returns (cadastre, buildings, {ADDR_ID: (lon, lat)})."""
        import shapely

        from .geometry import metres_per_degree
        lon0, lat0 = TOWN_ORIGIN
        kx, ky = metres_per_degree(lat0)
        to = lambda x, y: (lon0 + x / kx, lat0 + y / ky)
        wkt = lambda ring: shapely.to_wkt(shapely.Polygon([to(*c) for c in ring]), rounding_precision=8)
        cad = pd.DataFrame([{"CADASTRE_POLYGON_PID": p, "WKT": wkt(r)} for p, r in self.parcel_geo.items()])
        bld = pd.DataFrame([{"BUILDING_ID": b, "CADASTRE_POLYGON_PID": p, "WKT": wkt(r)}
                            for b, p, r in self.footprints])
        return cad, bld, {a: to(*xy) for a, xy in self.points.items()}

    # ------------------------------------------------------------------
    def use_flags(self, seed: int) -> pd.DataFrame:
        """Base ADDRESS_TYPE and a second dataset's RESIDENTIAL_DWELLING flag, each with
        recorded errors. Returns the use DataFrame; fills ADDRESS_TYPE and truth columns."""
        rng = random.Random(seed + 7)
        etype = {t["ENTITY_ID"]: t["ENTITY_TYPE"] for t in self.truth}
        dwelling_entities = sorted(e for e, t in etype.items() if t in ("DWELLING", "HOUSE"))
        missed = {e for e in dwelling_entities if rng.random() < 0.03}
        gap = {e for e in dwelling_entities if ":multi_tower:C:" in e}       # a whole tower missing
        flags = []
        for row, t in zip(self.rows, self.truth):
            aid, ent, et = row["ADDR_ID"], t["ENTITY_ID"], t["ENTITY_TYPE"]
            first = self.first_of[ent] == aid
            base_err = other_err = ""
            if et == "DWELLING":
                atype = "SHELL_CHILD"
                if rng.random() < 0.015:
                    atype, base_err = "COMMERCIAL", "dwelling_not_flagged"
            elif et == "HOUSE":
                atype = "RESIDENTIAL_DWELLING"
            elif et in ("BUILDING", "SITE", "COMMON"):
                atype = "SHELL_PARENT"
                if rng.random() < 0.35:
                    atype, base_err = "RESIDENTIAL_DWELLING", "building_flagged"
            else:
                atype = "COMMERCIAL"
                if rng.random() < 0.06:
                    atype, base_err = "SHELL_CHILD", "commercial_flagged"
            if et in ("DWELLING", "HOUSE"):
                res = first
                if first and ent in gap:
                    res, other_err = False, "block_missed"
                elif first and ent in missed:
                    res, other_err = False, "dwelling_missed"
            else:
                res = False
                if rng.random() < 0.03:
                    res, other_err = True, ("building_flagged" if et in ("BUILDING", "SITE", "COMMON")
                                            else "commercial_flagged")
            row["ADDRESS_TYPE"] = atype
            t.update({"IS_DWELLING": et in ("DWELLING", "HOUSE"), "BASE_ERROR": base_err,
                      "OTHER_ERROR": other_err})
            flags.append({"ADDR_ID": aid, "RESIDENTIAL_DWELLING": "Y" if res else "N"})
        return pd.DataFrame(flags)

    # ------------------------------------------------------------------
    def single_tower(self, loc, tag):
        P, sc, st = self.parcel(), "single_tower", ("HARBOURVIEW", "PARADE")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 50, 46), 50, 46, {"": (8, 8, 42, 38)})

    def multi_tower(self, loc, tag):
        P, sc, st = self.parcel(), "multi_tower", ("MERIDIAN", "AVENUE")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 140, 70), 140, 70,
                   {"A": (8, 8, 48, 40), "B": (58, 8, 90, 36), "C": (100, 10, 130, 34)}, _tower_key("ABC"))

    def corner_frontage(self, loc, tag):
        P, sc = self.parcel(), "corner_frontage"
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 40, 40), 40, 40, {"": (6, 6, 34, 34)})

    def strata_lot_numbering(self, loc, tag):
        P, sc, st = self.parcel(), "strata_lot_numbering", ("CANAL", "ROAD")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 36, 40), 36, 40, {"": (5, 6, 31, 34)})

    def level_convention(self, loc, tag):
        P, sc, st = self.parcel(), "level_convention", ("JETTY", "ROAD")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 30, 40), 30, 40, {"": (5, 6, 25, 34)})

    def staggered_coverage(self, loc, tag):
        P, sc, st = self.parcel(), "staggered_coverage", ("REGATTA", "ROAD")
        i0 = len(self.rows)
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:range", st, 20, 26)
        self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:single", st, 20)
        for L in range(1, 11):
            for u in range(1, 7):
                d = e(L, u)
                if L <= 8 and self.maybe(0.9):
                    self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU} range, levels 1-8", st, 20, 26,
                             "UNIT", fnum=L * 100 + u)
                if L >= 3 and self.maybe(0.9):
                    self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {L}{UU} single, levels 3-10", st, 20, None,
                             "UNIT", fnum=L * 100 + u)
                if 2 <= L <= 7 and self.maybe(0.8):
                    self.add(P, loc, sc, d, "DWELLING", "S3 UNIT {U}, LEVEL {L}, levels 2-7", st, 20, None,
                             "UNIT", fnum=u, lt="L", lnum=L)
                if self.maybe(0.5):
                    # LEVEL field occasionally one out from the door number
                    lnum = L + (1 if self.maybe(0.15) else 0)
                    self.add(P, loc, sc, d, "DWELLING", "S4 APARTMENT {L}{UU}, LEVEL {L}", st, 20, None,
                             "APT", fnum=L * 100 + u, lt="L", lnum=lnum)
        self.place(i0, P, self.lot(P, 40, 46), 40, 46, {"": (6, 6, 34, 38)})

    def twin_towers(self, loc, tag, resolvable: bool = True):
        sc = "twin_towers" if resolvable else "twin_towers_unresolved"
        P, st = self.parcel(), ("SKYLINE", "BOULEVARD")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 90, 50), 90, 50, {"A": (8, 8, 38, 40), "B": (52, 8, 82, 40)},
                   _tower_key("AB"))

    def townhouse_mix(self, loc, tag):
        P, sc = self.parcel(), "townhouse_mix"
        i0 = len(self.rows)
        front, rear = ("ORCHARD", "STREET"), ("ORCHARD", "LANE")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("SITE"), "SITE", "P:front street", front, 88)
        self.add(P, loc, sc, e("BLDG", "TH"), "SITE", "P:rear street", rear, 5)     # the townhouse row
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
        rects = {"U": (8, 6, 52, 30)}
        rects.update({"TH%d" % k: (4 + (k - 1) * 7, 50, 10 + (k - 1) * 7, 64) for k in range(1, 9)})

        def building_of(ent: str) -> str:
            parts = ent.split(":")
            if parts[2] == "U":
                return "U"
            return "TH" + parts[3] if parts[2] == "TH" else ""
        self.place(i0, P, self.lot(P, 60, 80), 60, 80, rects, building_of)

    def mixed_use(self, loc, tag):
        P, sc, st = self.parcel(), "mixed_use", ("BANKSIA", "STREET")
        i0 = len(self.rows)
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
        self.place(i0, P, self.lot(P, 46, 46), 46, 46, {"": (5, 5, 41, 41)})

    def common_property(self, loc, tag):
        P, sc, st = self.parcel(), "common_property", ("CORAL", "STREET")
        i0 = len(self.rows)
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:single", st, 10)
        if self.maybe(0.6):
            self.add(P, loc, sc, e("CP"), "COMMON", "C COMMON PROPERTY name", st, 10, bname="COMMON PROPERTY")
        for L in range(1, 9):
            for u in range(1, 7):
                d = e(L, u)
                self.add(P, loc, sc, d, "DWELLING", "S1 UNIT {L}{UU}", st, 10, None, "UNIT", fnum=L * 100 + u)
                if self.maybe(0.5):
                    self.add(P, loc, sc, d, "DWELLING", "S2 UNIT {U}, LEVEL {L}", st, 10, None, "UNIT",
                             fnum=u, lt="L", lnum=L)
        self.add(P, loc, sc, e("LOBBY"), "COMMON", "C LOBBY", st, 10, None, "LBBY")
        self.add(P, loc, sc, e("CARPARK"), "COMMON", "C CARPARK 1", st, 10, None, "CARP", fnum=1)
        for b in (1, 2):
            if b == 1 or self.maybe(0.5):
                self.add(P, loc, sc, e("BASEMENT", b), "COMMON", "C BASEMENT {N}", st, 10, None, lt="B", lnum=b)
        if self.maybe(0.5):
            self.add(P, loc, sc, e("ROOF"), "COMMON", "C ROOFTOP", st, 10, None, lt="RT")
        self.add(P, loc, sc, e("SUBS"), "COMMON", "C SUBSTATION", st, 10, None, "SUBS", fnum=1)
        for k in range(1, 13):           # private lots: neither dwellings nor shells
            self.add(P, loc, sc, e("CS", k), "CARSPACE", "X CARSPACE {N}", st, 10, None, "CARS", fnum=k)
        for k in range(1, 7):
            self.add(P, loc, sc, e("ST", k), "STORAGE", "X STORE {N}", st, 10, None, "STOR", fnum=k)
        self.place(i0, P, self.lot(P, 50, 52), 50, 52, {"": (8, 10, 42, 42), "SUBS": (44, 2, 48, 5)},
                   lambda ent: "SUBS" if ent.endswith(":SUBS") else "",
                   fixed={e("CARPARK"): (4.0, 3.0)})           # at the driveway, outside the tower

    def community_title(self, loc, tag):
        sc, st = "community_title", ("WARATAH", "CLOSE")
        e = lambda *k: ":".join((tag, sc) + tuple(str(x) for x in k))
        x0, y0 = self.x, self.y
        A = self.parcel()                # association property: the internal driveway
        i0 = len(self.rows)
        self.add(A, loc, sc, e("SITE"), "SITE", "P:association lot", st, 1)
        self.parcel_geo[A] = self.rect(x0 + 12, y0, x0 + 19, y0 + 70)
        self.place(i0, A, (x0 + 12, y0), 7, 70, {})
        k = 0
        for lx in (0, 19):
            for j in range(5):
                k += 1
                P, i0 = self.parcel(), len(self.rows)
                self.add(P, loc, sc, e(k), "DWELLING", "S1 UNIT {N} on its own lot", st, 1, None, "UNIT", fnum=k)
                o = (x0 + lx, y0 + 70 - (j + 1) * 14)
                self.parcel_geo[P] = self.rect(o[0], o[1], o[0] + 12, o[1] + 14)
                self.place(i0, P, o, 12, 14, {"": (2, 2, 10, 12)})
        self.x += 31 + 12
        for num in (3, 5, 7):            # ordinary houses further along the close
            P, i0 = self.parcel(), len(self.rows)
            self.add(P, loc, sc, e("HOUSE", num), "HOUSE", "P:house", st, num)
            self.place(i0, P, self.lot(P, 15, 40, gap=0), 15, 40, {"": (3, 6, 12, 22)})
        self.x += 12

    def background(self, loc, tag, n_lots: int = 40):
        sc = "background"
        streets = [("TIDEWATER", "DRIVE"), ("SALTBUSH", "TERRACE"), ("MARINER", "WAY")]
        for i in range(n_lots):
            st = streets[i % len(streets)]
            num = 2 * (i // len(streets)) + 1
            P = self.parcel()
            i0 = len(self.rows)
            e = lambda *k: ":".join((tag, sc, st[0], str(num)) + tuple(str(x) for x in k))
            kind = self.rng.random()
            building_of = lambda ent: ""
            if kind < 0.3:                            # small strata block
                self.add(P, loc, sc, e("BLDG"), "BUILDING", "P:small strata", st, num)
                synonyms = self.maybe(0.5)
                for u in range(1, self.rng.randint(2, 6) + 1):
                    self.add(P, loc, sc, e(u), "DWELLING", "S1 UNIT {N}", st, num, None, "UNIT", fnum=u)
                    if synonyms and self.maybe(0.6):
                        self.add(P, loc, sc, e(u), "DWELLING", "S2 FLAT {N}", st, num, None, "FLAT", fnum=u)
                rects = {"": (2, 6, 13, 34)}
            elif kind < 0.42:                         # front house + rear units: the base is the house
                self.add(P, loc, sc, e("HOUSE"), "HOUSE", "P:front house", st, num)
                for u in range(1, self.rng.randint(2, 3) + 1):
                    self.add(P, loc, sc, e(u), "DWELLING", "S1 UNIT {N} rear", st, num, None, "UNIT", fnum=u)
                rects = {"HOUSE": (3, 5, 12, 17), "REAR": (2, 24, 13, 36)}
                building_of = lambda ent: "HOUSE" if ent.endswith(":HOUSE") else "REAR"
            else:                                     # house, sometimes with an old lot address
                self.add(P, loc, sc, e("HOUSE"), "HOUSE", "P:house", st, num)
                if self.maybe(0.15):
                    self.add(P, loc, sc, e("HOUSE"), "HOUSE", "P:house LOT number", st,
                             lot=str(self.rng.randint(10, 400)))
                rects = {} if self.grng.random() < 0.08 else {"": (3, 6, 12, 22)}      # vacant lot
                if rects and self.grng.random() < 0.4:
                    rects["SHED"] = (10, 32, 13, 36)                               # 12 m2 garden shed
            self.place(i0, P, self.lot(P, 15, 40, gap=0), 15, 40, rects, building_of)


def _tower_key(keys: str) -> Callable[[str], str]:
    """Entity id -> the single-letter building key it belongs to ('' for site-wide)."""
    def f(ent: str) -> str:
        return next((k for k in ent.split(":")[2:] if len(k) == 1 and k in keys), "")
    return f


def make_synthetic(seed: int = 42, replicates: int = 1, with_use: bool = False,
                   with_geometry: bool = False, centroid_share: float = 0.85):
    """Return (input, truth), plus the use DataFrame [ADDR_ID, RESIDENTIAL_DWELLING]
    with ``with_use=True`` and the geometry {"cadastre", "buildings"} with
    ``with_geometry=True``, in that order. ``centroid_share`` is the share of
    geocodes placed at the parcel centroid; the rest are building-level.

    The input has INPUT_COLUMNS plus ADDRESS_TYPE (and LATITUDE/LONGITUDE with
    geometry). Truth has ENTITY_ID (shared by duplicates), ENTITY_TYPE, SCENARIO,
    STYLE, IS_DWELLING, SHELL_CLASS (BUILDING / SITE / COMMON_PROPERTY / ''), IS_SHELL
    and the errors injected into each use source (BASE_ERROR, OTHER_ERROR).
    """
    b = _Builder(seed, centroid_share)
    for r in range(replicates):
        loc = LOCALITIES[r % len(LOCALITIES)]
        if r >= len(LOCALITIES):
            loc = ("%s %d" % (loc[0], r // len(LOCALITIES) + 1), loc[1])
        tag = "R%d" % r
        b.x, b.y = 0.0, r * 300.0
        b.single_tower(loc, tag)
        b.multi_tower(loc, tag)
        b.corner_frontage(loc, tag)
        b.strata_lot_numbering(loc, tag)
        b.level_convention(loc, tag)
        b.staggered_coverage(loc, tag)
        b.twin_towers(loc, tag, resolvable=True)
        b.twin_towers(loc, tag, resolvable=False)
        b.townhouse_mix(loc, tag)
        b.mixed_use(loc, tag)
        b.common_property(loc, tag)
        b.community_title(loc, tag)
        b.background(loc, tag)
    use = b.use_flags(seed)
    df = pd.DataFrame(b.rows, columns=INPUT_COLUMNS + ["ADDRESS_TYPE"])
    truth = pd.DataFrame(b.truth)
    truth["SHELL_CLASS"] = truth["ENTITY_TYPE"].map(SHELL_CLASS_OF).fillna("")
    truth["IS_SHELL"] = truth["SHELL_CLASS"] != ""
    out = []
    if with_geometry:
        cad, bld, pts = b.geometry_frames()
        df["LATITUDE"] = df["ADDR_ID"].map(lambda a: pts[a][1] if a in pts else float("nan"))
        df["LONGITUDE"] = df["ADDR_ID"].map(lambda a: pts[a][0] if a in pts else float("nan"))
        out = [{"cadastre": cad, "buildings": bld}]
    # shuffle so nothing relies on input order
    order = list(range(len(df)))
    random.Random(seed + 1).shuffle(order)
    df, truth = df.iloc[order].reset_index(drop=True), truth.iloc[order].reset_index(drop=True)
    if with_use:
        out.insert(0, use.iloc[order].reset_index(drop=True))
    return tuple([df, truth] + out)
