"""Shell address detection.

A *shell* address stands for something other than one dwelling or tenancy: the
whole building ("10 CORAL ST" over its units), the whole site (a range address
over several buildings, an association lot), a common-property element (LOBBY,
CARPARK 1, BASEMENT 1, SUBSTATION, a COMMON PROPERTY building name) or a whole
floor (LEVEL 3 with no flat, offered for review). Each address gets

    log-odds(shell) = prior by kind of address   (parent / unit / shop / car space /
                                                   common flat or level type)
                    + keywords                    (building name)
                    + what hangs below a parent   (distinct sub-address premises on its
                                                   base, through the parcel's inferred
                                                   structures, and on neighbouring parcels)
                    + geometry (optional)         parcel level, needs no geocode: one building
                                                   holding all the parcel's premises; more
                                                   buildings than premises, or a few premises
                                                   over several buildings (a front house?);
                                                   no building next to its premises (an
                                                   association lot)
                                                  building level, only from geocodes that are
                                                   not at the parcel centroid: the footprint a
                                                   parent sits in, with or apart from its
                                                   sub-addresses; kiosk-sized footprints

The class (BUILDING / SITE / COMMON_PROPERTY / FLOOR) comes from the address type
and, for parents, from how many structures or footprints their sub-addresses
occupy. Members of one duplicate group share a verdict.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from . import codes
from .alignment import sigmoid
from .config import Config
from .features import Feat, build_feat, compare_brefs, is_common_name
from .model import Address
from .reconcile import ParcelResult, ParentInfo

if TYPE_CHECKING:                       # shapely is only needed when geometry is attached
    from .geometry import GeoData, ParcelGeo

SHELL_FLAT_TYPES = {
    "BLDG": "BUILDING", "BLCK": "BUILDING", "TWR": "BUILDING", "SITE": "SITE",
    "LBBY": "COMMON_PROPERTY", "CARP": "COMMON_PROPERTY", "SUBS": "COMMON_PROPERTY",
    "BBQ": "COMMON_PROPERTY", "RESV": "COMMON_PROPERTY", "ANT": "COMMON_PROPERTY",
    "SIGN": "COMMON_PROPERTY", "ATM": "COMMON_PROPERTY",
}
PRIVATE_LOT_TYPES = {"CARS", "GRGE", "CAGE", "SHED", "BTSD", "MBTH", "STOR", "COOL", "VLT"}
COMMON_LEVEL_TYPES = {"B", "SB", "P", "PDM", "RT", "LB", "OD", "PLF"}
VERDICTS = ("SHELL", "POSSIBLE_SHELL", "NOT_SHELL")
CATEGORY_NOTES = {
    "OK": "verdict and ADDRESS_TYPE agree (or there is nothing to compare)",
    "SHELL_FLAGGED_RESIDENTIAL": "shell address counted as a residential dwelling by ADDRESS_TYPE or the "
                                 "second dataset",
    "SHELL_NOT_TYPED": "shell address whose ADDRESS_TYPE is not a shell type",
    "TYPED_SHELL_NOT_DETECTED": "ADDRESS_TYPE says shell, but the address looks like a dwelling or tenancy",
    "REVIEW": "possible shell: the evidence is not conclusive",
}


def covers(p: Feat, c: Feat) -> bool:
    """Does parent p's base address cover sub-address c's (same base, or c's number
    inside p's range on the same street)?"""
    if p.base_key == c.base_key:
        return True
    if p.street_key != c.street_key or p.base_shape != "range" or c.nf is None:
        return False
    return p.nf <= c.nf and (c.nl if c.nl is not None else c.nf) <= p.nl


def kind_of(f: Feat) -> Tuple[str, str]:
    """(prior key, shell class the address type itself implies)."""
    if not f.is_sub:
        return "parent", ""
    if f.ftype in SHELL_FLAT_TYPES:
        cls = SHELL_FLAT_TYPES[f.ftype]
        return ("common_type" if cls == "COMMON_PROPERTY" else "building_type"), cls
    if not f.ftype and f.num is None and not f.pfx:
        if f.lvl_type in COMMON_LEVEL_TYPES:
            return "common_level", "COMMON_PROPERTY"
        return "floor", "FLOOR"
    if f.fclass in (codes.RESIDENTIAL, codes.ATTACHED) or not f.ftype:
        return "unit", ""
    if f.fclass == codes.COMMERCIAL:
        return "commercial", ""
    if f.ftype in PRIVATE_LOT_TYPES:
        return "private_lot", ""
    return "other_sub", ""


@dataclass
class Site:
    """What shell detection knows about one parcel."""
    parcel: str
    feats: List[Feat]
    result: Optional[ParcelResult] = None
    premise: Dict[str, str] = field(default_factory=dict)       # pid -> premise key
    structure: Dict[str, str] = field(default_factory=dict)     # sub pid -> structure id
    parents: Dict[str, ParentInfo] = field(default_factory=dict)
    geo: Optional[ParcelGeo] = None
    point: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    footprint: Dict[str, str] = field(default_factory=dict)     # pid -> footprint id ('' = open ground)
    off_parcel: Dict[str, float] = field(default_factory=dict)  # pid -> metres outside its parcel
    # geocodes that say nothing about which building: at the parcel centroid / label point,
    # or shared by sub-addresses of two or more inferred buildings
    property_level: Set[str] = field(default_factory=set)
    level: Dict[str, str] = field(default_factory=dict)         # pid -> PROPERTY | FOOTPRINT | OPEN | OFF_PARCEL

    def building_level(self, pid: str) -> bool:
        return pid in self.point and pid not in self.property_level


@dataclass
class Verdict:
    f: Feat
    parcel: str
    kind: str
    logit: float
    cls: str
    terms: List[Tuple[str, float]]
    n_children: int = 0
    n_cross: int = 0
    cross_parcels: int = 0
    child_structures: int = 0
    child_footprints: int = 0
    notes: List[str] = field(default_factory=list)
    borrowed_from: str = ""

    @property
    def p(self) -> float:
        return sigmoid(self.logit)


def _key(x: float, y: float) -> Tuple[float, float]:
    return (round(x, 1), round(y, 1))


def build_sites(results: Iterable[ParcelResult], extra: List[Address], geo: Optional[GeoData],
                coords: Dict[str, Tuple[float, float]], cfg: Config) -> Dict[str, Site]:
    sites: Dict[str, Site] = {}
    for r in results:
        s = Site(r.parcel, list(r.feats), r)
        for g in r.groups:
            for f in g.members:
                s.premise[f.pid] = g.id
        st_of = {b.id: b.structure_id for b in r.blocks}
        for f in r.feats:
            s.premise.setdefault(f.pid, f.pid)
            if f.is_sub and st_of.get(f.block_id):
                s.structure[f.pid] = st_of[f.block_id]
        s.parents = {p.feat.pid: p for p in r.parents}
        sites[r.parcel] = s
    for a in extra:
        if not a.parcel:
            continue
        s = sites.setdefault(a.parcel, Site(a.parcel, []))
        f = build_feat(a)
        s.feats.append(f)
        s.premise[f.pid] = f.pid
    if geo is None:
        return sites
    for s in sites.values():
        s.geo = geo.parcels.get(s.parcel)
        if s.geo is None:
            continue
        for f in s.feats:
            lonlat = coords.get(f.pid)
            xy = geo.point(s.parcel, *lonlat) if lonlat else None
            if xy is None:
                continue
            off = s.geo.distance_to_parcel(*xy)
            if off is not None and off > cfg.shell_off_parcel_m:
                s.off_parcel[f.pid] = off
                s.level[f.pid] = "OFF_PARCEL"
                continue
            s.point[f.pid] = xy
            s.footprint[f.pid] = s.geo.locate(xy[0], xy[1], cfg.shell_snap_m)[0]
            if s.geo.is_property_point(xy[0], xy[1], cfg.shell_property_point_m):
                s.property_level.add(f.pid)
        structs: Dict[Tuple[float, float], Set[str]] = defaultdict(set)
        at: Dict[Tuple[float, float], List[str]] = defaultdict(list)
        for f in s.feats:
            if f.pid in s.point:
                at[_key(*s.point[f.pid])].append(f.pid)
                if f.is_sub and s.structure.get(f.pid):
                    structs[_key(*s.point[f.pid])].add(s.structure[f.pid])
        for k, sts in structs.items():
            if len(sts) >= 2:
                s.property_level.update(at[k])
        for pid in s.point:
            s.level[pid] = "PROPERTY" if pid in s.property_level else ("FOOTPRINT" if s.footprint[pid] else "OPEN")
    return sites


def _children(p: Feat, s: Site) -> List[Feat]:
    info = s.parents.get(p.pid)
    structs = set(info.structures) if info else set()
    out = []
    for f in s.feats:
        if not f.is_sub:
            continue
        if structs and s.structure.get(f.pid) in structs:
            out.append(f)
        elif covers(p, f) and (not p.bref[0] or compare_brefs(p.bref, f.bref) in ("equal", "ordinal")):
            out.append(f)
    return out


def _street_index(sites: Dict[str, Site]) -> Dict[str, List[Tuple[str, Feat]]]:
    idx: Dict[str, List[Tuple[str, Feat]]] = defaultdict(list)
    for s in sites.values():
        for f in s.feats:
            if f.is_sub:
                idx[f.street_key].append((s.parcel, f))
    return idx


def _located(s: Site, members: Iterable[Feat]) -> Optional[str]:
    """Majority footprint of a premise's members ('' = open ground), from
    building-level geocodes only; None when none of them has one."""
    fps = [s.footprint[f.pid] for f in members if s.building_level(f.pid)]
    return Counter(fps).most_common(1)[0][0] if fps else None


def _living_premises(s: Site) -> int:
    """Distinct dwelling or tenancy premises on the parcel."""
    return len({s.premise.get(f.pid, f.pid) for f in s.feats
                if f.is_sub and kind_of(f)[0] in ("unit", "commercial")})


def score_parent(p: Feat, s: Site, sites: Dict[str, Site], street_idx, geo: Optional[GeoData],
                 cfg: Config) -> Verdict:
    terms: List[Tuple[str, float]] = [("parent", cfg.shell_prior["parent"])]
    v = Verdict(p, s.parcel, "parent", 0.0, "", terms)
    kids = _children(p, s)
    by_premise: Dict[str, List[Feat]] = defaultdict(list)
    for f in kids:
        by_premise[s.premise.get(f.pid, f.pid)].append(f)
    n = len(by_premise)
    v.n_children = n
    # structures of dwellings and tenancies (car-space or lobby blocks don't make a site)
    v.child_structures = len({s.structure[f.pid] for f in kids if s.structure.get(f.pid)
                              and kind_of(f)[0] in ("unit", "commercial")})
    cap = cfg.shell_children_cap
    if n:
        terms.append(("%d sub-address premise%s below it%s" % (
            n, "s" if n > 1 else "", " in %d structures" % v.child_structures if v.child_structures > 1 else ""),
            cfg.w_shell_children * min(math.log2(1 + n), cap)))
    # premises on neighbouring parcels that hang off the same base (community / multi-lot schemes)
    near = None
    cross: Set[Tuple[str, str]] = set()
    for q, f in street_idx.get(p.street_key, ()):
        if q == s.parcel or not covers(p, f):
            continue
        if p.bref[0] and compare_brefs(p.bref, f.bref) not in ("equal", "ordinal"):
            continue
        if geo is not None and s.geo is not None and q in geo.parcels:
            if near is None:
                near = geo.neighbours(s.parcel, cfg.shell_cross_parcel_m)
            if q not in near:
                continue
        cross.add((q, sites[q].premise.get(f.pid, f.pid)))
    v.n_cross, v.cross_parcels = len(cross), len({q for q, _ in cross})
    if cross:
        terms.append(("%d premise%s on %d neighbouring parcel%s share its base" % (
            len(cross), "s" if len(cross) > 1 else "", v.cross_parcels, "s" if v.cross_parcels > 1 else ""),
            cfg.w_shell_cross_parcel * min(math.log2(1 + len(cross)), cap)))
    if not n and not cross:
        terms.append(("nothing below it", cfg.w_shell_no_children))
    if p.base_shape == "range":
        terms.append(("range number", cfg.w_shell_range))
    if is_common_name(p.bname):
        terms.append(("building name '%s'" % p.bname, cfg.w_shell_keyword))
        v.cls = "COMMON_PROPERTY"
    # ---- geometry: building-level geocodes first, then the parcel's footprints alone
    fp_counts: Counter = Counter()
    located_open = 0
    footprints = s.geo is not None and geo is not None and geo.has_buildings
    if s.geo is not None:
        for members in by_premise.values():
            loc = _located(s, members)
            if loc:
                fp_counts[loc] += 1
            elif loc is not None:
                located_open += 1
        v.child_footprints = len(fp_counts)
    point_terms = False
    if s.geo is not None and s.building_level(p.pid) and n:
        pp, mine = _key(*s.point[p.pid]), s.footprint.get(p.pid, "")
        colocated = sum(1 for ms in by_premise.values()
                        if any(s.building_level(m.pid) and _key(*s.point[m.pid]) == pp for m in ms))
        in_mine = fp_counts.get(mine, 0) if mine else 0
        in_other = sum(c for fp, c in fp_counts.items() if fp != mine)
        if mine and (in_mine or colocated):
            terms.append(("geocode in footprint %s with %d of them" % (mine, max(in_mine, colocated)),
                          cfg.w_shell_geo_with_children))
            point_terms = True
        elif mine and in_other and in_other >= 0.6 * (in_other + located_open):
            terms.append(("geocode alone in footprint %s; its sub-addresses are in other footprints "
                          "(a front house?)" % mine, cfg.w_shell_geo_own_building))
            point_terms = True
        elif not mine and fp_counts and footprints:
            # geocode on open ground: is there a house-sized building nobody is addressed in?
            used = {s.footprint[pid] for pid in s.point if s.building_level(pid) and s.footprint[pid]}
            empty = [b for b in s.geo.buildings if b not in used
                     and s.geo.building_area(b) >= cfg.shell_house_min_m2]
            if empty:
                terms.append(("geocode on open ground; footprint %s (%.0f m2) holds no address while its "
                              "sub-addresses are in other footprints (a front house?)"
                              % (empty[0], s.geo.building_area(empty[0])), cfg.w_shell_geo_unaddressed))
                point_terms = True
    elif p.pid in s.off_parcel:
        v.notes.append("geocode %.0f m outside its parcel (geometry not used)" % s.off_parcel[p.pid])
    if footprints:
        if not s.geo.buildings and not n and cross:
            terms.append(("parcel has no building (association or access lot)", cfg.w_shell_geo_bare_parcel))
        elif n and not point_terms and not cross:
            # no building-level geocode: compare the parcel's buildings with its premises
            dwellings = [b for b in s.geo.buildings if s.geo.building_area(b) >= cfg.shell_house_min_m2]
            living = _living_premises(s)
            if len(dwellings) == 1 and living >= 2:
                terms.append(("the parcel's one building holds its %d premises" % living,
                              cfg.w_shell_geo_with_children))
            elif living and len(dwellings) >= living + 1 and living <= cfg.shell_spare_building_max:
                terms.append(("%d dwelling-sized buildings for %d premise%s on the parcel: the base address "
                              "may be a house of its own" % (len(dwellings), living, "s" if living > 1 else ""),
                              cfg.w_shell_geo_spare_building))
            elif living and len(dwellings) >= 2 and living <= cfg.shell_small_scheme:
                terms.append(("%d premises spread over %d buildings: the base address may be a front house"
                              % (living, len(dwellings)), cfg.w_shell_geo_small_scheme))
    # ---- class
    if not v.cls and (n or cross):
        located = sum(fp_counts.values())
        dom = max(fp_counts.values()) / located if located else 0.0
        geo_site = len(fp_counts) >= 2 and (1 - dom) >= cfg.shell_site_share
        # one real building on the parcel holds everything, whatever the geocodes say
        big = [b for b in s.geo.buildings if s.geo.building_area(b) >= cfg.shell_tiny_footprint_m2] \
            if footprints else []
        geo_building = (located >= 3 and dom >= 0.9) or (len(big) == 1 and n and not cross)
        if geo_site or v.cross_parcels >= 2 or (not n and cross):
            v.cls = "SITE"
        elif geo_building:
            v.cls = "BUILDING"
        else:
            v.cls = "SITE" if v.child_structures >= 2 else "BUILDING"
    v.logit = sum(t for _, t in terms)
    return v


def score_sub(f: Feat, s: Site, cfg: Config) -> Verdict:
    kind, cls = kind_of(f)
    terms: List[Tuple[str, float]] = [(kind.replace("_", " "), cfg.shell_prior.get(kind, 0.0))]
    v = Verdict(f, s.parcel, kind, 0.0, cls, terms)
    if is_common_name(f.bname) and kind not in ("common_type", "common_level"):
        terms.append(("building name '%s'" % f.bname, cfg.w_shell_keyword))
        v.cls = v.cls or "COMMON_PROPERTY"
    if s.geo is not None and s.building_level(f.pid) and kind != "unit":
        fp = s.footprint.get(f.pid, "")
        if fp and s.geo.building_area(fp) < cfg.shell_tiny_footprint_m2:
            terms.append(("geocode in a %.0f m2 footprint (kiosk, substation, plant)" % s.geo.building_area(fp),
                          cfg.w_shell_geo_tiny_footprint))
            v.cls = v.cls or "COMMON_PROPERTY"
    elif f.pid in s.off_parcel:
        v.notes.append("geocode %.0f m outside its parcel (geometry not used)" % s.off_parcel[f.pid])
    v.logit = sum(t for _, t in terms)
    return v


def detect(sites: Dict[str, Site], geo: Optional[GeoData], cfg: Config) -> Dict[str, Verdict]:
    street_idx = _street_index(sites)
    out: Dict[str, Verdict] = {}
    for s in sites.values():
        for f in s.feats:
            out[f.pid] = (score_sub(f, s, cfg) if f.is_sub else
                          score_parent(f, s, sites, street_idx, geo, cfg))
        # one verdict per duplicate group: its strongest member's
        groups: Dict[str, List[Feat]] = defaultdict(list)
        for f in s.feats:
            groups[s.premise.get(f.pid, f.pid)].append(f)
        for members in groups.values():
            if len(members) < 2:
                continue
            best = max(members, key=lambda f: out[f.pid].logit)
            for f in members:
                v = out[f.pid]
                if f is not best and out[best.pid].logit > v.logit:
                    b = out[best.pid]
                    v.logit, v.cls, v.borrowed_from = b.logit, b.cls, best.pid
    return out


def verdict_of(p: float, cfg: Config) -> str:
    if p >= cfg.shell_threshold:
        return "SHELL"
    if p >= cfg.shell_possible:
        return "POSSIBLE_SHELL"
    return "NOT_SHELL"


def decide(sites: Dict[str, Site], verdicts: Dict[str, Verdict], base_type: Dict[str, str],
           other: Optional[Dict[str, Optional[bool]]], cfg: Config) -> pd.DataFrame:
    """One row per ADDR_ID."""
    has_type = any(base_type.values())
    rows = []
    for s in sites.values():
        for f in s.feats:
            v = verdicts[f.pid]
            p = v.p
            verdict = verdict_of(p, cfg)
            t = base_type.get(f.pid, "")
            base_res = (t in cfg.residential_address_types) if has_type else None
            base_shell = (t in cfg.shell_address_types) if has_type and cfg.shell_address_types else None
            o = other.get(f.pid) if other else None
            action = []
            if verdict == "SHELL":
                if base_res or o:
                    cat = "SHELL_FLAGGED_RESIDENTIAL"
                    if base_res:
                        action.append("base ADDRESS_TYPE %s -> a shell type" % t)
                    if o:
                        action.append("other -> not residential")
                elif base_shell is False:
                    cat = "SHELL_NOT_TYPED"
                    action.append("ADDRESS_TYPE %s -> a shell type" % (t or "(blank)"))
                else:
                    cat = "OK"
            elif verdict == "NOT_SHELL" and base_shell:
                cat = "TYPED_SHELL_NOT_DETECTED"
                action.append("review: looks like a %s" % ("dwelling" if v.kind in ("unit", "parent")
                                                          else v.kind.replace("_", " ")))
            elif verdict == "POSSIBLE_SHELL" and not base_shell:
                cat = "REVIEW"
                action.append("review")
            else:
                cat = "OK"
            ev = "; ".join("%s %+.1f" % (k, x) for k, x in v.terms)
            if v.borrowed_from:
                ev += "; verdict of duplicate %s" % v.borrowed_from
            if v.notes:
                ev += "; " + "; ".join(v.notes)
            rows.append({
                "ADDR_ID": f.pid, "SOURCE_ID": f.a.source_id, "CADASTRE_POLYGON_PID": s.parcel,
                "LABEL": f.label, "KIND": v.kind.upper(), "P_SHELL": round(p, 4), "VERDICT": verdict,
                "SHELL_CLASS": v.cls if verdict != "NOT_SHELL" else "",
                "N_CHILD_PREMISES": v.n_children if not f.is_sub else None,
                "N_CROSS_PARCEL_PREMISES": v.n_cross if not f.is_sub else None,
                "N_CHILD_FOOTPRINTS": v.child_footprints if not f.is_sub else None,
                "BUILDING_ID": s.footprint.get(f.pid, "") if s.building_level(f.pid) else None,
                "GEOCODED": f.pid in s.point, "GEOCODE_LEVEL": s.level.get(f.pid, ""),
                "PREMISE_GROUP": s.premise.get(f.pid, "") if s.premise.get(f.pid, f.pid) != f.pid else "",
                "BASE_ADDRESS_TYPE": t, "BASE_SAYS_SHELL": base_shell, "BASE_RESIDENTIAL": base_res,
                "OTHER_RESIDENTIAL": o, "CATEGORY": cat, "SUGGESTED_ACTION": "; ".join(action),
                "EVIDENCE": ev,
            })
    return pd.DataFrame(rows)


def building_table(sites: Dict[str, Site], table: pd.DataFrame) -> pd.DataFrame:
    """Per footprint: what has a building-level geocode in it, and which shells.
    Property-level geocodes (parcel centroid) are not counted."""
    verdict = dict(zip(table["ADDR_ID"], table["VERDICT"]))
    rows = []
    for s in sites.values():
        if s.geo is None:
            continue
        inside: Dict[str, List[Feat]] = defaultdict(list)
        for f in s.feats:
            if s.building_level(f.pid) and s.footprint.get(f.pid):
                inside[s.footprint[f.pid]].append(f)
        for bid, poly in s.geo.buildings.items():
            fs = inside.get(bid, [])
            shells = [f for f in fs if verdict.get(f.pid) == "SHELL"]
            rows.append({
                "CADASTRE_POLYGON_PID": s.parcel, "BUILDING_ID": bid, "AREA_M2": round(float(poly.area), 1),
                "N_BUILDING_LEVEL_GEOCODES": len(fs),
                "N_SUB_PREMISES": len({s.premise.get(f.pid, f.pid) for f in fs if f.is_sub}),
                "N_PARENTS": sum(1 for f in fs if not f.is_sub), "N_SHELLS": len(shells),
                "SHELLS": " | ".join(f.label for f in shells[:5]),
            })
    return pd.DataFrame(rows, columns=["CADASTRE_POLYGON_PID", "BUILDING_ID", "AREA_M2",
                                       "N_BUILDING_LEVEL_GEOCODES", "N_SUB_PREMISES", "N_PARENTS", "N_SHELLS",
                                       "SHELLS"])


def parcel_table(sites: Dict[str, Site], table: pd.DataFrame) -> pd.DataFrame:
    """Per parcel: shells by class, possible shells, flagged-residential shells, geometry."""
    if table.empty:
        return pd.DataFrame()
    t = table.assign(_s=table["VERDICT"] == "SHELL", _p=table["VERDICT"] == "POSSIBLE_SHELL",
                     _r=table["CATEGORY"] == "SHELL_FLAGGED_RESIDENTIAL", _g=table["GEOCODED"],
                     _b=table["GEOCODE_LEVEL"].isin(["FOOTPRINT", "OPEN"]))
    g = t.groupby("CADASTRE_POLYGON_PID")
    out = pd.DataFrame({"N_ADDRESSES": g.size(), "N_SHELLS": g["_s"].sum(), "N_POSSIBLE": g["_p"].sum(),
                        "N_SHELLS_FLAGGED_RESIDENTIAL": g["_r"].sum(), "N_GEOCODED": g["_g"].sum(),
                        "N_BUILDING_LEVEL_GEOCODES": g["_b"].sum()})
    cls = pd.crosstab(t.loc[t._s, "CADASTRE_POLYGON_PID"], t.loc[t._s, "SHELL_CLASS"])
    for c in ("BUILDING", "SITE", "COMMON_PROPERTY", "FLOOR"):
        out[c] = cls[c] if c in cls.columns else 0
    out = out.fillna(0).astype(int)
    geo = {p: s.geo for p, s in sites.items()}
    out["N_FOOTPRINTS"] = [len(geo[p].buildings) if geo.get(p) is not None else None for p in out.index]
    out["PARCEL_AREA_M2"] = [round(geo[p].area, 1) if geo.get(p) is not None and geo[p].polygon is not None
                             else None for p in out.index]
    out = out.reset_index()
    return out.sort_values(["N_SHELLS", "N_POSSIBLE", "N_ADDRESSES"], ascending=False).reset_index(drop=True)


def geocode_issues(sites: Dict[str, Site], coords: Dict[str, Tuple[float, float]],
                   geo: Optional[GeoData]) -> pd.DataFrame:
    rows = []

    def issue(check, severity, ids, detail):
        if ids:
            rows.append({"CHECK": check, "SEVERITY": severity, "ROWS": len(ids), "DETAIL": detail,
                         "EXAMPLE_IDS": ", ".join(sorted(ids)[:5])})
    if geo is not None:
        no_geo = [f.pid for s in sites.values() if s.geo is None for f in s.feats]
        issue("parcel_without_geometry", "warning", no_geo, "addresses on parcels missing from the "
              "cadastre and footprints (scored from text alone)")
        off = [pid for s in sites.values() for pid in s.off_parcel]
        issue("geocode_off_parcel", "warning", off, "geocodes far outside their own parcel (geometry ignored "
              "for these addresses)")
        no_pt = [f.pid for s in sites.values() if s.geo is not None for f in s.feats
                 if f.pid not in coords]
        issue("no_geocode", "info", no_pt, "addresses without LATITUDE/LONGITUDE")
        prop = [pid for s in sites.values() for pid in s.property_level]
        issue("property_level_geocode", "info", prop, "geocodes at the parcel centroid or label point, or shared "
              "by several buildings' addresses: they say nothing about which building an address is in, so "
              "only the parcel's footprints are used for these")
    return pd.DataFrame(rows, columns=["CHECK", "SEVERITY", "ROWS", "DETAIL", "EXAMPLE_IDS"])
