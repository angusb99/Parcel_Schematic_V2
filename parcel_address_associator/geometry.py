"""Cadastre parcels and building footprints in metres around each parcel.

Geometry arrives as WKT (or shapely geometries). Geographic coordinates (lon/lat,
as Geoscape supplies GDA94/GDA2020) are projected per parcel onto a local plane
centred on the parcel: equirectangular with ellipsoidal scale factors, accurate
to centimetres at parcel scale, so no CRS database is needed. Projected WKT
(metres) is used as is, and address LATITUDE/LONGITUDE are then transformed with
pyproj (``crs="EPSG:7856"`` etc.).

    geo = prepare_geometry(cadastre_df, buildings_df)
    geo.issues                    # what was parsed, repaired, merged or skipped
    pg = geo.parcels["cpl..."]    # ParcelGeo: polygon and footprints in metres
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.ops import polylabel

WKT_COLUMNS = ("WKT", "GEOMETRY", "GEOM", "WKT_GEOMETRY", "GEOMETRY_WKT", "SHAPE")
BUILDING_ID_COLUMNS = ("BUILDING_ID", "BUILDING_PID", "BLD_PID", "FOOTPRINT_ID", "BUILDING_FOOTPRINT_PID")
GEOGRAPHIC_CRS = {"GEOGRAPHIC", "EPSG:4326", "EPSG:4283", "EPSG:7844", "EPSG:4939", "EPSG:7843"}


def metres_per_degree(lat: float) -> Tuple[float, float]:
    """(metres per degree of longitude, of latitude) at ``lat`` on the GRS80 ellipsoid."""
    p = math.radians(lat)
    m_lat = 111132.954 - 559.822 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    m_lon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)
    return m_lon, m_lat


@dataclass
class Frame:
    """Local plane: metres east/north of (x0, y0)."""
    x0: float
    y0: float
    kx: float = 1.0
    ky: float = 1.0

    def geom(self, g):
        return shapely.transform(g, lambda c: (c - (self.x0, self.y0)) * (self.kx, self.ky))

    def xy(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.x0) * self.kx, (y - self.y0) * self.ky


@dataclass
class ParcelGeo:
    parcel: str
    frame: Frame
    polygon: object = None                                   # parcel in metres (None: footprints only)
    buildings: Dict[str, object] = field(default_factory=dict)   # footprint id -> polygon in metres
    area: float = 0.0
    width: float = 0.0                                       # short side of the oriented envelope
    length: float = 0.0
    # where a property-centroid geocode would sit: centroid, label point, pole of inaccessibility
    anchors: List[Tuple[float, float]] = field(default_factory=list)

    def is_property_point(self, x: float, y: float, tol: float = 2.0) -> bool:
        """Is (x, y) the parcel's own centroid (or label point)? Such a geocode says
        nothing about which building on the parcel an address is in."""
        return any(math.hypot(x - ax, y - ay) <= tol for ax, ay in self.anchors)

    def building_area(self, bid: str) -> float:
        return float(self.buildings[bid].area) if bid in self.buildings else 0.0

    def locate(self, x: float, y: float, snap: float = 1.0) -> Tuple[str, float]:
        """(footprint id containing the point or within ``snap`` m, or '';
        distance to the nearest footprint, inf when there is none)."""
        if not self.buildings:
            return "", math.inf
        p = shapely.Point(x, y)
        best, dist = "", math.inf
        for bid, poly in self.buildings.items():
            d = 0.0 if poly.covers(p) else poly.distance(p)
            # inside two overlapping footprints: the smaller is the more specific
            if d < dist or (d == dist == 0.0 and best and poly.area < self.buildings[best].area):
                best, dist = bid, d
        return (best if dist <= snap else ""), dist

    def distance_to_parcel(self, x: float, y: float) -> Optional[float]:
        if self.polygon is None:
            return None
        p = shapely.Point(x, y)
        return 0.0 if self.polygon.covers(p) else float(self.polygon.distance(p))


@dataclass
class GeoData:
    parcels: Dict[str, ParcelGeo]
    geographic: bool
    issues: pd.DataFrame
    transformer: object = None              # pyproj lon/lat -> projected CRS, when the WKT is projected
    has_buildings: bool = False             # footprints were supplied (so 'no building' means something)
    _src: Dict[str, object] = field(default_factory=dict)    # parcel polygons in source coordinates
    _tree: Optional[STRtree] = None
    _tree_ids: List[str] = field(default_factory=list)

    def point(self, parcel: str, lon: float, lat: float) -> Optional[Tuple[float, float]]:
        """An address geocode in its parcel's local frame (None if either is missing)."""
        pg = self.parcels.get(parcel)
        if pg is None or lon is None or lat is None or lon != lon or lat != lat:
            return None
        if not self.geographic:
            if self.transformer is None:
                return None
            lon, lat = self.transformer.transform(lon, lat)
        return pg.frame.xy(lon, lat)

    def neighbours(self, parcel: str, metres: float) -> Set[str]:
        """Parcels within ``metres`` of this one (itself excluded)."""
        g = self._src.get(parcel)
        if g is None or self._tree is None:
            return set()
        if self.geographic:
            m_lon, m_lat = metres_per_degree(g.centroid.y)
            d = metres / min(m_lon, m_lat)
        else:
            d = metres
        idx = self._tree.query(g.buffer(d), predicate="intersects")
        return {self._tree_ids[i] for i in idx} - {parcel}


def _column(df: pd.DataFrame, wanted: Optional[str], candidates, what: str, required: bool = True):
    cols = {str(c).strip().upper(): c for c in df.columns}
    if wanted:
        if wanted.upper() not in cols:
            raise ValueError("%s has no column %s (have %s)" % (what, wanted, list(df.columns)))
        return cols[wanted.upper()]
    for c in candidates:
        if c in cols:
            return cols[c]
    geom = getattr(df, "geometry", None)        # a GeoDataFrame
    if candidates is WKT_COLUMNS and geom is not None and hasattr(geom, "name"):
        return geom.name
    if required:
        raise ValueError("%s has no geometry column (looked for %s)" % (what, ", ".join(candidates)))
    return None


def _ids(series: pd.Series) -> pd.Series:
    return series.map(lambda v: "" if v is None or (isinstance(v, float) and v != v)
                      else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()))


def _polygonal(g):
    """Polygon/MultiPolygon or None; invalid rings repaired. Returns (geom, repaired)."""
    if g is None or g.is_empty:
        return None, False
    repaired = False
    if not g.is_valid:
        g, repaired = shapely.make_valid(g), True
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g, repaired
    if g.geom_type == "GeometryCollection":
        polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        return (shapely.union_all(polys) if polys else None), repaired
    return None, repaired


def _parse(values: pd.Series, what: str, ids: pd.Series, issue) -> List[object]:
    vals = list(values)
    if all(v is None or isinstance(v, str) or (isinstance(v, float) and v != v) for v in vals):
        arr = np.array([v if isinstance(v, str) and v.strip() else None for v in vals], dtype=object)
        geoms = list(shapely.from_wkt(arr, on_invalid="ignore"))
        unparsed = [i for i, (v, g) in enumerate(zip(arr, geoms)) if v is not None and g is None]
        issue("unparsed_wkt", "warning", len(unparsed), "%s rows whose WKT could not be read" % what,
              [ids.iloc[i] for i in unparsed])
    else:
        geoms = [v if hasattr(v, "geom_type") else None for v in vals]
    out, not_poly, fixed = [], [], []
    for i, g in enumerate(geoms):
        p, rep = _polygonal(g)
        if g is not None and p is None:
            not_poly.append(ids.iloc[i])
        if rep and p is not None:
            fixed.append(ids.iloc[i])
        out.append(p)
    issue("not_polygonal", "warning", len(not_poly), "%s geometries that are not polygons (skipped)" % what,
          not_poly)
    issue("geometry_repaired", "info", len(fixed), "%s polygons that were invalid and repaired" % what, fixed)
    blank = sum(1 for v in vals if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str)
                                                                                      and not v.strip()))
    issue("blank_geometry", "warning", blank, "%s rows without geometry" % what)
    return out


def prepare_geometry(cadastre: Optional[pd.DataFrame] = None, buildings: Optional[pd.DataFrame] = None,
                     crs: Optional[str] = None, wkt_col: Optional[str] = None,
                     building_wkt_col: Optional[str] = None, building_id_col: Optional[str] = None,
                     parcel_col: str = "CADASTRE_POLYGON_PID") -> GeoData:
    """Parse the cadastre [CADASTRE_POLYGON_PID, WKT] and building footprints
    [BUILDING_ID (optional), CADASTRE_POLYGON_PID, WKT]. Either may be None.

    Footprints without a parcel column are linked to the parcel they overlap most.
    ``crs``: None detects lon/lat vs projected from the coordinates; pass e.g.
    'EPSG:7856' for projected WKT so address LATITUDE/LONGITUDE can be placed.
    """
    issues: List[Dict[str, object]] = []

    def issue(check, severity, n, detail, examples=()):
        if n:
            issues.append({"CHECK": check, "SEVERITY": severity, "ROWS": int(n), "DETAIL": detail,
                           "EXAMPLE_IDS": ", ".join(str(x) for x in list(examples)[:5])})

    parcel_src: Dict[str, object] = {}
    if cadastre is not None and len(cadastre):
        pcol = _column(cadastre, parcel_col, (), "cadastre")
        ids = _ids(cadastre[pcol])
        geoms = _parse(cadastre[_column(cadastre, wkt_col, WKT_COLUMNS, "cadastre")], "cadastre", ids, issue)
        multi: Dict[str, List[object]] = {}
        for pid, g in zip(ids, geoms):
            if pid and g is not None:
                multi.setdefault(pid, []).append(g)
        merged = [p for p, gs in multi.items() if len(gs) > 1]
        issue("parcel_rows_merged", "info", len(merged), "parcels supplied as several rows; parts merged", merged)
        parcel_src = {p: (gs[0] if len(gs) == 1 else shapely.union_all(gs)) for p, gs in multi.items()}
        issue("parcels_loaded", "info", len(parcel_src), "%d parcel polygons loaded" % len(parcel_src))

    bld_src: List[Tuple[str, str, object]] = []          # (footprint id, parcel, geometry)
    if buildings is not None and len(buildings):
        bcol = _column(buildings, building_id_col, BUILDING_ID_COLUMNS, "buildings", required=False)
        pcol = _column(buildings, parcel_col, (), "buildings", required=False) if any(
            str(c).strip().upper() == parcel_col for c in buildings.columns) else None
        bids = _ids(buildings[bcol]) if bcol is not None else pd.Series(
            ["B%07d" % (i + 1) for i in range(len(buildings))], index=buildings.index)
        geoms = _parse(buildings[_column(buildings, building_wkt_col or wkt_col, WKT_COLUMNS, "buildings")],
                       "building", bids, issue)
        if pcol is not None:
            pids = list(_ids(buildings[pcol]))
            nolink = [b for b, p, g in zip(bids, pids, geoms) if g is not None and not p]
            issue("building_without_parcel", "warning", len(nolink),
                  "footprints with a blank %s (skipped)" % parcel_col, nolink)
        else:
            pids = [""] * len(geoms)
            if parcel_src:
                order = list(parcel_src)
                tree = STRtree([parcel_src[p] for p in order])
                for i, g in enumerate(geoms):
                    if g is None:
                        continue
                    hits = tree.query(g, predicate="intersects")
                    if len(hits):
                        pids[i] = max((order[h] for h in hits), key=lambda p: parcel_src[p].intersection(g).area)
                issue("buildings_linked_spatially", "info", sum(1 for p in pids if p),
                      "no %s on the footprints: each linked to the parcel it overlaps most" % parcel_col)
        seen = set()
        for b, p, g in zip(bids, pids, geoms):
            if g is None or not p:
                continue
            if (b, p) in seen:
                b = "%s#%d" % (b, len(seen))
            seen.add((b, p))
            bld_src.append((b, p, g))
        issue("buildings_loaded", "info", len(bld_src), "%d footprint-parcel links loaded" % len(bld_src))

    allg = list(parcel_src.values()) + [g for _, _, g in bld_src]
    if crs is None:
        if allg:
            minx, miny, maxx, maxy = shapely.total_bounds(np.array(allg, dtype=object))
            geographic = -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90
        else:
            geographic = True
    else:
        geographic = crs.strip().upper() in GEOGRAPHIC_CRS
    transformer = None
    if not geographic:
        if crs is None:
            issue("projected_without_crs", "error", 1, "WKT looks projected (metres) but no crs was given: "
                  "address geocodes cannot be placed; pass crs='EPSG:....'")
        else:
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:7844", crs, always_xy=True)
            except Exception as exc:                        # pragma: no cover - environment specific
                issue("crs_unavailable", "error", 1, "cannot transform geocodes to %s: %s" % (crs, exc))
    issue("coordinates", "info", 1, "geometry read as %s" % ("longitude/latitude (projected per parcel "
                                                              "to local metres)" if geographic else
                                                              "projected metres (%s)" % (crs or "unknown CRS")))

    by_parcel: Dict[str, List[Tuple[str, object]]] = {}
    for b, p, g in bld_src:
        by_parcel.setdefault(p, []).append((b, g))
    parcels: Dict[str, ParcelGeo] = {}
    for pid in set(parcel_src) | set(by_parcel):
        poly = parcel_src.get(pid)
        anchor = poly if poly is not None else shapely.union_all([g for _, g in by_parcel[pid]])
        c = anchor.centroid
        if geographic:
            kx, ky = metres_per_degree(c.y)
            fr = Frame(c.x, c.y, kx, ky)
        else:
            fr = Frame(c.x, c.y)
        pg = ParcelGeo(pid, fr, fr.geom(poly) if poly is not None else None,
                       {b: fr.geom(g) for b, g in by_parcel.get(pid, [])})
        if pg.polygon is not None:
            pg.area = float(pg.polygon.area)
            env = shapely.oriented_envelope(pg.polygon)
            xs, ys = env.exterior.coords.xy if env.geom_type == "Polygon" else ([0], [0])
            sides = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(min(2, len(xs) - 1))]
            if sides:
                pg.width, pg.length = min(sides), max(sides)
            main = max(pg.polygon.geoms, key=lambda g: g.area) if pg.polygon.geom_type == "MultiPolygon" \
                else pg.polygon
            pts = [pg.polygon.centroid, pg.polygon.representative_point()]
            try:
                pts.append(polylabel(main, tolerance=0.5))
            except Exception:                               # pragma: no cover - degenerate rings
                pass
            pg.anchors = [(q.x, q.y) for q in pts]
        parcels[pid] = pg
    orphan = sorted(set(by_parcel) - set(parcel_src))
    issue("buildings_on_unknown_parcel", "info", len(orphan),
          "footprints linked to parcels missing from the cadastre (kept; no parcel outline)", orphan)
    tree_ids = list(parcel_src)
    tree = STRtree([parcel_src[p] for p in tree_ids]) if tree_ids else None
    cols = ["CHECK", "SEVERITY", "ROWS", "DETAIL", "EXAMPLE_IDS"]
    return GeoData(parcels, geographic, pd.DataFrame(issues, columns=cols), transformer, bool(bld_src),
                   parcel_src, tree, tree_ids)
