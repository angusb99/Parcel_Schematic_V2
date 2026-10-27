"""DataFrame in -> per-parcel results out. No I/O."""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import codes
from .config import Config
from .model import GEO_COLUMNS, INPUT_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, Address
from .reconcile import ParcelResult, analyse_parcel

_IDENT = re.compile(r"^([A-Z]*)(\d+)([A-Z]*)$")
# (prefix, number, suffix) column triples whose number may arrive as 'G01' / '12A'
_ID_TRIPLES = [("FLAT_NUMBER_PREFIX", "FLAT_NUMBER", "FLAT_NUMBER_SUFFIX"),
               ("LEVEL_NUMBER_PREFIX", "LEVEL_NUMBER", "LEVEL_NUMBER_SUFFIX"),
               ("NUMBER_FIRST_PREFIX", "NUMBER_FIRST", "NUMBER_FIRST_SUFFIX"),
               ("NUMBER_LAST_PREFIX", "NUMBER_LAST", "NUMBER_LAST_SUFFIX")]
_CODE_COLUMNS = [("FLAT_TYPE_CODE", codes.FLAT_TYPE_WORDS, codes.FLAT_TYPE_AUT),
                 ("LEVEL_TYPE_CODE", codes.LEVEL_TYPE_WORDS, codes.LEVEL_TYPE_AUT),
                 ("STREET_TYPE_CODE", codes.STREET_TYPE_WORDS, codes.STREET_TYPE_AUT),
                 ("STREET_SUFFIX_CODE", codes.STREET_SUFFIX_WORDS, codes.STREET_SUFFIX_AUT)]


ID_COLUMNS = ("ADDR_ID", "SOURCE_ID", "CADASTRE_POLYGON_PID")


def clean_text(s: pd.Series, upper: bool = True) -> pd.Series:
    """Any dtype -> trimmed strings (upper-cased unless ``upper=False``);
    NaN/None -> ''; 101.0 -> '101'."""
    if pd.api.types.is_float_dtype(s):
        return s.map(lambda v: "" if pd.isna(v) else
                     (str(int(v)) if float(v).is_integer() else str(v))).astype(object)
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.astype("string").fillna("").astype(object)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d").fillna("").astype(object)

    def one(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        t = str(v).strip()
        if t.upper() in ("NAN", "NONE", "NULL", "<NA>"):
            return ""
        return re.sub(r"\s+", " ", t.upper() if upper else t)
    return s.map(one).astype(object)


def prepare_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalise the input. Returns (clean frame, issues frame).

    Raises ValueError only when a required column (ADDR_ID, CADASTRE_POLYGON_PID)
    is missing; everything else is repaired or reported.
    """
    issues: List[Dict[str, object]] = []

    def issue(check, severity, mask_or_n, detail, ids=None):
        n = int(mask_or_n.sum()) if hasattr(mask_or_n, "sum") else int(mask_or_n)
        if n:
            issues.append({"CHECK": check, "SEVERITY": severity, "ROWS": n, "DETAIL": detail,
                           "EXAMPLE_ADDR_IDS": ", ".join(list(ids)[:5]) if ids is not None else ""})

    lookup = {str(c).strip().upper(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in lookup]
    if missing:
        raise ValueError("Input is missing required column(s): %s" % ", ".join(missing))
    out = pd.DataFrame(index=df.index)
    for c in INPUT_COLUMNS:
        if c in lookup:
            out[c] = clean_text(df[lookup[c]], upper=c not in ID_COLUMNS)
        else:
            out[c] = ""
            issue("missing_column", "warning", len(df), "%s not supplied; treated as blank" % c)
    for c in OPTIONAL_COLUMNS:
        if c in lookup:
            out[c] = clean_text(df[lookup[c]])
        else:
            out[c] = ""
            issue("optional_column_absent", "info", len(df), "%s not supplied; use reconciliation "
                  "will rely on the second dataset alone" % c)
    _geocodes(df, lookup, out, issue)
    n_in = len(out)

    blank = out["ADDR_ID"] == ""
    issue("blank_addr_id", "error", blank, "rows without ADDR_ID were dropped")
    out = out[~blank]
    dup = out["ADDR_ID"].duplicated(keep="first")
    issue("duplicate_addr_id", "error", dup, "repeated ADDR_ID; first occurrence kept",
          out.loc[dup, "ADDR_ID"])
    out = out[~dup]
    nopar = out["CADASTRE_POLYGON_PID"] == ""
    issue("blank_parcel", "warning", nopar, "no CADASTRE_POLYGON_PID; excluded from analysis",
          out.loc[nopar, "ADDR_ID"])

    for col, words, valid in _CODE_COLUMNS:
        v = out[col]
        mapped = v.map(lambda x: x if (not x or x in valid) else words.get(x, x))
        changed = mapped != v
        issue("code_normalised", "info", changed, "%s names/abbreviations mapped to G-NAF codes "
              "(e.g. %s)" % (col, ", ".join(sorted({"%s->%s" % (a, b) for a, b in
                                                    zip(v[changed], mapped[changed])})[:4])))
        out[col] = mapped
        unknown = (mapped != "") & ~mapped.isin(set(valid))
        issue("unknown_code", "info", unknown, "%s values not in the G-NAF authority table: %s"
              % (col, ", ".join(sorted(set(mapped[unknown]))[:6])), out.loc[unknown, "ADDR_ID"])

    for pcol, ncol, scol in _ID_TRIPLES:
        num = out[ncol]
        packed = (num != "") & ~num.str.isdigit() & (out[pcol] == "") & (out[scol] == "")
        if packed.any():
            parts = num[packed].map(lambda t: _IDENT.match(t.replace(" ", "")))
            ok = parts.map(lambda m: m is not None)
            idx = parts[ok].index
            out.loc[idx, pcol] = parts[ok].map(lambda m: m.group(1))
            out.loc[idx, ncol] = parts[ok].map(lambda m: str(int(m.group(2))))
            out.loc[idx, scol] = parts[ok].map(lambda m: m.group(3))
            issue("identifier_split", "info", ok, "%s held prefix/suffix letters (e.g. G01); split "
                  "into %s/%s" % (ncol, pcol, scol), out.loc[idx, "ADDR_ID"])
        bad = (out[ncol] != "") & ~out[ncol].str.isdigit()
        issue("non_numeric_number", "warning", bad, "%s is not numeric (e.g. %s); used as text"
              % (ncol, ", ".join(sorted(set(out.loc[bad, ncol]))[:4])), out.loc[bad, "ADDR_ID"])

    nobase = (out["NUMBER_FIRST"] == "") & (out["LOT_NUMBER"] == "")
    issue("no_street_number", "info", nobase, "neither NUMBER_FIRST nor LOT_NUMBER",
          out.loc[nobase, "ADDR_ID"])
    issue("rows_accepted", "info", len(out), "%d of %d input rows accepted" % (len(out), n_in))
    issues_df = pd.DataFrame(issues, columns=["CHECK", "SEVERITY", "ROWS", "DETAIL", "EXAMPLE_ADDR_IDS"])
    return out.reset_index(drop=True), issues_df


def _geocodes(df: pd.DataFrame, lookup: Dict[str, str], out: pd.DataFrame, issue) -> None:
    """LATITUDE / LONGITUDE -> floats (NaN when absent or unusable)."""
    present = [c for c in GEO_COLUMNS if c in lookup]
    if len(present) < len(GEO_COLUMNS):
        for c in GEO_COLUMNS:
            out[c] = float("nan")
        issue("geocodes_absent", "info", len(df), "LATITUDE/LONGITUDE not supplied; shell detection "
              "uses address text and parcel linkage only")
        return
    for c in GEO_COLUMNS:
        raw = df[lookup[c]]
        v = pd.to_numeric(raw, errors="coerce").astype(float)
        bad = v.isna() & (clean_text(raw) != "")
        issue("non_numeric_coordinate", "warning", bad, "%s not numeric; treated as missing" % c,
              out.loc[bad, "ADDR_ID"])
        out[c] = v
    lat, lon = out["LATITUDE"], out["LONGITUDE"]
    swapped = (lat.abs() > 90) & (lon.abs() <= 90)
    if swapped.any() and swapped.sum() >= 0.5 * (lat.notna() & lon.notna()).sum():
        issue("coordinates_swapped", "warning", swapped, "LATITUDE and LONGITUDE look swapped; swapped back",
              out.loc[swapped, "ADDR_ID"])
        out.loc[swapped, ["LATITUDE", "LONGITUDE"]] = out.loc[swapped, ["LONGITUDE", "LATITUDE"]].values
        lat, lon = out["LATITUDE"], out["LONGITUDE"]
    bad = (lat.abs() > 90) | (lon.abs() > 180)
    issue("coordinate_out_of_range", "warning", bad, "geocode outside -90..90 / -180..180; treated as missing",
          out.loc[bad, "ADDR_ID"])
    out.loc[bad, ["LATITUDE", "LONGITUDE"]] = float("nan")


def analyse(clean: pd.DataFrame, cfg: Config, parcels: Optional[Iterable[str]] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, ParcelResult]:
    """Run the engine parcel by parcel on a frame from ``prepare_input``."""
    df = clean[clean["CADASTRE_POLYGON_PID"] != ""]
    if parcels is not None:
        df = df[df["CADASTRE_POLYGON_PID"].isin(set(parcels))]
    sizes = df.groupby("CADASTRE_POLYGON_PID").size()
    keep = set(sizes[sizes >= cfg.min_addresses].index)
    df = df[df["CADASTRE_POLYGON_PID"].isin(keep)]
    results: Dict[str, ParcelResult] = {}
    groups = df.groupby("CADASTRE_POLYGON_PID", sort=False)
    total = groups.ngroups
    for i, (pid, g) in enumerate(groups, 1):
        addrs = [Address.from_row(r) for r in g.to_dict("records")]
        results[pid] = analyse_parcel(pid, addrs, cfg)
        if progress is not None:
            progress(i, total)
    return results


def run(df: pd.DataFrame, cfg: Optional[Config] = None) -> Tuple[Dict[str, ParcelResult], pd.DataFrame]:
    """One call, no notebook: returns (results by parcel, validation issues)."""
    cfg = cfg or Config()
    clean, issues = prepare_input(df)
    return analyse(clean, cfg), issues
