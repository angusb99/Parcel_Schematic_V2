"""Notebook session handler: run the engine stage by stage and inspect every step.

    from parcel_address_associator import AssociatorSession
    s = AssociatorSession(df)        # df: one row per address, INPUT_COLUMNS
    s.validate()                     # 1. schema and data checks -> issues table
    s.profile()                      # 2. what each parcel holds
    s.run()                          # 3. analyse every parcel
    s.overview()                     # 4. headline numbers
    s.show_parcel(pid)               # 5. one parcel end to end (styles, mappings, stacking plans)
    s.duplicate_groups()             # 6. results as DataFrames
    s.explain(addr_id)               # 7. why one address was (or was not) grouped
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Dict, Iterable, List, Optional

import pandas as pd

from . import tables
from .config import Config
from .engine import analyse, prepare_input
from .evaluate import evaluate as _evaluate
from .reconcile import ParcelResult
from .render import overview_html, parcel_html


def _display_html(html: str):
    """Render in a notebook if possible; otherwise hand the HTML back."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            display(HTML(html))
            return None
    except ImportError:
        pass
    return html


class AssociatorSession:
    def __init__(self, df: pd.DataFrame, config: Optional[Config] = None, verbose: bool = True):
        self.raw = df
        self.config = config or Config()
        self.verbose = verbose
        self.clean: Optional[pd.DataFrame] = None
        self.issues: Optional[pd.DataFrame] = None
        self.results: Dict[str, ParcelResult] = {}
        self._log: List[Dict[str, object]] = []
        self._cache: Dict[str, pd.DataFrame] = {}

    def __repr__(self) -> str:
        if not self.results:
            state = "validated" if self.clean is not None else "not validated"
            return "AssociatorSession(%d rows, %s)" % (len(self.raw), state)
        return "AssociatorSession(%d rows, %d parcels analysed, %d duplicate groups, %d review pairs)" % (
            len(self.raw), len(self.results), sum(len(r.groups) for r in self.results.values()),
            sum(len(r.review_pairs) for r in self.results.values()))

    # ------------------------------------------------------------------ util
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _record(self, stage: str, t0: float, **detail) -> None:
        self._log.append({"STAGE": stage, "SECONDS": round(time.time() - t0, 3),
                          "DETAIL": ", ".join("%s=%s" % (k.lower(), v) for k, v in detail.items())})

    @property
    def log(self) -> pd.DataFrame:
        """Every stage run so far, with timings and counts."""
        return pd.DataFrame(self._log)

    def _need_clean(self):
        if self.clean is None:
            self.validate()

    def _need_results(self):
        if not self.results:
            raise RuntimeError("Run the engine first: session.run()")

    def _cached(self, name: str, builder) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = builder()
        return self._cache[name]

    def _results(self, parcel: Optional[str] = None) -> List[ParcelResult]:
        self._need_results()
        if parcel is None:
            return list(self.results.values())
        if parcel not in self.results:
            raise KeyError("Parcel %s was not analysed (fewer than %d addresses, or not in the input)"
                           % (parcel, self.config.min_addresses))
        return [self.results[parcel]]

    @staticmethod
    def _filter(df: pd.DataFrame, parcel: Optional[str]) -> pd.DataFrame:
        if parcel is None or df.empty:
            return df
        return df[df["CADASTRE_POLYGON_PID"] == parcel].reset_index(drop=True)

    # --------------------------------------------------------- 1. validate
    def validate(self) -> pd.DataFrame:
        """Check and normalise the input; returns the issues table."""
        t0 = time.time()
        self.clean, self.issues = prepare_input(self.raw)
        errors = int(self.issues.loc[self.issues.SEVERITY == "error", "ROWS"].sum()) if len(self.issues) else 0
        self._record("validate", t0, ROWS_IN=len(self.raw), ROWS_ACCEPTED=len(self.clean),
                     ISSUES=len(self.issues), ERROR_ROWS=errors)
        self._say("validate: %d of %d rows accepted, %d issue types (%d error rows)"
                  % (len(self.clean), len(self.raw), len(self.issues), errors))
        return self.issues

    # ---------------------------------------------------------- 2. profile
    def profile(self, top: Optional[int] = None) -> pd.DataFrame:
        """Per parcel: how many addresses, which flat types, levels, building names, bases."""
        self._need_clean()
        t0 = time.time()
        c = self.clean[self.clean["CADASTRE_POLYGON_PID"] != ""]
        sub = (c["FLAT_TYPE_CODE"] != "") | (c["FLAT_NUMBER"] != "") | (c["LEVEL_TYPE_CODE"] != "") \
            | (c["LEVEL_NUMBER"] != "")
        base = (c["NUMBER_FIRST"] + "-" + c["NUMBER_LAST"] + " " + c["STREET_NAME"] + " " + c["STREET_TYPE_CODE"])
        df = pd.DataFrame({
            "CADASTRE_POLYGON_PID": c["CADASTRE_POLYGON_PID"], "SUB": sub,
            "LEVEL": c["LEVEL_TYPE_CODE"] != "", "BNAME": c["BUILDING_NAME"] != "",
            "RANGE": c["NUMBER_LAST"] != "", "BASE": base, "FT": c["FLAT_TYPE_CODE"],
        })
        g = df.groupby("CADASTRE_POLYGON_PID")
        out = pd.DataFrame({
            "N_ADDRESSES": g.size(), "N_SUB_ADDRESSES": g["SUB"].sum(),
            "N_WITH_LEVEL": g["LEVEL"].sum(), "N_WITH_BUILDING_NAME": g["BNAME"].sum(),
            "N_RANGE_ADDRESSES": g["RANGE"].sum(), "N_DISTINCT_BASES": g["BASE"].nunique(),
            "FLAT_TYPES": g["FT"].agg(lambda s: ", ".join("%s:%d" % kv for kv in Counter(x for x in s if x).most_common(5))),
        }).reset_index()
        out["WILL_ANALYSE"] = out["N_ADDRESSES"] >= self.config.min_addresses
        out = out.sort_values("N_ADDRESSES", ascending=False).reset_index(drop=True)
        self._record("profile", t0, PARCELS=len(out), PARCELS_TO_ANALYSE=int(out.WILL_ANALYSE.sum()))
        self._say("profile: %d parcels, %d with >= %d addresses will be analysed"
                  % (len(out), int(out.WILL_ANALYSE.sum()), self.config.min_addresses))
        return out.head(top) if top else out

    # -------------------------------------------------------------- 3. run
    def run(self, parcels: Optional[Iterable[str]] = None, config: Optional[Config] = None,
            progress: bool = True) -> "AssociatorSession":
        """Analyse every parcel (or only ``parcels``). Re-running a subset replaces
        just those parcels' results."""
        self._need_clean()
        if config is not None:
            self.config = config
        t0 = time.time()
        cb = None
        if progress and self.verbose:
            def cb(i, n):                     # plain lines: render the same in any notebook
                if n >= 200 and (i == n or i % (n // 10) == 0):
                    print("  %d/%d parcels (%.0fs)" % (i, n, time.time() - t0))
        new = analyse(self.clean, self.config, parcels=parcels, progress=cb)
        if parcels is None:
            self.results = new
        else:
            self.results.update(new)
        self._cache.clear()
        groups = sum(len(r.groups) for r in new.values())
        self._record("run" if parcels is None else "rerun", t0, PARCELS=len(new), GROUPS=groups,
                     REVIEW_PAIRS=sum(len(r.review_pairs) for r in new.values()))
        self._say("run: %d parcels analysed in %.2fs, %d duplicate groups"
                  % (len(new), time.time() - t0, groups))
        return self

    # ------------------------------------------------------- 4. overview
    def overview(self):
        """Headline numbers (HTML in a notebook) and the parcel summary table."""
        self._need_results()
        stats = {"rows_in": len(self.raw), "rows_accepted": len(self.clean)}
        out = _display_html(overview_html(self.results.values(), stats))
        return out if out is not None else self.parcel_summary()

    def interesting_parcels(self, n: int = 5) -> List[str]:
        """Parcels with the most duplicate groups, then the most styles."""
        ps = self.parcel_summary()
        return list(ps.sort_values(["N_DUPLICATE_GROUPS", "N_STYLES", "N_ADDRESSES"],
                                   ascending=False)["CADASTRE_POLYGON_PID"].head(n))

    # ------------------------------------------------------ 5. one parcel
    def show_parcel(self, parcel: str):
        """Styles, how they map, stacking plans, parents, groups and review candidates."""
        return _display_html(parcel_html(self._results(parcel)[0], self.config))

    # ---------------------------------------------------------- 6. tables
    def parcel_summary(self) -> pd.DataFrame:
        return self._cached("parcel_summary", lambda: tables.parcel_summary(self._results()))

    def styles(self, parcel: Optional[str] = None) -> pd.DataFrame:
        return self._filter(self._cached("styles", lambda: tables.styles(self._results())), parcel)

    def blocks(self, parcel: Optional[str] = None) -> pd.DataFrame:
        return self._filter(self._cached("blocks", lambda: tables.blocks(self._results())), parcel)

    def mappings(self, parcel: Optional[str] = None, include_rejected: bool = False) -> pd.DataFrame:
        key = "mappings_all" if include_rejected else "mappings"
        return self._filter(self._cached(key, lambda: tables.mappings(
            self._results(), self.config, include_rejected)), parcel)

    def structures(self, parcel: Optional[str] = None) -> pd.DataFrame:
        return self._filter(self._cached("structures", lambda: tables.structures(self._results())), parcel)

    def addresses(self, parcel: Optional[str] = None) -> pd.DataFrame:
        """Every analysed address with style, block, structure, slot and duplicate group."""
        return self._filter(self._cached("addresses", lambda: tables.addresses(
            self._results(), self.config)), parcel)

    def duplicate_pairs(self, parcel: Optional[str] = None, include_review: bool = False) -> pd.DataFrame:
        key = "pairs_all" if include_review else "pairs"
        return self._filter(self._cached(key, lambda: tables.duplicate_pairs(
            self._results(), self.config, include_review)), parcel)

    def duplicate_groups(self, parcel: Optional[str] = None) -> pd.DataFrame:
        return self._filter(self._cached("groups", lambda: tables.duplicate_groups(
            self._results(), self.config)), parcel)

    def review_candidates(self, parcel: Optional[str] = None) -> pd.DataFrame:
        """Pairs that look like duplicates but cannot be confirmed (never grouped)."""
        return self._filter(self._cached("review", lambda: tables.duplicate_pairs(
            self._results(), self.config, review_only=True)), parcel)

    def to_frames(self) -> Dict[str, pd.DataFrame]:
        """All outputs as DataFrames, ready to write wherever you like."""
        return {
            "issues": self.issues, "parcel_summary": self.parcel_summary(), "styles": self.styles(),
            "blocks": self.blocks(), "mappings": self.mappings(include_rejected=True),
            "structures": self.structures(), "addresses": self.addresses(),
            "duplicate_pairs": self.duplicate_pairs(), "duplicate_groups": self.duplicate_groups(),
            "review_candidates": self.review_candidates(),
        }

    # -------------------------------------------------------- 7. explain
    def explain(self, addr_id: str) -> Dict[str, object]:
        """Trace one address through every stage."""
        self._need_results()
        addr_id = str(addr_id).strip()
        for r in self.results.values():
            f = next((x for x in r.feats if x.pid == addr_id), None)
            if f is None:
                continue
            block = next((b for b in r.blocks if b.id == f.block_id), None)
            style = next((s for s in r.styles if s.id == f.style_id), None)
            group = next((g for g in r.groups if any(m.pid == addr_id for m in g.members)), None)
            kept = {l.id for l in r.links}
            links = [{"LINK": l.id, "TRANSFORM": l.t.describe(), "P": round(l.p, 3),
                      "STATUS": tables.link_status(l, kept), "EVIDENCE": "; ".join(l.evidence)}
                     for l in sorted(r.scored_links, key=lambda l: -l.logit)
                     if block is not None and block in (l.x, l.y)]
            pairs = [{"OTHER": (p.b if p.a.pid == addr_id else p.a).pid,
                      "OTHER_LABEL": (p.b if p.a.pid == addr_id else p.a).label,
                      "RELATION": p.relation, "P": round(p.p, 3), "METHOD": p.method}
                     for p in r.pairs if addr_id in (p.a.pid, p.b.pid)]
            out = {
                "ADDR_ID": addr_id, "SOURCE_ID": f.a.source_id, "PARCEL": r.parcel,
                "LABEL": f.label, "KIND": "sub-address" if f.is_sub else "parent",
                "STYLE": "%s  %s" % (style.id, style.template) if style else "",
                "BLOCK": "%s  %s" % (block.id, block.context_label()) if block else "",
                "NUMBERING_MODEL": block.describe_model() if block else "",
                "KEYS": {k: v for k, v in f.keys.items()},
                "STRUCTURE": block.structure_id if block else ",".join(
                    next((p.structures for p in r.parents if p.feat.pid == addr_id), ())),
                "PARENT_ROLE": next((p.role for p in r.parents if p.feat.pid == addr_id), ""),
                "GROUP": ("%s (%s, preferred %s)" % (group.id, self.config.band(group.p_min),
                                                     group.preferred.pid)) if group else "not grouped",
                "PAIRS": pd.DataFrame(pairs), "BLOCK_LINKS": pd.DataFrame(links),
            }
            if self.verbose:
                for k, v in out.items():
                    if not isinstance(v, pd.DataFrame):
                        print("%-16s %s" % (k, v))
            return out
        raise KeyError("ADDR_ID %s is not in any analysed parcel" % addr_id)

    def schematic(self, parcel: str) -> str:
        """Plain-text schematic of one parcel (styles, mappings, structures, groups)."""
        from .render import parcel_text
        return parcel_text(self._results(parcel)[0], self.config)

    # ------------------------------------------------------- evaluation
    def evaluate(self, truth: pd.DataFrame) -> Dict[str, object]:
        """Score against a truth table [ADDR_ID, ENTITY_ID, (SCENARIO)] (synthetic data)."""
        self._need_results()
        ev = _evaluate(self.results.values(), truth)
        o, rv = ev["overall"], ev["review"]
        self._say("precision %.4f  recall %.4f  (TP %d, FP %d, FN %d); review candidates %d (%d true); "
                  "recall if reviewed %.4f" % (o["PRECISION"], o["RECALL"], o["TP"], o["FP"], o["FN"],
                                               rv["PAIRS"], rv["TRUE"], rv["RECALL_IF_REVIEWED"]))
        return ev
