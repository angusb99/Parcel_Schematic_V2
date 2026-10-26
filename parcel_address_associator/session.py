"""Notebook session handler: run the engine stage by stage and inspect every step.

    from parcel_address_associator import AssociatorSession
    s = AssociatorSession(df)        # df: one row per address, INPUT_COLUMNS
    s.validate()                     # 1. schema and data checks -> issues table
    s.profile()                      # 2. what each parcel holds
    s.run()                          # 3. analyse every parcel
    s.overview()                     # 4. headline numbers
    s.save_html("report.html")       # 5. HTML report: styles, mappings, stacking plans per parcel
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
from .render import parcel_html


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
        self.use_flags: Optional[Dict[str, Optional[bool]]] = None   # second dataset, by ADDR_ID
        self.use_issues: Optional[pd.DataFrame] = None
        self.use: Dict[str, pd.DataFrame] = {}                       # reconciliation outputs

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
        if self.use:
            self.use = {}
            self._say("  (use reconciliation cleared: run reconcile_use() again)")
        groups = sum(len(r.groups) for r in new.values())
        self._record("run" if parcels is None else "rerun", t0, PARCELS=len(new), GROUPS=groups,
                     REVIEW_PAIRS=sum(len(r.review_pairs) for r in new.values()))
        self._say("run: %d parcels analysed in %.2fs, %d duplicate groups"
                  % (len(new), time.time() - t0, groups))
        return self

    # ------------------------------------------------------- 4. overview
    def overview(self) -> pd.DataFrame:
        """Headline numbers as a small table."""
        self._need_results()
        res = list(self.results.values())
        rows = [
            ("input rows", len(self.raw)),
            ("rows accepted", len(self.clean)),
            ("parcels analysed", len(res)),
            ("addresses analysed", sum(len(r.feats) for r in res)),
            ("duplicate groups", sum(len(r.groups) for r in res)),
            ("addresses in duplicate groups", sum(len(g.members) for r in res for g in r.groups)),
            ("redundant addresses (group size - 1)", sum(len(g.members) - 1 for r in res for g in r.groups)),
            ("review candidate pairs", sum(len(r.review_pairs) for r in res)),
        ]
        return pd.DataFrame(rows, columns=["METRIC", "VALUE"])

    # ----------------------------------------------------- HTML reports
    def save_html(self, path: str = "parcel_address_report.html", parcels: Optional[Iterable[str]] = None,
                  max_parcels: int = 50, title: str = "Parcel Address Associator") -> str:
        """Write a standalone HTML report and return its absolute path. Nothing is
        opened or displayed.

        The report has the headline numbers, an index of every analysed parcel, and
        a full section (styles, mappings, stacking plans, parents, groups, review
        candidates) for each parcel in ``parcels``, or by default for the
        ``max_parcels`` parcels with the most duplicate groups.
        """
        import os

        from .render import report_html
        self._need_results()
        t0 = time.time()
        if parcels is None:
            detailed = self.interesting_parcels(max_parcels)
        else:
            detailed = [p for p in parcels]
            missing = [p for p in detailed if p not in self.results]
            if missing:
                raise KeyError("Not analysed: %s" % ", ".join(missing[:5]))
        stats = {"rows_in": len(self.raw), "rows_accepted": len(self.clean)}
        html_doc = report_html(self.results.values(), self.config, stats, detailed, title=title,
                               use=self.use or None)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        self._record("save_html", t0, file=os.path.basename(path), parcels_detailed=len(detailed),
                     kb=round(len(html_doc.encode("utf-8")) / 1024))
        try:
            shown = os.path.relpath(path)
        except ValueError:                    # other drive on Windows
            shown = path
        self._say("saved %s (%d parcels in detail, %.0f KB)"
                  % (shown, len(detailed), len(html_doc.encode("utf-8")) / 1024))
        return path

    def interesting_parcels(self, n: int = 5) -> List[str]:
        """Parcels with the most duplicate groups, then the most styles."""
        ps = self.parcel_summary()
        return list(ps.sort_values(["N_DUPLICATE_GROUPS", "N_STYLES", "N_ADDRESSES"],
                                   ascending=False)["CADASTRE_POLYGON_PID"].head(n))

    # ------------------------------------------------------ 5. one parcel
    def show_parcel(self, parcel: str):
        """Optional inline view of one parcel (rendered in the notebook cell; returns the
        HTML string outside a notebook). Use ``save_html`` to write files instead."""
        use_rows, use_blocks = None, None
        if self.use:
            t = self.use["addresses"]
            use_rows = {row["ADDR_ID"]: row for row in t[t.CADASTRE_POLYGON_PID == parcel].to_dict("records")}
            ub = self.use["blocks"]
            use_blocks = ub[ub.CADASTRE_POLYGON_PID == parcel] if len(ub) else None
        return _display_html(parcel_html(self._results(parcel)[0], self.config, use_rows=use_rows,
                                         use_blocks=use_blocks))

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
        frames = {
            "issues": self.issues, "parcel_summary": self.parcel_summary(), "styles": self.styles(),
            "blocks": self.blocks(), "mappings": self.mappings(include_rejected=True),
            "structures": self.structures(), "addresses": self.addresses(),
            "duplicate_pairs": self.duplicate_pairs(), "duplicate_groups": self.duplicate_groups(),
            "review_candidates": self.review_candidates(),
        }
        if self.use:
            frames.update({"use_issues": self.use_issues, "use_reconciliation": self.use["addresses"],
                           "use_areas": self.use["blocks"], "use_by_parcel": self.use["parcels"],
                           "source_reliability": self.use["reliability"]})
        return frames

    # --------------------------------------------- 8. residential use
    def attach_use(self, use_df: pd.DataFrame, id_col: str = "ADDR_ID",
                   flag_col: str = "RESIDENTIAL_DWELLING") -> pd.DataFrame:
        """Attach the second use dataset (one row per ADDR_ID with a yes/no flag).
        Returns its validation issues."""
        from .usage import prepare_use
        self._need_clean()
        t0 = time.time()
        self.use_flags, self.use_issues = prepare_use(use_df, id_col, flag_col,
                                                      known_ids=set(self.clean["ADDR_ID"]))
        self.use = {}
        self._record("attach_use", t0, rows=len(use_df), flags=len(self.use_flags),
                     residential=sum(1 for v in self.use_flags.values() if v))
        self._say("attach_use: %d flags (%d residential), %d issue types"
                  % (len(self.use_flags), sum(1 for v in self.use_flags.values() if v), len(self.use_issues)))
        return self.use_issues

    def reconcile_use(self) -> "AssociatorSession":
        """Reconcile the base ADDRESS_TYPE with the attached dataset, per premise,
        using the parcel structure. Covers every accepted ADDR_ID (parcels too small
        to analyse are reconciled from flat type and flags alone)."""
        from .model import Address
        from .usage import (block_use_table, build_premises, decide, find_area_patterns, fit,
                            parcel_use_summary, review_partners)
        self._need_results()
        if self.use_flags is None:
            raise RuntimeError("Attach the second dataset first: session.attach_use(use_df)")
        t0 = time.time()
        cfg = self.config
        base_type = dict(zip(self.clean["ADDR_ID"], self.clean["ADDRESS_TYPE"]))
        has_type = any(base_type.values())
        base = {k: ((v in cfg.residential_address_types) if has_type else None) for k, v in base_type.items()}
        analysed = {f.pid for r in self.results.values() for f in r.feats}
        extra = [Address.from_row(r) for r in
                 self.clean[~self.clean["ADDR_ID"].isin(analysed)].to_dict("records")]
        premises, blocks = build_premises(self.results.values(), extra, base, self.use_flags,
                                          review_partners(self.results.values()))
        find_area_patterns(premises, blocks, cfg)
        reliability = fit(premises, cfg)
        table = decide(premises, base_type, base, self.use_flags, cfg)
        self.use = {"addresses": table, "blocks": block_use_table(premises, blocks),
                    "parcels": parcel_use_summary(table), "reliability": reliability}
        cats = table["CATEGORY"].value_counts().to_dict()
        self._record("reconcile_use", t0, addresses=len(table), premises=len(premises),
                     **{c.lower(): n for c, n in cats.items() if c != "OK"})
        self._say("reconcile_use: %d addresses in %d premises; %d base-only, %d other-only; "
                  "%d suggested changes to base, %d to other"
                  % (len(table), len(premises), int((table.DIRECTION == "BASE_ONLY").sum()),
                     int((table.DIRECTION == "OTHER_ONLY").sum()), int(table.CHANGE_BASE.sum()),
                     int(table.CHANGE_OTHER.sum())))
        return self

    def _need_use(self):
        if not self.use:
            raise RuntimeError("Run session.reconcile_use() first")

    def use_reconciliation(self, parcel: Optional[str] = None, direction: Optional[str] = None,
                           category: Optional[str] = None, changes_only: bool = False) -> pd.DataFrame:
        """Per ADDR_ID: both flags, reconciled use, category, suggested flags, evidence.

        direction: 'BASE_ONLY' (base residential, other not), 'OTHER_ONLY', 'BOTH', 'NEITHER'.
        """
        self._need_use()
        df = self._filter(self.use["addresses"], parcel)
        if direction:
            df = df[df["DIRECTION"] == direction.upper()]
        if category:
            df = df[df["CATEGORY"] == category.upper()]
        if changes_only:
            df = df[df["CHANGE_BASE"] | df["CHANGE_OTHER"]]
        return df.reset_index(drop=True)

    def use_areas(self, parcel: Optional[str] = None, problems_only: bool = True) -> pd.DataFrame:
        """Per block (one scheme in one building): source agreement and area verdict.
        By default only coverage gaps and whole-block disagreements."""
        self._need_use()
        df = self._filter(self.use["blocks"], parcel)
        if problems_only and not df.empty:
            df = df[df["VERDICT"].isin(["OTHER_COVERAGE_GAP", "BASE_COVERAGE_GAP", "BLOCK_DISAGREEMENT"])]
        return df.reset_index(drop=True)

    def use_summary(self) -> pd.DataFrame:
        """Per parcel: residential counts (base, other, reconciled dwellings) and categories."""
        self._need_use()
        return self.use["parcels"]

    def source_reliability(self) -> pd.DataFrame:
        """Fitted sensitivity/specificity of each source per context: how far each can be trusted."""
        self._need_use()
        return self.use["reliability"]

    def use_category_summary(self) -> pd.DataFrame:
        """Categories x direction, with what each category means."""
        from .usage import CATEGORY_NOTES
        self._need_use()
        t = self.use["addresses"]
        out = pd.crosstab(t["CATEGORY"], t["DIRECTION"]).reset_index()
        out.insert(1, "MEANING", out["CATEGORY"].map(CATEGORY_NOTES))
        return out

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
            if self.use:
                u = self.use["addresses"]
                u = u[u.ADDR_ID == addr_id]
                if len(u):
                    x = u.iloc[0]
                    out["USE"] = ("%s (P=%.2f) base=%s other=%s -> %s %s" % (
                        x.RECONCILED_USE, x.P_DWELLING, x.BASE_RESIDENTIAL, x.OTHER_RESIDENTIAL, x.CATEGORY,
                        x.SUGGESTED_ACTION)).strip()
                    out["USE_EVIDENCE"] = x.EVIDENCE
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
    def evaluate_use(self, truth: pd.DataFrame) -> Dict[str, object]:
        """Score the use reconciliation against truth [ADDR_ID, ENTITY_ID, IS_DWELLING, ...]."""
        from .evaluate import evaluate_use
        self._need_use()
        ev = evaluate_use(self.use["addresses"], truth)
        self._say("use: decided accuracy %s, uncertain %.1f%%, %d true dwellings"
                  % (ev["USE_ACCURACY_DECIDED"], 100 * ev["UNCERTAIN_SHARE"], ev["TRUE_DWELLINGS"]))
        return ev

    def evaluate(self, truth: pd.DataFrame) -> Dict[str, object]:
        """Score against a truth table [ADDR_ID, ENTITY_ID, (SCENARIO)] (synthetic data)."""
        self._need_results()
        ev = _evaluate(self.results.values(), truth)
        o, rv = ev["overall"], ev["review"]
        self._say("precision %.4f  recall %.4f  (TP %d, FP %d, FN %d); review candidates %d (%d true); "
                  "recall if reviewed %.4f" % (o["PRECISION"], o["RECALL"], o["TP"], o["FP"], o["FN"],
                                               rv["PAIRS"], rv["TRUE"], rv["RECALL_IF_REVIEWED"]))
        return ev
