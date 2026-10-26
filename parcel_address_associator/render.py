"""HTML views for notebooks: parcel schematics with stacking plans.

Plain HTML + inline SVG (hover tooltips via SVG <title>, no JavaScript), so it
renders in Jupyter, JupyterLab, VS Code and Databricks ``displayHTML``.
"""
from __future__ import annotations

import html
from typing import Dict, Iterable, List, Optional

from . import codes
from .config import Config
from .reconcile import ParcelResult

CSS = """
<style>
.paa{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--ring:rgba(11,11,11,.12);
--empty:#f0efec;--s0:#898781;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;
--s7:#6250d6;--s8:#e34948;color:var(--ink);font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.paa{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--ring:rgba(255,255,255,.12);--empty:#2a2a28;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s7:#9085e9;--s8:#e66767}}
body[data-jp-theme-light="false"] .paa,.vscode-dark .paa{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--ring:rgba(255,255,255,.12);--empty:#2a2a28;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s7:#9085e9;--s8:#e66767}
.paa h3{font-size:15px;margin:14px 0 4px}.paa h4{font-size:12px;margin:12px 0 6px;color:var(--ink2);
text-transform:uppercase;letter-spacing:.04em}.paa .sub{color:var(--ink2)}.paa .muted{color:var(--muted)}
.paa table{border-collapse:collapse;font-size:12px;margin:0}.paa th,.paa td{text-align:left!important;
padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top}.paa th{color:var(--ink2);font-weight:600}
.paa td.n{text-align:right!important;font-variant-numeric:tabular-nums}.paa .mono{font-family:ui-monospace,Consolas,monospace}
.paa .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.paa .chip{display:inline-block;padding:0 7px;border:1px solid var(--ring);border-radius:999px;color:var(--ink2);
font-size:11px;margin-right:4px}.paa .ev{color:var(--ink2);font-size:11px}
.paa .plans{display:flex;flex-wrap:wrap;gap:18px}.paa svg text{fill:var(--muted);font-size:10px}
.paa svg .cnt{fill:var(--ink);font-weight:600}.paa .tiles{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
.paa .tile{border:1px solid var(--ring);border-radius:8px;padding:6px 12px;min-width:120px}
.paa .tile b{display:block;font-size:20px;font-weight:600}.paa .tile span{color:var(--ink2);font-size:11px}
</style>"""


def _e(s) -> str:
    return html.escape(str("" if s is None else s), quote=True)


def _lvl(v) -> str:
    return codes.level_label(v) if v is not None else ""


def _n(s) -> tuple:
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return (int(d) if d else -1, str(s))


def style_slots(r: ParcelResult) -> Dict[str, int]:
    """Fixed colour slot per style (largest first); beyond 8 -> grey 'other'."""
    return {s.id: (i + 1 if i < 8 else 0) for i, s in enumerate(r.styles)}


def stacking_svg(r: ParcelResult, st, slots: Dict[str, int], max_units: int = 40) -> str:
    grid = r.cells.get(st.id, {})
    if not grid:
        return '<p class="muted">No level information to draw a stacking plan.</p>'
    levels = sorted({k[0] for k in grid}, reverse=True)
    units = sorted({k[1] for k in grid}, key=_n)[:max_units]
    here = sorted({f.style_id for fs in grid.values() for f in fs if f.style_id}, key=lambda s: int(s[1:]))
    nslots = max(1, len(here))
    cw, ch, gap, lm, tm, badge = 36, 20, 2, 34, 18, 13
    w = lm + len(units) * (cw + gap) + 4
    h = tm + len(levels) * (ch + gap) + 4
    out = ['<svg role="img" aria-label="stacking plan %s" width="%d" height="%d" viewBox="0 0 %d %d">'
           % (_e(st.id), w, h, w, h)]
    for j, u in enumerate(units):
        out.append('<text x="%d" y="12" text-anchor="middle">%s</text>' % (lm + j * (cw + gap) + cw // 2, _e(u)))
    for i, lv in enumerate(levels):
        y = tm + i * (ch + gap)
        out.append('<text x="%d" y="%d" text-anchor="end">%s</text>' % (lm - 6, y + 14, _e(_lvl(lv))))
        for j, u in enumerate(units):
            x = lm + j * (cw + gap)
            fs = grid.get((lv, u))
            out.append('<g><rect x="%d" y="%d" width="%d" height="%d" rx="3" fill="var(--empty)"/>'
                       % (x, y, cw, ch))
            if fs:
                present = {f.style_id for f in fs}
                sw = (cw - badge - 2) / nslots
                for k, sid in enumerate(here):
                    if sid in present:
                        out.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="var(--s%d)"/>'
                                   % (x + 1 + k * sw, y + 2, max(sw - 1, 1), ch - 4, slots.get(sid, 0)))
                if len(fs) > 1:
                    out.append('<text class="cnt" x="%.1f" y="%d" text-anchor="middle">%d</text>'
                               % (x + cw - badge / 2.0, y + 14, len(fs)))
                tip = "%s %s unit %s\n%s" % (st.id, _lvl(lv), u, "\n".join(
                    "%s  %s  [%s]" % (f.style_id, f.label, f.pid) for f in fs))
                out.append("<title>%s</title>" % _e(tip))
            out.append("</g>")
    out.append("</svg>")
    return "".join(out)


USE_FILL = {"RESIDENTIAL": "var(--s1)", "NOT_RESIDENTIAL": "var(--muted)", "UNCERTAIN": "#fab219"}


def use_plan_svg(r: ParcelResult, st, use_rows: Dict[str, dict], max_units: int = 40) -> str:
    """Stacking plan coloured by reconciled use; a red outline marks slots where a
    change to either source is suggested."""
    grid = r.cells.get(st.id, {})
    if not grid:
        return '<p class="muted">No level information to draw a use plan.</p>'
    levels = sorted({k[0] for k in grid}, reverse=True)
    units = sorted({k[1] for k in grid}, key=_n)[:max_units]
    cw, ch, gap, lm, tm = 24, 18, 2, 34, 18
    w = lm + len(units) * (cw + gap) + 4
    h = tm + len(levels) * (ch + gap) + 4
    out = ['<svg role="img" aria-label="use plan %s" width="%d" height="%d" viewBox="0 0 %d %d">'
           % (_e(st.id), w, h, w, h)]
    for j, u in enumerate(units):
        out.append('<text x="%d" y="12" text-anchor="middle">%s</text>' % (lm + j * (cw + gap) + cw // 2, _e(u)))
    for i, lv in enumerate(levels):
        y = tm + i * (ch + gap)
        out.append('<text x="%d" y="%d" text-anchor="end">%s</text>' % (lm - 6, y + 13, _e(_lvl(lv))))
        for j, u in enumerate(units):
            x = lm + j * (cw + gap)
            fs = grid.get((lv, u))
            rows = [use_rows[f.pid] for f in fs or [] if f.pid in use_rows]
            if not rows:
                out.append('<rect x="%d" y="%d" width="%d" height="%d" rx="3" fill="var(--empty)"/>' % (x, y, cw, ch))
                continue
            use = rows[0]["RECONCILED_USE"]
            # outline use corrections only; de-counting duplicates is listed, not drawn
            change = any((rw["CHANGE_BASE"] or rw["CHANGE_OTHER"])
                         and rw["CATEGORY"] != "DUPLICATE_OF_COUNTED_DWELLING" for rw in rows)
            review = any(rw["CATEGORY"] in ("UNRESOLVED", "POSSIBLE_DUPLICATE") for rw in rows)
            stroke = ' stroke="var(--s8)" stroke-width="2"' if change else (
                ' stroke="#fab219" stroke-width="2" stroke-dasharray="3 2"' if review else "")
            tip = "%s %s unit %s: %s (P=%.2f)\n%s" % (st.id, _lvl(lv), u, use, rows[0]["P_DWELLING"], "\n".join(
                "%s  base=%s other=%s  %s %s" % (rw["LABEL"], _yn(rw["BASE_RESIDENTIAL"]),
                                                  _yn(rw["OTHER_RESIDENTIAL"]), rw["CATEGORY"],
                                                  rw["SUGGESTED_ACTION"]) for rw in rows))
            out.append('<g><rect x="%d" y="%d" width="%d" height="%d" rx="3" fill="%s"%s/><title>%s</title></g>'
                       % (x + 1, y + 1, cw - 2, ch - 2, USE_FILL.get(use, "var(--empty)"), stroke, _e(tip)))
    out.append("</svg>")
    return "".join(out)


def _yn(v) -> str:
    return "?" if v is None or (isinstance(v, float) and v != v) else ("Y" if v else "N")


def use_section_html(r: ParcelResult, use_rows: Dict[str, dict], blocks_df, max_rows: int = 200) -> str:
    rows = [use_rows[f.pid] for f in r.feats if f.pid in use_rows]
    if not rows:
        return ""
    base_n = sum(1 for x in rows if x["BASE_RESIDENTIAL"] == True)    # noqa: E712 (None-safe)
    other_n = sum(1 for x in rows if x["OTHER_RESIDENTIAL"] == True)  # noqa: E712
    rec_n = sum(1 for x in rows if x["COUNTED_DWELLING_ADDR_ID"] == x["ADDR_ID"])
    h = ["<h4>7. Residential use</h4><div class='tiles'>"]
    for v, t in ((base_n, "base residential"), (other_n, "other residential"), (rec_n, "reconciled dwellings"),
                 (sum(1 for x in rows if x["CHANGE_BASE"]), "base changes suggested"),
                 (sum(1 for x in rows if x["CHANGE_OTHER"]), "other changes suggested"),
                 (sum(1 for x in rows if x["CATEGORY"] in ("UNRESOLVED", "POSSIBLE_DUPLICATE")), "to review")):
        h.append("<div class='tile'><b>%s</b><span>%s</span></div>" % (format(v, ","), _e(t)))
    h.append("</div>")
    if blocks_df is not None and len(blocks_df):
        h.append("<table><tr><th>Block</th><th>Style</th><th>Premises</th><th>Both</th><th>Base only</th>"
                 "<th>Other only</th><th>Neither</th><th>Verdict</th></tr>")
        for b in blocks_df.itertuples(index=False):
            h.append("<tr><td class='mono'>%s</td><td class='mono'>%s</td><td class='n'>%d</td><td class='n'>%d</td>"
                     "<td class='n'>%d</td><td class='n'>%d</td><td class='n'>%d</td><td>%s</td></tr>"
                     % (_e(b.BLOCK_ID), _e(b.STYLE_TEMPLATE), b.N_PREMISES, b.BOTH_RESIDENTIAL, b.BASE_ONLY,
                        b.OTHER_ONLY, b.NEITHER, _e(b.VERDICT)))
        h.append("</table>")
    if r.structures:
        h.append("<div class='sub' style='margin-top:6px'><span class='sw' style='background:var(--s1)'></span>"
                 "residential <span class='sw' style='background:var(--muted)'></span>not residential "
                 "<span class='sw' style='background:#fab219'></span>uncertain · red outline = use correction "
                 "suggested · dashed = review · duplicate de-counting is in the table below · hover for flags"
                 "</div><div class='plans'>")
        for st in r.structures:
            h.append("<div><div><b>%s</b> %s</div>%s</div>" % (_e(st.id), _e(st.label), use_plan_svg(r, st, use_rows)))
        h.append("</div>")
    issues = [x for x in rows if x["CATEGORY"] != "OK"]
    if issues:
        h.append("<table><tr><th>Address</th><th>Base</th><th>Other</th><th>Reconciled</th><th>Category</th>"
                 "<th>Suggested</th></tr>")
        order = {"BUILDING_OR_SITE_ADDRESS": 0, "MISSED_BY_OTHER": 1, "MISSED_BY_BASE": 2,
                 "LIKELY_NOT_RESIDENTIAL": 3, "MISSED_BY_BOTH": 4, "UNRESOLVED": 5, "POSSIBLE_DUPLICATE": 6,
                 "DUPLICATE_OF_COUNTED_DWELLING": 7}
        for x in sorted(issues, key=lambda x: (order.get(x["CATEGORY"], 9), x["LABEL"]))[:max_rows]:
            h.append("<tr><td>%s <span class='muted mono'>%s</span></td><td>%s</td><td>%s</td><td>%s %.2f</td>"
                     "<td>%s</td><td class='ev'>%s</td></tr>"
                     % (_e(x["LABEL"]), _e(x["ADDR_ID"]), _yn(x["BASE_RESIDENTIAL"]), _yn(x["OTHER_RESIDENTIAL"]),
                        _e(x["RECONCILED_USE"]), x["P_DWELLING"], _e(x["CATEGORY"]), _e(x["SUGGESTED_ACTION"])))
        h.append("</table>")
    return "".join(h)


def parcel_html(r: ParcelResult, cfg: Config, max_rows: int = 200, include_css: bool = True,
                use_rows: Optional[Dict[str, dict]] = None, use_blocks=None) -> str:
    slots = style_slots(r)
    n_rev = len(r.review_pairs)
    h = [CSS if include_css else "",
         '<div class="paa"><h3>Parcel <span class="mono">%s</span></h3>' % _e(r.parcel)]
    h.append('<div class="sub">%d addresses: %d sub-addresses in %d styles and %d blocks, %d parents. '
             '<b>%d duplicate groups</b>, %d review candidates.</div>'
             % (len(r.feats), r.n_sub, len(r.styles), len(r.blocks), len(r.parents), len(r.groups), n_rev))
    if r.styles:
        h.append("<h4>1. Address styles (schematics)</h4><table><tr><th>Style</th><th>Template</th>"
                 "<th>N</th><th>Blocks: context · numbering model</th><th>Examples</th></tr>")
        for s in r.styles:
            bl = "<br>".join("<b>%s</b> %s · <span class='ev'>%s</span> (%d)%s" % (
                _e(b.id), _e(b.context_label()), _e(b.describe_model()), b.size,
                " · <i>spans buildings</i>" if b.spanning else "") for b in s.blocks)
            h.append("<tr><td style='white-space:nowrap'><span class='sw' style='background:var(--s%d)'></span>%s"
                     "</td><td class='mono'>%s</td><td class='n'>%d</td><td>%s</td><td class='ev'>%s</td></tr>"
                     % (slots[s.id], _e(s.id), _e(s.template), s.size, bl,
                        "<br>".join(_e(f.label) for f in s.feats[:3])))
        h.append("</table>")
    kept = {l.id for l in r.links}
    shown = [l for l in sorted(r.scored_links, key=lambda l: -l.logit)
             if l.id in kept or l.review_logit is not None or l.p >= 0.2]
    if shown:
        h.append("<h4>2. How the styles map to each other</h4><table><tr><th>Blocks</th><th>Transform</th>"
                 "<th>Matched</th><th>P</th><th>Status</th><th>Evidence (log-odds)</th></tr>")
        for l in shown:
            status = ("AMBIGUOUS" if l.ambiguous else "KEPT") if l.id in kept else (
                "REVIEW" if l.review_logit is not None else "rejected")
            h.append("<tr><td class='mono'>%s ↔ %s</td><td>%s</td><td class='n'>%d of %d/%d</td>"
                     "<td class='n'>%.2f</td><td>%s</td><td class='ev'>%s</td></tr>"
                     % (_e(l.x.id), _e(l.y.id), _e(l.t.describe()), l.t.m, l.t.nx, l.t.ny, l.p,
                        _e(status), _e("; ".join(l.evidence))))
        h.append("</table>")
    if r.structures:
        legend = " ".join("<span><span class='sw' style='background:var(--s%d)'></span>%s</span>"
                          % (slots[s.id], _e(s.id)) for s in r.styles)
        h.append("<h4>3. Inferred structures · stacking plans</h4><div class='sub'>%s "
                 "<span class='muted'>· stripe = style present in the slot · number = addresses in the "
                 "slot · hover a cell for its addresses</span></div><div class='plans'>" % legend)
        for st in r.structures:
            un = r.unplaced.get(st.id, [])
            h.append("<div><div><b>%s</b> %s <span class='muted'>(%s)</span></div>%s%s</div>" % (
                _e(st.id), _e(st.label), _e(", ".join(b.id for b in st.blocks)), stacking_svg(r, st, slots),
                ("<div class='ev'>%d without level information: %s</div>" % (
                    len(un), _e(", ".join(f.label for f in un[:10])) + (" …" if len(un) > 10 else "")))
                if un else ""))
        h.append("</div>")
    if r.parents:
        h.append("<h4>4. Parent addresses</h4><table><tr><th>Address</th><th>Role</th><th>Structures</th></tr>")
        for p in r.parents:
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (_e(p.feat.label), _e(p.role), _e(", ".join(p.structures))))
        h.append("</table>")
    if r.groups:
        h.append("<h4>5. Duplicate groups (★ = suggested record to keep; first %d)</h4><table><tr><th>Group</th>"
                 "<th>Band</th><th>P min</th><th>Relation</th><th>Members</th></tr>" % min(max_rows, len(r.groups)))
        for g in sorted(r.groups, key=lambda g: g.p_min)[:max_rows]:
            mem = "<br>".join("%s%s <span class='muted mono'>%s · %s</span>" % (
                "★ " if f is g.preferred else "", _e(f.label), _e(f.style_id or "parent"), _e(f.pid))
                for f in g.members)
            h.append("<tr><td class='mono'>%s</td><td>%s</td><td class='n'>%.3f</td><td>%s</td><td>%s</td></tr>"
                     % (_e(g.id), cfg.band(g.p_min), g.p_min, _e(g.relation), mem))
        h.append("</table>")
    if n_rev:
        h.append("<h4>6. Review candidates (not grouped; first %d)</h4><table><tr><th>A</th><th>B</th>"
                 "<th>Relation</th><th>Why</th></tr>" % min(max_rows, n_rev))
        for p in sorted(r.review_pairs, key=lambda p: (p.relation, p.a.label))[:max_rows]:
            why = ("fits more than one building equally well" if p.relation == "AMBIGUOUS_BUILDING"
                   else "order-based renumbering; needs another source (e.g. strata lot ids) to confirm")
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td class='ev'>%s</td></tr>"
                     % (_e(p.a.label), _e(p.b.label), _e(p.relation), _e(why)))
        h.append("</table>")
    if use_rows:
        h.append(use_section_html(r, use_rows, use_blocks, max_rows))
    h.append("</div>")
    return "".join(h)


def parcel_text(r: ParcelResult, cfg: Config, max_groups: int = 10) -> str:
    out = ["PARCEL %s  %d addresses (%d sub-addresses, %d parents) | %d duplicate groups | %d review"
           % (r.parcel, len(r.feats), r.n_sub, len(r.parents), len(r.groups), len(r.review_pairs))]
    if r.styles:
        out.append("  STYLES")
        for s in r.styles:
            out.append("    %-4s %-56s %4d  %s" % (s.id, s.template, s.size, s.signature))
            for b in s.blocks:
                out.append("         %-7s %-38s n=%-4d %s%s" % (b.id, b.context_label()[:38], b.size,
                                                             b.describe_model(),
                                                             "  [spans buildings]" if b.spanning else ""))
    if r.links:
        out.append("  HOW STYLES MAP")
        for l in sorted(r.links, key=lambda l: -l.logit):
            out.append("    %-7s <-> %-7s %-50s %3d keys  p=%.3f%s" % (
                l.x.id, l.y.id, l.t.describe()[:50], l.t.m, l.p, "  AMBIGUOUS" if l.ambiguous else ""))
    if r.structures:
        out.append("  STRUCTURES")
        for st in r.structures:
            grid = r.cells.get(st.id, {})
            lv = sorted({k[0] for k in grid})
            out.append("    %-4s %-36s blocks %-26s %s, %d slots (%d with duplicates)" % (
                st.id, st.label[:36], ",".join(b.id for b in st.blocks),
                ("levels %s-%s" % (_lvl(lv[0]), _lvl(lv[-1]))) if lv else "no levels", len(grid),
                sum(1 for fs in grid.values() if len(fs) > 1)))
    if r.parents:
        out.append("  PARENTS")
        for p in r.parents:
            out.append("    %-44s %-10s %s" % (p.feat.label[:44], p.role, ",".join(p.structures)))
    if r.groups:
        out.append("  DUPLICATE GROUPS (weakest %d of %d)" % (min(max_groups, len(r.groups)), len(r.groups)))
        for g in sorted(r.groups, key=lambda g: g.p_min)[:max_groups]:
            out.append("    %-5s %-6s %.3f %-15s %s" % (g.id, cfg.band(g.p_min), g.p_min, g.relation,
                                                      "  ==  ".join(("*" if f is g.preferred else "") + f.label
                                                                    for f in g.members)))
    return "\n".join(out)


def overview_html(results: Iterable[ParcelResult], stats: Dict[str, object]) -> str:
    res = list(results)
    tiles = [
        (stats.get("rows_in", 0), "input rows"),
        (stats.get("rows_accepted", 0), "rows accepted"),
        (len(res), "parcels analysed"),
        (sum(len(r.groups) for r in res), "duplicate groups"),
        (sum(len(g.members) - 1 for r in res for g in r.groups), "redundant addresses"),
        (sum(len(r.review_pairs) for r in res), "review candidates"),
    ]
    h = [CSS, "<div class='paa'><div class='tiles'>"]
    h += ["<div class='tile'><b>%s</b><span>%s</span></div>" % (format(v, ","), _e(t)) for v, t in tiles]
    h.append("</div></div>")
    return "".join(h)


PAGE_CSS = """<style>
body{margin:0;background:#fcfcfb}@media (prefers-color-scheme:dark){body{background:#1a1a19}}
.paa.page{max-width:1400px;margin:0 auto;padding:24px 16px 64px;background:var(--surface)}
.paa h1{font-size:22px;margin:0 0 4px}.paa section{border-top:1px solid var(--grid);margin-top:28px;padding-top:6px}
.paa a{color:var(--s1);text-decoration:none}.paa table{display:block;overflow-x:auto;max-width:100%}
</style>"""


def report_html(results: Iterable[ParcelResult], cfg: Config, stats: Dict[str, object],
                detailed: List[str], title: str = "Parcel Address Associator", index_rows: int = 1000,
                use: Optional[Dict[str, object]] = None) -> str:
    """Standalone HTML document: overview, parcel index, and a full section for each
    parcel in ``detailed``. Self-contained (inline CSS and SVG, no scripts)."""
    import datetime

    res = sorted(results, key=lambda r: (-len(r.groups), -len(r.feats), r.parcel))
    by_id = {r.parcel: r for r in res}
    detailed = [p for p in detailed if p in by_id]
    shown = set(detailed)
    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width,initial-scale=1">'
         '<title>%s</title>%s%s</head><body><div class="paa page">' % (_e(title), CSS, PAGE_CSS)]
    h.append("<h1>%s</h1><div class='sub'>Generated %s · %d parcels analysed · %d shown in detail below</div>"
             % (_e(title), datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), len(res), len(detailed)))
    h.append(overview_html(res, stats).replace(CSS, ""))
    h.append("<h4>Parcels (%s)</h4><table><tr><th>Parcel</th><th>Addresses</th><th>Styles</th><th>Structures</th>"
             "<th>Duplicate groups</th><th>Redundant addresses</th><th>Review pairs</th></tr>"
             % ("first %d of %d" % (index_rows, len(res)) if len(res) > index_rows else len(res)))
    for r in res[:index_rows]:
        name = ("<a href='#p-%s'>%s</a>" % (_e(r.parcel), _e(r.parcel))) if r.parcel in shown else _e(r.parcel)
        h.append("<tr><td class='mono'>%s</td><td class='n'>%d</td><td class='n'>%d</td><td class='n'>%d</td>"
                 "<td class='n'>%d</td><td class='n'>%d</td><td class='n'>%d</td></tr>"
                 % (name, len(r.feats), len(r.styles), len(r.structures), len(r.groups),
                    sum(len(g.members) - 1 for g in r.groups), len(r.review_pairs)))
    h.append("</table>")
    use_rows, use_blocks = {}, None
    if use:
        use_rows = {row["ADDR_ID"]: row for row in use["addresses"].to_dict("records")}
        use_blocks = use["blocks"]
    for pid in detailed:
        ub = use_blocks[use_blocks.CADASTRE_POLYGON_PID == pid] if use_blocks is not None and len(use_blocks) else None
        h.append("<section id='p-%s'>%s</section>" % (_e(pid), parcel_html(
            by_id[pid], cfg, include_css=False, use_rows=use_rows or None, use_blocks=ub)))
    h.append("</div></body></html>")
    return "".join(h)
