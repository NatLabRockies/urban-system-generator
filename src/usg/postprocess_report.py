"""What post-processing changed, cell by cell, and why.

Post-processing (:class:`usg.postprocessor.USGPostProcessor`) does three things:
- adds the required columns the model does not predict;
- replaces values that are not valid options;
- applies consistency rules that repair contradictions between attributes (e.g. ducted heating without ducts).

If those steps rewrite many of the model's predictions, the predictions are not usable as they are. Every run
therefore records two outputs.

``postprocess_changes.csv`` has one row per changed cell:
- building, attribute, and the value before -> after;
- where the value came from: predicted by the model, provided by you, or the project location;
- the step and rule(s) that changed it.

``postprocess_report.json`` / ``.html`` summarize the run:
- the share of predicted values and of your inputs that changed;
- changes by rule and by attribute;
- empty cells filled and columns added with defaults;
- rule violations still present in the final file.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from usg.constants import ID_COLUMN

REPORT_FORMAT = "usg-postprocess-report/v1"
FEATURE_ID_COLUMN = "Feature ID"
REVIEW_SHARE = 0.05  # flag a run when post-processing changes more than 5% of the predicted values
MAX_HTML_CHANGES = 500
SOURCES = ("model", "input", "project")
SOURCE_LABELS = {"model": "predicted by the model", "input": "provided by you", "project": "project location"}
CHANGE_COLUMNS = ["Building", "row", "column", "source", "before", "after", "step", "rules", "note"]


def _text(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


FILL_KINDS = ("derived", "most common for", "most common nationally", "project location", "state default",
              "project climate zone", "static default")


def _methods(fill_log: dict | None, column: str) -> list[dict]:
    """How post-processing filled ``column``: ``[{method, n}]``, most cells first."""
    counts = (fill_log or {}).get(column, {})
    return [{"method": m, "n": int(n)} for m, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def _fill_kinds(fill_log: dict | None) -> dict[str, int]:
    kinds: Counter = Counter()
    for counts in (fill_log or {}).values():
        for label, n in counts.items():
            kinds[next((k for k in FILL_KINDS if label.startswith(k)), label)] += int(n)
    return dict(kinds.most_common())


@dataclass
class PostprocessReport:
    summary: dict
    changes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CHANGE_COLUMNS))

    @property
    def headline(self) -> str:
        return self.summary["headline"]

    @property
    def needs_review(self) -> bool:
        return self.summary["verdict"] == "review"


def _sources(column: str, n: int, imputed: pd.DataFrame | None, context: set[str]) -> pd.Series:
    if column in context:
        return pd.Series(["project"] * n)
    if imputed is not None and column in imputed.columns:
        return imputed[column].fillna(False).astype(bool).map({True: "model", False: "input"}).reset_index(drop=True)
    return pd.Series(["input"] * n)


def _remaining_violations(after: pd.DataFrame, rules: Iterable[dict], ids: list[str]) -> list[dict]:
    from usg.postprocessor import eval_condition, eval_then_clause

    records = after.to_dict("records")
    remaining = []
    for rule in rules:
        when, then = rule.get("when") or {}, rule.get("then") or []
        if not then:
            continue
        bad = []
        for i, rec in enumerate(records):
            try:
                if (when and not eval_condition(when, rec)) or all(eval_then_clause(c, rec) for c in then):
                    continue
            except Exception:  # noqa: BLE001, S112 - the consistency engine already reports malformed rules
                continue
            bad.append(i)
        if bad:
            remaining.append({
                "rule_id": rule.get("id", "rule_without_id"),
                "severity": rule.get("severity", "error"),
                "note": rule.get("note", ""),
                "buildings": len(bad),
                "examples": [ids[i] for i in bad[:5]],
            })
    return remaining


def build_postprocess_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    schema_issues: Iterable = (),
    audit_events: Iterable = (),
    rules: Iterable[dict] = (),
    *,
    imputed: pd.DataFrame | None = None,
    context_columns: Iterable[str] = (),
    id_col: str = ID_COLUMN,
    fill_log: dict | None = None,
) -> PostprocessReport:
    """Compare the file before and after post-processing.

    ``imputed`` (True where the model filled a cell, aligned with ``before`` rows) and ``context_columns`` (location
    values taken from the project) label each value's source. Without ``imputed`` every value counts as input.
    """
    before = before.reset_index(drop=True)
    after = after.reset_index(drop=True)
    if imputed is not None:
        imputed = imputed.reset_index(drop=True) if len(imputed) == len(before) else None
    rules = list(rules)
    schema_issues, audit_events = list(schema_issues), list(audit_events)
    context = set(context_columns)
    ids = [_text(v) for v in after[id_col]] if id_col in after.columns else [str(i) for i in range(len(after))]

    schema_fix = {(i.row_index, i.column): i for i in schema_issues if i.fixed_to_default_value}
    rule_fix: dict[tuple[int, str], list[str]] = defaultdict(list)
    for ev in audit_events:
        if ev.status == "FIXED" and ev.field:
            rule_fix[(ev.row_index, ev.field)].append(ev.rule_id)

    totals: Counter = Counter()
    filled: Counter = Counter()
    rows: list[dict] = []
    shared = [c for c in before.columns if c in after.columns and c not in (id_col, FEATURE_ID_COLUMN)]
    for column in shared:
        old = before[column].map(_text)
        new = after[column].map(_text)
        present = old != ""
        source = _sources(column, len(before), imputed, context)
        totals.update(source[present].tolist())
        filled[column] = int((~present & (new != "")).sum())
        for i in (present & (old != new))[lambda s: s].index:
            steps, notes = [], []
            if (i, column) in schema_fix:
                steps.append("schema_validation")
                notes.append(schema_fix[(i, column)].issue_type)
            rule_ids = list(dict.fromkeys(rule_fix.get((i, column), [])))
            if rule_ids:
                steps.append("consistency_rule")
            rows.append({
                "Building": ids[i], "row": int(i), "column": column, "source": source.iat[i],
                "before": old.iat[i], "after": new.iat[i], "step": "+".join(steps) or "normalization",
                "rules": "; ".join(rule_ids), "note": "; ".join(notes),
            })
    changes = pd.DataFrame(rows, columns=CHANGE_COLUMNS)

    changed = Counter(r["source"] for r in rows)
    cells = {
        s: {"total": int(totals[s]), "changed": int(changed[s]),
            "share": round(changed[s] / totals[s], 4) if totals[s] else 0.0}
        for s in SOURCES
    }

    remaining = _remaining_violations(after, rules, ids)
    rule_meta = {r.get("id", "rule_without_id"): r for r in rules}
    detected: Counter = Counter()
    for rule_id, _row in {(ev.rule_id, ev.row_index) for ev in audit_events if ev.status == "VIOLATION"}:
        detected[rule_id] += 1
    by_rule_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for rule_id in filter(None, r["rules"].split("; ")):
            by_rule_rows[rule_id].append(r)
    still = {v["rule_id"]: v["buildings"] for v in remaining}
    by_rule = sorted(
        (
            {
                "rule_id": rule_id,
                "severity": rule_meta.get(rule_id, {}).get("severity", ""),
                "note": rule_meta.get(rule_id, {}).get("note", ""),
                "changes": len(by_rule_rows[rule_id]),
                "buildings": len({r["row"] for r in by_rule_rows[rule_id]}),
                "columns": sorted({r["column"] for r in by_rule_rows[rule_id]}),
                "violations": detected[rule_id],
                "remaining": still.get(rule_id, 0),
            }
            for rule_id in set(detected) | set(by_rule_rows) | set(still)
        ),
        key=lambda d: (-d["changes"], -d["violations"], d["rule_id"]),
    )

    by_column = []
    for column, group in changes.groupby("column", sort=False):
        pairs = Counter(zip(group["before"], group["after"])).most_common(3)
        counts = group["source"].value_counts()
        by_column.append({
            "column": column, "changed": len(group),
            **{s: int(counts.get(s, 0)) for s in SOURCES},
            "top": [{"before": b, "after": a, "n": n} for (b, a), n in pairs],
        })
    by_column.sort(key=lambda d: -d["changed"])

    added = []
    for column in (c for c in after.columns if c not in before.columns and c != FEATURE_ID_COLUMN):
        values = after[column].map(_text).value_counts().head(3)
        added.append({"column": column, "values": [{"value": v, "n": int(n)} for v, n in values.items()],
                      "methods": _methods(fill_log, column)})
    dropped = [c for c in before.columns if c not in after.columns]

    provenance = imputed is not None
    n_remaining = sum(v["buildings"] for v in remaining)
    verdict = "review" if cells["model"]["share"] > REVIEW_SHARE or (provenance and cells["input"]["changed"]) else "ok"
    if provenance:
        headline = (
            f"Post-processing changed {cells['model']['changed']:,} of {cells['model']['total']:,} predicted values "
            f"({_pct(cells['model']['share'])}) and {cells['input']['changed']:,} of your "
            f"{cells['input']['total']:,} input values"
        )
    else:
        headline = (
            f"Post-processing changed {cells['input']['changed']:,} of {cells['input']['total']:,} values in the input "
            f"file ({_pct(cells['input']['share'])})"
        )
    headline += (f"; filled {sum(filled.values()):,} empty cells and added {len(added)} required columns the model "
                 f"does not predict")
    if n_remaining:
        headline += f"; {n_remaining:,} rule violation(s) remain"

    summary = {
        "report_format": REPORT_FORMAT,
        "headline": headline,
        "verdict": verdict,
        "review_threshold": REVIEW_SHARE,
        "provenance": provenance,
        "n_buildings": len(after),
        "n_columns_in": len(before.columns),
        "n_columns_out": len(after.columns),
        "cells": cells,
        "n_changed": len(rows),
        "changes_by_step": dict(Counter(r["step"] for r in rows)),
        "n_filled_empty": int(sum(filled.values())),
        "filled_empty_by_column": [{"column": c, "n": n, "methods": _methods(fill_log, c)}
                                   for c, n in filled.most_common() if n],
        "fill_methods": _fill_kinds(fill_log),
        "by_rule": by_rule,
        "by_column": by_column,
        "remaining_violations": remaining,
        "columns_added": added,
        "columns_dropped": dropped,
        "schema_issues": dict(Counter(i.issue_type for i in schema_issues)),
    }
    return PostprocessReport(summary=summary, changes=changes)


def write_postprocess_report(
    report: PostprocessReport,
    out_dir: str | Path,
    formats: Iterable[str] = ("json", "html"),
    meta: dict | None = None,
    basename: str = "postprocess_report",
) -> list[str]:
    """Write ``postprocess_changes.csv`` plus ``<basename>.json`` / ``.html`` (other formats are ignored)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = set(formats)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    written = []
    changes_path = out / "postprocess_changes.csv"
    report.changes.to_csv(changes_path, index=False)
    written.append(str(changes_path))
    if "json" in formats:
        path = out / f"{basename}.json"
        payload = {"generated_at": generated_at, "meta": meta or {}, **report.summary}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written.append(str(path))
    if "html" in formats:
        path = out / f"{basename}.html"
        path.write_text(_render_html(report, generated_at, meta or {}), encoding="utf-8")
        written.append(str(path))
    return written


_STYLE = """
  :root { --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --soft:#eaeef2; --panel:#f6f8fa; --bg:#ffffff;
          --accent:#0969da; --good:#1a7f37; --warn:#9a6700; --bad:#cf222e; }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e6edf3; --muted:#8d96a0; --line:#30363d; --soft:#21262d; --panel:#161b22; --bg:#0d1117;
            --accent:#4493f8; --good:#3fb950; --warn:#d29922; --bad:#f85149; }
  }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:var(--bg);
         margin: 2rem auto; max-width: 1040px; color:var(--ink); line-height:1.5; padding:0 1rem; }
  h1 { font-size: 22px; margin-bottom: .25rem; }
  .muted { color:var(--muted); font-size: 13px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.75rem; margin:1rem 0; }
  .card { border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; }
  .card .n { font-size: 24px; font-weight:700; font-variant-numeric: tabular-nums; }
  .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
  .note { background:var(--panel); border-left:3px solid var(--accent); padding:.75rem 1rem; border-radius:4px;
          font-size:13px; }
  .note.review { border-left-color:var(--warn); }
  .scroll { overflow-x:auto; }
  table { border-collapse: collapse; width:100%; margin:.5rem 0 1rem; font-size:13px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--soft); vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  code { font-size:12px; }
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """``rows`` hold already-escaped HTML."""
    if not rows:
        return '<p class="muted">None.</p>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _render_html(report: PostprocessReport, generated_at: str, meta: dict) -> str:
    e = html.escape
    s = report.summary
    cells, provenance = s["cells"], s["provenance"]

    def card(value: str, label: str, cls: str = "") -> str:
        return f'<div class="card"><div class="n {cls}">{e(value)}</div><div class="muted">{e(label)}</div></div>'

    model, inputs, project = cells["model"], cells["input"], cells["project"]
    if provenance:
        cards = [
            card(f"{model['changed']:,} / {model['total']:,}", f"predicted values changed ({_pct(model['share'])})",
                 "warn" if model["share"] > s["review_threshold"] else "good"),
            card(f"{inputs['changed']:,} / {inputs['total']:,}", "of your input values changed",
                 "bad" if inputs["changed"] else "good"),
        ]
        if project["total"]:
            cards.append(card(f"{project['changed']:,} / {project['total']:,}", "project location values changed"))
    else:
        cards = [card(f"{inputs['changed']:,} / {inputs['total']:,}",
                      f"values in the input file changed ({_pct(inputs['share'])})")]
    n_remaining = sum(v["buildings"] for v in s["remaining_violations"])
    cards += [
        card(f"{s['n_filled_empty']:,}", "empty cells filled"),
        card(f"{len(s['columns_added'])}", "required columns added (not predicted)"),
        card(f"{n_remaining:,}", "rule violations remaining", "bad" if n_remaining else "good"),
    ]

    purpose = (
        "Post-processing makes the completed file valid for URBANopt-BuildStock. It adds the required columns the "
        "model does not predict — derived from ResStock consistently with the known and predicted attributes (e.g. "
        "every location column from one ResStock location), per-state defaults only when nothing else informs them "
        "— replaces values that are not valid options, "
        "and applies consistency rules that repair contradictions between attributes (for example ducted heating "
        "without ducts). A few changes are expected; if many predicted values change, review the rules or the inputs "
        "before using the results."
    )
    if s["verdict"] == "review":
        purpose = f"<b>Review this run.</b> {e(s['headline'])}.<br><br>{purpose}"

    input_label = "Your inputs" if provenance else "Input values"
    by_rule = _table(
        ["Rule", "Severity", "Values changed", "Buildings", "Attributes changed", "Violations found", "Still violated",
         "Note"],
        [[f"<code>{e(r['rule_id'])}</code>", e(r["severity"]), f"{r['changes']:,}", f"{r['buildings']:,}",
          e(", ".join(r["columns"])), f"{r['violations']:,}", f"{r['remaining']:,}", e(r["note"])]
         for r in s["by_rule"]],
    )
    by_column = _table(
        ["Attribute", "Changed", "Predicted", input_label, "Project", "Most common changes"],
        [[e(c["column"]), f"{c['changed']:,}", f"{c['model']:,}", f"{c['input']:,}", f"{c['project']:,}",
          e("; ".join(f"{t['before']} → {t['after']} ({t['n']})" for t in c["top"]))]
         for c in s["by_column"]],
    )
    remaining = _table(
        ["Rule", "Severity", "Buildings", "Examples", "Note"],
        [[f"<code>{e(v['rule_id'])}</code>", e(v["severity"]), f"{v['buildings']:,}", e(", ".join(v["examples"])),
          e(v["note"])] for v in s["remaining_violations"]],
    )
    def how(entry: dict) -> str:
        return e("; ".join(f"{m['method']} ({m['n']:,})" for m in entry.get("methods", [])))

    added = _table(
        ["Column", "Value(s) filled", "How it was filled"],
        [[e(c["column"]), e("; ".join(f"{v['value'] or '(empty)'} ({v['n']})" for v in c["values"])), how(c)]
         for c in s["columns_added"]],
    )
    filled = _table(
        ["Column", "Empty cells filled", "How it was filled"],
        [[e(c["column"]), f"{c['n']:,}", how(c)] for c in s["filled_empty_by_column"]],
    )
    shown = report.changes.head(MAX_HTML_CHANGES).to_dict("records")
    details = _table(
        ["Building", "Attribute", "Source", "Before", "After", "Step", "Rule(s)"],
        [[e(str(r["Building"])), e(r["column"]), e(SOURCE_LABELS.get(r["source"], r["source"])), e(r["before"]),
          e(r["after"]), e(r["step"]), e(r["rules"] or r["note"])] for r in shown],
    )
    more = ""
    if len(report.changes) > MAX_HTML_CHANGES:
        more = f'<p class="muted">First {MAX_HTML_CHANGES} of {len(report.changes):,}; all changes are in postprocess_changes.csv.</p>'
    meta_html = "".join(f"<li><b>{e(str(k))}:</b> {e(str(v))}</li>" for k, v in meta.items())
    dropped = e(", ".join(s["columns_dropped"])) or "none"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USG Post-processing Report</title>
<style>{_STYLE}</style></head><body>
  <h1>USG Post-processing Report</h1>
  <p class="muted">Generated {e(generated_at)} · {s['n_buildings']:,} buildings · {s['n_columns_in']} columns in,
   {s['n_columns_out']} out</p>
  <ul class="muted">{meta_html}</ul>
  <div class="cards">{''.join(cards)}</div>
  <div class="{'note review' if s['verdict'] == 'review' else 'note'}">{purpose}</div>
  <h3>Changes by consistency rule</h3>
  {by_rule}
  <h3>Changes by attribute</h3>
  {by_column}
  <h3>Rule violations still in the final file</h3>
  {remaining}
  <h3>Required columns the model does not predict</h3>
  {added}
  <h3>Empty cells filled</h3>
  {filled}
  <p class="muted">Columns dropped (not used by URBANopt-BuildStock): {dropped}</p>
  <h3>Every change</h3>
  {details}
  {more}
</body></html>"""
