"""Confidence report rendering: machine-readable JSON, Markdown summary and a self-contained HTML page.

Inference-stage records can be reconciled against the final post-processed CSV so the report shows exactly the
columns and values the user receives, each labeled by its source.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

SOURCE_LABELS = {
    "model_inferred": "predicted by model",
    "model_inferred_adjusted": "predicted, then adjusted by postprocessing",
    "user_provided": "provided by you",
    "project_context": "taken from the GeoJSON project (e.g. weather file state, climate zone)",
    "postprocess_default": "filled by postprocessing (schema default / derived)",
}

_BANDS = ("high", "medium", "low", "unknown", "provided", "context", "default")
_FACT_KEYS = ("known_facts", "missing_facts", "suggested_facts")
NEAR_CONSTANT_BASELINE = 0.995  # most-common-value accuracy at or above which an attribute is left out of the lists

# GeoJSON building properties that supply each fact (see ``GeoJSONProcessor.map_properties``)
GEOJSON_FIELDS = {
    "Geometry Building Type RECS": "building_type",
    "Vintage": "year_built",
    "Geometry Floor Area": "floor_area",
    "Geometry Stories": "number_of_stories",
    "Heating Fuel": "heating_system_fuel_type",
    "HVAC Cooling Type": "system_type",
    "Geometry Foundation Type": "foundation_type",
    "Geometry Attic Type": "attic_type",
    "Bedrooms": "number_of_bedrooms",
    "Geometry Building Number Units SFA": "number_of_residential_units",
}


def _display(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    return str(v).strip()


def _norm(v) -> str:
    s = _display(v)
    return "" if s.lower() in ("nan", "none", "") else s


def _mean(values: list, digits: int = 3) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def _pts(x) -> str:
    return "" if x is None else f"{x * 100:+.1f} pts"


def _fact_label(name: str) -> str:
    field = GEOJSON_FIELDS.get(name)
    return f"{name} ({field})" if field else name


def reconcile_with_final(
    records: list[dict],
    final_df: pd.DataFrame,
    id_col: str,
    inference_columns: list[str],
) -> list[dict]:
    """Re-key inference-stage records onto the final file; every final column gets a source label.

    ``model_inferred[_adjusted]`` cells keep their ML confidence (adjusted = a schema or consistency rule changed the
    model's value), ``user_provided`` / ``project_context`` cells are inputs, anything else is a
    ``postprocess_default``. The reported value is always the final value.
    """
    inference_cols = set(inference_columns)
    by_id = {str(r["building"]): r for r in records}
    final_cols = [c for c in final_df.columns if c != id_col]

    reconciled: list[dict] = []
    for _, row in final_df.iterrows():
        src = by_id.get(str(row[id_col]))
        conf = src["confidence"] if src else {}
        known = set(conf.get("known_inputs", []))
        context = set(conf.get("context_inputs", []))
        old_attrs = conf.get("attributes", {})

        new_attrs: dict[str, dict] = {}
        for col in final_cols:
            final_val = _display(row[col])
            if col in known:
                new_attrs[col] = {"value": final_val, "source": "user_provided", "confidence": None, "band": "provided"}
            elif col in context:
                new_attrs[col] = {"value": final_val, "source": "project_context", "confidence": None, "band": "context"}
            elif col in old_attrs:
                d = dict(old_attrs[col])
                model_val = _norm(d.get("value"))
                d["value"] = final_val
                d["source"] = "model_inferred"
                if model_val != _norm(final_val):
                    d["source"] = "model_inferred_adjusted"
                    d["model_value"] = model_val
                    d["adjusted_by_postprocessor"] = True
                new_attrs[col] = d
            elif col in inference_cols:
                # survived from inference but was never scored (e.g. an excluded target)
                new_attrs[col] = {"value": final_val, "source": "model_inferred", "confidence": None, "band": "unknown"}
            else:
                new_attrs[col] = {
                    "value": final_val, "source": "postprocess_default", "confidence": None, "band": "default"
                }

        model_cells = [
            d for d in new_attrs.values() if d["source"].startswith("model_inferred") and d["confidence"] is not None
        ]
        confs = [d["confidence"] for d in model_cells]
        expected = [d["expected_accuracy"] for d in model_cells if d.get("expected_accuracy") is not None]
        summary = {
            "n_final_columns": len(new_attrs),
            "n_model_inferred": sum(d["source"].startswith("model_inferred") for d in new_attrs.values()),
            "n_user_provided": sum(d["source"] == "user_provided" for d in new_attrs.values()),
            "n_project_context": sum(d["source"] == "project_context" for d in new_attrs.values()),
            "n_postprocess_default": sum(d["source"] == "postprocess_default" for d in new_attrs.values()),
            "n_adjusted": sum(d["source"] == "model_inferred_adjusted" for d in new_attrs.values()),
            "mean_confidence": _mean(confs),
            "min_confidence": round(min(confs), 3) if confs else None,
            "n_high": sum(d["band"] == "high" for d in model_cells),
            "n_medium": sum(d["band"] == "medium" for d in model_cells),
            "n_low": sum(d["band"] == "low" for d in model_cells),
        }
        if expected:
            summary["mean_expected_accuracy"] = _mean(expected)
        confidence = {
            "known_inputs": sorted(known),
            "context_inputs": sorted(context),
            "n_known": len(known) + len(context),
            "attributes": new_attrs,
            "summary": summary,
        }
        for key in ("method", "model_id", *_FACT_KEYS):
            if key in conf:
                confidence[key] = conf[key]
        reconciled.append({"building": row[id_col], "confidence": confidence})
    return reconciled


def _fact_table(records: list[dict]) -> list[dict]:
    """Run-level view of the GeoJSON facts: how many buildings supplied each, and what adding it would gain."""
    order: dict[str, None] = {}
    known: dict[str, int] = {}
    gains: dict[str, list[float]] = {}
    improved: dict[str, list[int]] = {}
    for rec in records:
        c = rec["confidence"]
        for name in c.get("known_facts", []):
            order.setdefault(name)
            known[name] = known.get(name, 0) + 1
        for name in c.get("missing_facts", []):
            order.setdefault(name)
        for s in c.get("suggested_facts", []):
            gains.setdefault(s["fact"], []).append(s["mean_gain"])
            improved.setdefault(s["fact"], []).append(s.get("n_attributes_improved", 0))
    rows = []
    for name in order:
        g = gains.get(name, [])
        rows.append({
            "fact": name,
            "geojson_field": GEOJSON_FIELDS.get(name),
            "n_buildings_known": known.get(name, 0),
            "n_buildings_suggested": len(g),
            "mean_gain": _mean(g),
            "mean_attributes_improved": _mean(improved.get(name, []), 1),
        })
    return rows


def run_summary(records: list[dict]) -> dict:
    """Aggregate per-building records into run-level confidence stats."""
    per_attr: dict[str, list] = {}
    near_constant: set[str] = set()
    band_counts = dict.fromkeys(_BANDS, 0)
    source_counts: dict[str, int] = {}
    bld_conf, bld_expected = [], []
    adjusted: list[int] = []
    for rec in records:
        s = rec["confidence"]["summary"]
        if s.get("mean_confidence") is not None:
            bld_conf.append(s["mean_confidence"])
        if s.get("n_adjusted") is not None:
            adjusted.append(s["n_adjusted"])
        if s.get("mean_expected_accuracy") is not None:
            bld_expected.append(s["mean_expected_accuracy"])
        for attr, a in rec["confidence"]["attributes"].items():
            b = a.get("band", "unknown")
            band_counts[b] = band_counts.get(b, 0) + 1
            src = a.get("source")
            if src:
                source_counts[src] = source_counts.get(src, 0) + 1
            # only scored model cells (all cells before reconciliation) feed the weak/strong lists
            if a.get("confidence") is not None and str(src or "model_inferred").startswith("model_inferred"):
                per_attr.setdefault(attr, []).append(a["confidence"])
                if (a.get("baseline_accuracy") or 0) >= NEAR_CONSTANT_BASELINE:
                    near_constant.add(attr)

    attr_means = {a: round(sum(v) / len(v), 3) for a, v in per_attr.items() if v}
    # attributes (nearly) always equal to one value in ResStock are trivially "confident"; keep them out of the lists
    informative = [(a, m) for a, m in attr_means.items() if a not in near_constant]
    facts = _fact_table(records)
    suggested = sorted((f for f in facts if f["n_buildings_suggested"]),
                       key=lambda f: -(f["n_buildings_suggested"] * (f["mean_gain"] or 0)))
    return {
        "n_buildings": len(records),
        "mean_building_confidence": _mean(bld_conf),
        "mean_expected_accuracy": _mean(bld_expected),
        "cell_band_counts": band_counts,
        "source_counts": source_counts,
        "n_adjusted_by_postprocessing": sum(adjusted) if adjusted else None,
        "weakest_attributes": sorted(informative, key=lambda kv: kv[1])[:15],
        "strongest_attributes": sorted(informative, key=lambda kv: kv[1], reverse=True)[:15],
        "near_constant_attributes": sorted(near_constant),
        "facts": facts,
        "suggested_facts": [f["fact"] for f in suggested[:3]],
    }


def write_reports(
    records: list[dict],
    out_dir: str | Path,
    basename: str = "confidence_report",
    formats: Iterable[str] = ("json", "html"),
    meta: dict | None = None,
    validation: dict | None = None,
    summary: dict | None = None,
) -> list[str]:
    """Write ``<basename>.{json,md,html}``; ``records`` are ``{"building": id, "confidence": record}``.

    ``validation`` is the held-out reliability of the confidence model (``ConfidenceEstimator.validation``);
    ``summary`` is a precomputed :func:`run_summary` of ``records``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = set(formats)
    unknown = formats - {"json", "md", "html"}
    if unknown:
        raise ValueError(f"unknown report format(s) {sorted(unknown)}; use json, md, html")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "meta": meta or {},
        "validation": validation or {},
        "source_labels": SOURCE_LABELS,
        "run_summary": summary or run_summary(records),
        "buildings": records,
    }

    written = []
    if "json" in formats:
        p = out / f"{basename}.json"
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written.append(str(p))
    if "md" in formats:
        p = out / f"{basename}.md"
        p.write_text(_render_markdown(payload), encoding="utf-8")
        written.append(str(p))
    if "html" in formats:
        p = out / f"{basename}.html"
        p.write_text(_render_html(payload), encoding="utf-8")
        written.append(str(p))
    return written


def _validation_sentence(v: dict) -> str | None:
    if not v or v.get("ece_confidence") is None:
        return None
    n = v.get("n_validation_cells")
    on = f"On {n:,} held-out predictions" if isinstance(n, int) else "On held-out buildings"
    text = (f"{on}, stated confidence differed from observed accuracy by {v['ece_confidence'] * 100:.1f} points on "
            f"average (expected calibration error)")
    raw = v.get("ece_model_probability")
    if raw is not None and raw >= 1.5 * v["ece_confidence"] and raw - v["ece_confidence"] >= 0.002:
        text += f", compared with {raw * 100:.1f} points for the raw model probability"
    return text + "."


def _render_markdown(payload: dict) -> str:
    agg = payload["run_summary"]
    bc = agg["cell_band_counts"]
    lines = ["# USG Inference Confidence Report\n", f"_Generated {payload['generated_at']}_\n"]
    if payload["meta"]:
        lines += [f"- **{k}**: {v}" for k, v in payload["meta"].items()] + [""]
    lines.append("## Run summary\n")
    lines.append(f"- Buildings analyzed: **{agg['n_buildings']}**")
    lines.append(f"- Mean per-building confidence: **{agg['mean_building_confidence']}**")
    if agg.get("mean_expected_accuracy") is not None:
        lines.append(f"- Mean expected accuracy for these inputs: **{agg['mean_expected_accuracy']}**")
    lines.append(f"- Filled cells by band: high **{bc['high']}**, medium **{bc['medium']}**, low **{bc['low']}**")
    sentence = _validation_sentence(payload.get("validation", {}))
    if sentence:
        lines.append(f"- {sentence}")
    if agg.get("facts"):
        lines.append("\n### Input facts\n")
        lines.append("| Fact | GeoJSON property | Buildings that provided it | Would help (buildings) | Mean gain when added |\n"
                     "|---|---|---|---|---|")
        lines += [
            f"| {f['fact']} | {f.get('geojson_field') or ''} | {f['n_buildings_known']} | {f['n_buildings_suggested']} | "
            f"{_pts(f['mean_gain'])} |"
            for f in agg["facts"]
        ]
    lines.append("\n### Least confident attributes (review these)\n")
    lines.append("| Attribute | Mean confidence |\n|---|---|")
    lines += [f"| {a} | {c} |" for a, c in agg["weakest_attributes"]]
    lines.append("\n### Most confident attributes\n")
    lines.append("| Attribute | Mean confidence |\n|---|---|")
    lines += [f"| {a} | {c} |" for a, c in agg["strongest_attributes"]]
    if agg.get("near_constant_attributes"):
        lines.append(f"\n_{len(agg['near_constant_attributes'])} attributes that are (nearly) always the same value in "
                     f"ResStock are not listed._")
    lines.append("\n## Per-building detail\n")
    for rec in payload["buildings"]:
        c = rec["confidence"]
        s = c["summary"]
        lines.append(f"### Building `{rec['building']}`")
        lines.append(f"- Known inputs ({len(c['known_inputs'])}): {', '.join(c['known_inputs']) or '_none_'}")
        if c.get("context_inputs"):
            lines.append(f"- From the project: {', '.join(c['context_inputs'])}")
        if "n_postprocess_default" in s:
            lines.append(
                f"- Final columns: {s.get('n_final_columns')} | model-predicted: {s.get('n_model_inferred')} "
                f"(adjusted by postprocessing: {s.get('n_adjusted', 0)}) | user-provided: {s.get('n_user_provided')} | "
                f"project: {s.get('n_project_context', 0)} | postprocessing defaults: {s.get('n_postprocess_default')}"
            )
        lines.append(
            f"- Scored fills: {s.get('n_model_inferred', s.get('n_targets'))} | mean conf {s.get('mean_confidence')} | "
            f"low-confidence cells: {s.get('n_low', 0)}"
        )
        if c.get("suggested_facts"):
            tips = ", ".join(f"{_fact_label(x['fact'])} ({_pts(x['mean_gain'])})" for x in c["suggested_facts"])
            lines.append(f"- Most helpful facts to add: {tips}")
        low = [(a, d) for a, d in c["attributes"].items() if d.get("band") == "low"]
        if low:
            lines.append("- Low-confidence model fills:")
            lines += [f"  - `{a}` = {d['value']} (conf {d['confidence']})" for a, d in low[:20]]
        lines.append("")
    return "\n".join(lines)


def _attribute_detail(d: dict) -> str:
    parts = []
    if d.get("source") == "model_inferred_adjusted":
        parts.append(f"model said: {d.get('model_value')}")
    if d.get("type") == "numerical":
        parts.append(f"mean {d.get('ensemble_mean')} ± {d.get('ensemble_std')}")
    if d.get("expected_accuracy") is not None:
        parts.append(f"expected {d['expected_accuracy']}")
    if d.get("probability") is not None:
        parts.append(f"p {d['probability']}")
    if d.get("agreement") is not None:
        parts.append(f"agree {d['agreement']}")
    if d.get("drivers"):
        parts.append("driven by " + ", ".join(f"{x['fact']} ({_pts(x['effect'])})" for x in d["drivers"]))
    if d.get("suggestions"):
        parts.append("add " + ", ".join(f"{x['fact']} ({_pts(x['gain'])})" for x in d["suggestions"]))
    if d.get("type") == "categorical" and d.get("ensemble_alternatives") and d.get("source") != "model_inferred_adjusted":
        alts = ", ".join(f"{x['value']} ({int((x['share'] or 0) * 100)}%)" for x in d["ensemble_alternatives"])
        parts.append(f"alts: {alts}")
    if d.get("expected_accuracy") is None:
        if d.get("coverage") is not None:
            parts.append(f"cov {d['coverage']}")
        if d.get("calibrated_accuracy") is not None:
            parts.append(f"cal {d['calibrated_accuracy']}")
    return " · ".join(parts)


def _render_html(payload: dict) -> str:
    agg = payload["run_summary"]
    bc = agg["cell_band_counts"]
    e = html.escape

    def chip(b):
        return f'<span class="chip {e(str(b))}">{e(str(b))}</span>'

    def card(value, label, cls=""):
        shown = "–" if value is None else value
        return f'<div class="card"><div class="n {cls}">{e(str(shown))}</div><div class="muted">{e(label)}</div></div>'

    rows_weak = "".join(f"<tr><td>{e(a)}</td><td>{c}</td></tr>" for a, c in agg["weakest_attributes"])
    rows_strong = "".join(f"<tr><td>{e(a)}</td><td>{c}</td></tr>" for a, c in agg["strongest_attributes"])
    n_constant = len(agg.get("near_constant_attributes", []))
    constant_note = (f'<p class="muted">{n_constant} attributes that are (nearly) always the same value in ResStock '
                     f'are not listed.</p>') if n_constant else ""
    src_short = {
        "model_inferred": "model",
        "model_inferred_adjusted": "model→adjusted",
        "user_provided": "you",
        "project_context": "project",
        "postprocess_default": "postprocess",
    }

    facts_html = ""
    if agg.get("facts"):
        n = agg["n_buildings"]
        helps = any(f["n_buildings_suggested"] for f in agg["facts"])
        fact_rows = "".join(
            f"<tr><td>{e(f['fact'])}</td><td><code>{e(f.get('geojson_field') or '')}</code></td>"
            f"<td>{f['n_buildings_known']} / {n}</td>"
            + (f"<td>{f['n_buildings_suggested'] or ''}</td><td>{e(_pts(f['mean_gain']))}</td>" if helps else "")
            + "</tr>"
            for f in agg["facts"]
        )
        extra_heads = "<th>Would help (buildings)</th><th>Mean gain when added</th>" if helps else ""
        tip = ""
        if agg.get("suggested_facts"):
            labels = ", ".join(f"<b>{e(_fact_label(name))}</b>" for name in agg["suggested_facts"])
            tip = f"<p>Adding {labels} to the GeoJSON buildings would raise expected accuracy the most for this project.</p>"
        elif all(f["n_buildings_known"] == n for f in agg["facts"]):
            tip = "<p>Every building supplied all of these facts.</p>"
        facts_html = f"""
  <h3>Input facts</h3>
  <p class="muted">The facts the models read from each building. Confidence is higher when more of them are known; the
   gain is the average rise in expected accuracy across the predicted attributes when that fact is added.</p>
  {tip}
  <div class="scroll"><table>
    <thead><tr><th>Fact</th><th>GeoJSON property</th><th>Buildings that provided it</th>{extra_heads}</tr></thead>
    <tbody>{fact_rows}</tbody>
  </table></div>"""

    buildings_html = []
    for rec in payload["buildings"]:
        c = rec["confidence"]
        s = c["summary"]
        attr_rows = []
        for a, d in sorted(c["attributes"].items(), key=lambda kv: kv[1].get("confidence") or 0):
            src = d.get("source", "")
            attr_rows.append(
                f"<tr><td>{e(a)}</td><td>{e(_display(d.get('value')))}</td>"
                f"<td class='muted'>{e(src_short.get(src, src))}</td>"
                f"<td>{'' if d.get('confidence') is None else d['confidence']}</td>"
                f"<td>{chip(d.get('band', 'unknown'))}</td>"
                f"<td class='muted'>{e(_attribute_detail(d))}</td></tr>"
            )
        if "n_postprocess_default" in s:
            sub = (
                f"{s.get('n_model_inferred')} model-predicted ({s.get('n_adjusted', 0)} adjusted), "
                f"{s.get('n_user_provided')} yours, {s.get('n_project_context', 0)} from the project, "
                f"{s.get('n_postprocess_default')} postprocess defaults"
            )
        else:
            sub = f"{s.get('n_targets')} targets"
        known = ", ".join(c["known_inputs"]) or "none"
        facts_line = ""
        if "known_facts" in c:
            n_facts = len(c.get("known_facts", [])) + len(c.get("missing_facts", []))
            facts_line = f"<p class='muted'>Facts known: {len(c['known_facts'])} of {n_facts}"
            if c.get("missing_facts"):
                facts_line += f" (missing: {e(', '.join(c['missing_facts']))})"
            facts_line += "</p>"
            if c.get("suggested_facts"):
                tips = ", ".join(f"{_fact_label(x['fact'])} ({_pts(x['mean_gain'])})" for x in c["suggested_facts"])
                facts_line += f"<p class='muted'>Most helpful to add: {e(tips)}</p>"
        expected = s.get("mean_expected_accuracy")
        expected_txt = "" if expected is None else f", expected accuracy {expected}"
        buildings_html.append(f"""
        <details>
          <summary><b>Building {e(str(rec['building']))}</b> &mdash; {e(sub)};
            mean model conf {s.get('mean_confidence')}{expected_txt}, {s.get('n_low', 0)} low-confidence
            ({c['n_known']} known inputs)</summary>
          <p class="muted">Known: {e(known)}</p>
          {facts_line}
          <div class="scroll"><table>
            <thead><tr><th>Attribute</th><th>Value</th><th>Source</th><th>Conf.</th><th>Band</th><th>Detail</th></tr></thead>
            <tbody>{''.join(attr_rows)}</tbody>
          </table></div>
        </details>""")

    meta_html = "".join(f"<li><b>{e(str(k))}:</b> {e(str(v))}</li>" for k, v in payload["meta"].items())
    sentence = _validation_sentence(payload.get("validation", {}))
    if agg.get("mean_expected_accuracy") is not None:
        note = ("Confidence is the calibrated chance that a filled value is right. It comes from small models fit on "
                "held-out ResStock buildings and uses two things: how strongly the model favoured this value for this "
                "building, and which facts you supplied. <b>Expected accuracy</b> is how often the attribute is right "
                "for buildings with the same known facts. <b>Driven by</b> lists the known facts that raise it most; "
                "<b>add</b> lists missing facts that would raise it.")
    else:
        note = ("Confidence combines the model's probability for each filled value and/or how stable the value stays "
                "when some of your inputs are hidden, optionally anchored to held-out calibration.")
    note += " It is a guide to which fills to trust or review &mdash; not a guarantee of correctness."
    if agg.get("n_adjusted_by_postprocessing") is not None:
        note += (f"<br><br><b>Post-processing.</b> {agg['n_adjusted_by_postprocessing']:,} predicted value(s) were "
                 "changed by option validation or consistency rules; they are marked <i>model→adjusted</i> and "
                 "listed in postprocess_report.html.")
    if sentence:
        note += f"<br><br><b>Reliability.</b> {e(sentence)}"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USG Inference Confidence Report</title>
<style>
  :root {{ --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --soft:#eaeef2; --panel:#f6f8fa; --bg:#ffffff;
          --accent:#0969da; --high:#1a7f37; --medium:#9a6700; --low:#cf222e; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink:#e6edf3; --muted:#8d96a0; --line:#30363d; --soft:#21262d; --panel:#161b22; --bg:#0d1117;
            --accent:#4493f8; --high:#3fb950; --medium:#d29922; --low:#f85149; }}
  }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:var(--bg);
         margin: 2rem auto; max-width: 1040px; color:var(--ink); line-height:1.5; padding:0 1rem; }}
  h1 {{ font-size: 22px; margin-bottom: .25rem; }}
  .muted {{ color:var(--muted); font-size: 13px; }}
  td.muted {{ font-size: 12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:.75rem; margin:1rem 0; }}
  .card {{ border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; }}
  .card .n {{ font-size: 26px; font-weight:700; font-variant-numeric: tabular-nums; }}
  .n.high {{ color:var(--high); }} .n.medium {{ color:var(--medium); }} .n.low {{ color:var(--low); }}
  .chip {{ font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.04em; color:var(--muted); }}
  .chip.high {{ color:var(--high); }} .chip.medium {{ color:var(--medium); }} .chip.low {{ color:var(--low); }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse: collapse; width:100%; margin:.5rem 0 1rem; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--soft); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; }}
  details {{ border:1px solid var(--line); border-radius:10px; padding:.5rem 1rem; margin:.5rem 0; }}
  summary {{ cursor:pointer; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
  @media(max-width:720px){{ .grid2{{grid-template-columns:1fr}} }}
  .note {{ background:var(--panel); border-left:3px solid var(--accent); padding:.75rem 1rem; border-radius:4px;
          font-size:13px; }}
</style></head><body>
  <h1>USG Inference Confidence Report</h1>
  <p class="muted">Generated {payload['generated_at']}</p>
  <ul class="muted">{meta_html}</ul>
  <div class="cards">
    {card(agg['n_buildings'], 'buildings')}
    {card(agg['mean_building_confidence'], 'mean confidence')}
    {card(agg['mean_expected_accuracy'], 'mean expected accuracy') if agg.get('mean_expected_accuracy') is not None else ''}
    {card(bc['high'], 'high-conf cells', 'high')}
    {card(bc['medium'], 'medium', 'medium')}
    {card(bc['low'], 'low', 'low')}
  </div>
  <div class="note">{note}</div>
  {facts_html}
  <div class="grid2">
    <div><h3>Least confident attributes</h3>
      <table><thead><tr><th>Attribute</th><th>Mean conf.</th></tr></thead><tbody>{rows_weak}</tbody></table></div>
    <div><h3>Most confident attributes</h3>
      <table><thead><tr><th>Attribute</th><th>Mean conf.</th></tr></thead><tbody>{rows_strong}</tbody></table>
      {constant_note}</div>
  </div>
  <h3>Per-building detail</h3>
  {''.join(buildings_html)}
</body></html>"""
