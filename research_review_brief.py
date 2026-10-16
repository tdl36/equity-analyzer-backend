"""Compact recovery brief from explicitly revalidated editorial evidence."""
import html
from onepager_edition_evidence import (AcceptedEditionEvidence, PendingEditionEvidence,
    check_render_inputs, render_evidence, preview_notice)
from research_source_provenance import render as source_provenance


def render(run, assignment, *, preview):
    if type(preview) not in (PendingEditionEvidence, AcceptedEditionEvidence):
        raise ValueError("Recovery brief requires a validated editorial edition.")
    check_render_inputs(run, assignment, None, preview)
    origin = "investor" if type(preview) is AcceptedEditionEvidence else "pending_editorial"
    derived, evidence = render_evidence(run, preview)
    sources = derived['sources']; refs={s['id']:i+1 for i,s in enumerate(sources)}
    esc=lambda x:html.escape(str(x),quote=True)
    def field(path):
        c=evidence.get(path)
        if not c or c.get('reviewOrigin') != origin:
            raise ValueError('Recovery brief requires an explicit edition correction for '+path)
        if len(c['statement'])>300:raise ValueError('Shorten the brief claim without truncating its meaning.')
        ids=sorted({refs[e['sourceId']] for e in c.get('evidence',[]) if e.get('matched') and e.get('sourceId') in refs})
        if not ids:raise ValueError('Recovery brief source reference missing.')
        return esc(c['statement'])+'<sup>['+','.join(map(str,ids))+']</sup>'
    def section(title, paths):
        return '<section><h2>'+esc(title)+'</h2>'+''.join('<p>'+field(p)+'</p>' for p in paths)+'</section>'
    blocks=section('The business & ownership case',['/summary/one_liner','/summary/investment_thesis/0'])
    blocks+=section('What changed / what remains contested',['/summary/earnings_direction','/debates/0/evidence_for_market','/debates/0/evidence_for_alternative'])
    blocks+=section('Valuation discipline',['/valuation/summary','/valuation/metrics/0/historical_range','/valuation/metrics/0/peer_reference'])
    blocks+=section('Next evidence to watch',['/monitor/kpis/0/current','/monitor/kpis/0/bull_threshold','/monitor/kpis/0/bear_threshold','/monitor/kpis/0/timing'])
    rows=''
    names=[str(row.get('name','')).strip().title() for row in derived.get('state',{}).get('report',{}).get('scenarios',[])]
    if len(names)!=3 or set(names)!={'Bull','Base','Bear'}:
        raise ValueError('Recovery brief requires three distinct named scenarios.')
    for i,name in enumerate(names):
        root=f'/scenarios/{i}/'
        rows+='<tr><th>'+name+'</th><td>'+field(root+'business_conditions')+'</td><td>'+field(root+'eps_or_fcf')+'<br>'+field(root+'valuation_multiple')+'</td><td>'+field(root+'implied_value')+'<br>'+field(root+'upside_downside')+'</td></tr>'
    short=' · '.join('['+str(i)+'] '+esc(s['filename']) for i,s in enumerate(sources,1))
    register=''.join('<li>'+esc(s['filename'])+'<br>SHA-256 '+esc(s['originalHash'])+source_provenance(s, assignment)+'</li>' for s in sources)
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box;color:#000;font-family:Calibri,sans-serif}body{background:#edf0ef;margin:0;font-size:14px;line-height:1.3}.sheet{max-width:920px;background:white;margin:20px auto;padding:30px}header{border-bottom:2px solid #000;padding-bottom:10px}h1{font-size:30px;margin:0 0 4px}h2{font-size:15px;margin:0 0 7px}p{margin:0 0 8px}.meta,.notice,.edition-notice{font-size:12px}.notice{padding:8px;background:#eef2f0;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px 22px}section{border-top:1px solid #999;padding-top:9px;break-inside:avoid}sup{font-size:8px}table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #ccc;padding:7px}thead{background:#eef2f0}th:first-child{width:7%}td:nth-child(2){width:27%}td:nth-child(3){width:33%}footer{overflow-wrap:anywhere;font-size:10px;margin-top:12px}.scenarios{margin-top:14px}details{max-width:920px;margin:20px auto;padding:20px;background:white;overflow-wrap:anywhere}@media(max-width:650px){.sheet{padding:18px;margin:0}.grid{display:block}section{margin-top:14px}table{font-size:11px}td,th{padding:4px}}@media print{@page{size:A4;margin:12mm}body{background:white;font-size:9pt;line-height:1.2}.sheet{padding:0;margin:0;max-width:none}h1{font-size:23pt}h2{font-size:10pt}.meta,.notice,.edition-notice{font-size:8pt}.notice{margin:8px 0;padding:6px}.grid{gap:9px 15px}section{padding-top:6px}p{margin-bottom:5px}table{font-size:8pt}td,th{padding:5px}footer{font-size:7pt;line-height:1.2}sup{font-size:6pt}details{display:none}}</style></head><body><article class="sheet">'''+f'<header><h1>{esc(run["ticker"])} · Reconciled investment brief</h1><div class="meta">Discovery window {esc(assignment.get("since", "unknown"))}–{esc(assignment.get("until", "unknown"))} · {esc(assignment.get("horizon",""))} horizon</div></header>{preview_notice(preview, run)}<p class="notice">Editorial recovery. This edition does not apply thesis changes or reclassify the original report’s evidence review. Discovery dates limit source search, not the longitudinal investment case.</p><main class="grid">{blocks}</main><section class="scenarios"><h2>Scenario assumptions · retain each case’s stated period and qualifications</h2><table><thead><tr><th>Case</th><th>Operating conditions</th><th>EPS &amp; multiple</th><th>Value &amp; price return</th></tr></thead><tbody>{rows}</tbody></table></section><footer><p>{len(sources)} original sources in the saved report. Source references do not establish exhaustive review. Publication dates, extraction limits and selection provenance are listed in the source register. Estimates and historical comparisons retain their source dates; this rendering does not fetch a new historical series or live quote. See the saved research for additional monitoring, risks and evidence gaps.</p><p>{short}</p><p>Edition {esc(preview.edition_id)} · report {esc(run["id"])}</p></footer></article><details><summary>Original identities and source limitations</summary><ol>{register}</ol></details></body></html>'
