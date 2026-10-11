"""Editorial one-page research brief from reviewed evidence; no model calls.

Keep claims whole, attach references, and never present an orphan numeric field as
an identified metric or a sector multiple as company valuation.
"""
from research_extraction import has_omissions
from stock_evidence import valid_citations,report_fields,missing_value
import html
import re
from urllib.parse import urlparse

VERSION = 'research-onepager-v3'


def esc(value):
    return html.escape(str(value or ''), quote=True)


def render(run, assignment):
    if assignment.get('_visualFundamentals'):
        from research_infographic import render as infographic
        return infographic(run, assignment, assignment['_visualFundamentals'])
    citations = valid_citations(run)
    sources = run.get('sources', [])
    refs = {s['id']: i + 1 for i, s in enumerate(sources)}

    def valid(c):
        return bool(c and c.get('review') == 'supported' and c.get('passageMatched')
                    and c.get('statement') and any(e.get('matched') and e.get('sourceId') in refs for e in c.get('evidence', [])))

    def get(path):
        c = citations.get(path)
        return c if valid(c) else None

    def select(prefix, limit=2):
        return [(p, c) for p, c in sorted(citations.items()) if p.startswith(prefix) and valid(c)][:limit]

    used = set()

    def claim(c, label=None):
        if not c:
            return ''
        ids = sorted({refs[e['sourceId']] for e in c.get('evidence', []) if e.get('matched') and e.get('sourceId') in refs})
        used.update(ids)
        # A mixed factual/conditional statement remains a source-backed view, not a certified fact.
        basis = {'broker estimate': 'Broker estimate', 'reported fact': 'Source-backed view',
                 'interpretation': 'Interpretation'}.get(str(c.get('basis', '')).replace('_', ' '), 'Source-backed view')
        return '<div class="claim">' + ('<h3>' + esc(label) + '</h3>' if label else '') + '<p>' + esc(c['statement']) + '<sup>' + ','.join(f'<a href="#source-{n}">{n}</a>' for n in ids) + '</sup></p><span class="basis">' + basis + '</span></div>'

    def group(prefix, limit=2):
        return ''.join(claim(c) for _, c in select(prefix, limit))

    def gap(text):
        return '<p class="gap">' + esc(text) + '</p>'

    def panel(n, title, content, cls=''):
        return f'<section class="panel {cls}"><h2><span>{n:02}</span>{esc(title)}</h2>{content}</section>'

    one_liner = get('/summary/one_liner')
    headline = get('/infographic/headline')
    headline_text = headline['statement'] if headline else 'The investment case, at a glance'
    headline_text = re.sub(r'^' + re.escape(run['ticker']) + r'\s*:\s*', '', headline_text)
    if headline:
        used.update(refs[e['sourceId']] for e in headline.get('evidence', []) if e.get('matched') and e.get('sourceId') in refs)
    company = claim(one_liner) or gap('Company overview requires further source review.')
    business = group('/business/revenue_engine', 1)
    if not business:
        names = select('/business/segments/', 20)
        names = [c for p, c in names if p.endswith('/name')][:4]
        business = ('<div class="segments">' + ''.join('<span>' + claim(n) + '</span>' for n in names) + '</div>' if names else '')
        business += gap('Segments discussed in this source pack. Revenue mix and business economics are not established here.')
    improvements = group('/financials/improved/', 2) or group('/monitor/strengtheners/', 2)
    pressures = group('/financials/weakened/', 2) or group('/monitor/breakers/', 2)
    signals = '<div class="split"><div class="positive"><h3 class="eyebrow">Recovery signals</h3>' + (improvements or gap('Not established.')) + '</div><div class="negative"><h3 class="eyebrow">Pressure points</h3>' + (pressures or gap('Not established.')) + '</div></div>'
    thesis = group('/summary/investment_thesis/', 2) or gap('Investment thesis requires further source review.')
    # Pair competing evidence within the same debate; never take the first unrelated leaves.
    debates = []
    for i in range(10):
        p = f'/debates/{i}/'
        left = get(p + 'evidence_for_market') or get(p + 'market_view')
        right = get(p + 'evidence_for_alternative') or get(p + 'alternative_view')
        if left and right:
            title = get(p + 'topic')
            content = ('<h3>' + esc(title['statement']) + '</h3>' if title else '')
            content += '<div class="debate"><div><b class="eyebrow">Recovery argument</b>' + claim(left) + '</div><div><b class="eyebrow">Counterpoint</b>' + claim(right) + '</div></div>'
            debates.append(content)
    debate = ''.join(debates[:1]) or gap('A complete, source-supported pair of competing views is unavailable.')
    catalysts = []
    for i in range(12):
        c = get(f'/catalysts/{i}/catalyst')
        if c:
            catalysts.append(claim(c) + claim(get(f'/catalysts/{i}/impact')))
    catalyst = ''.join(catalysts[:1]) or gap('No complete catalyst entry is supported.')
    # Require identity AND company value before displaying a valuation metric.
    valuations = []
    for i in range(12):
        p = f'/valuation/metrics/{i}/'
        name = get(p + 'metric') or get(p + 'name')
        value = get(p + 'current') or get(p + 'value')
        if name and value:
            valuations.append(claim(value, name['statement']))
    valuation = ''.join(valuations[:2]) or gap('Current company valuation and comparable historical multiples are not established in this evidence pack. No price target or fair-value conclusion is presented.')
    monitor = group('/monitor/breakers/', 2) or group('/risks/', 1) or gap('Monitoring evidence requires further review.')
    # Only a complete named scenario is usable; never print an unlabelled implied price.
    scenarios = []
    for i in range(3):
        p = f'/scenarios/{i}/'
        name = get(p + 'name') or get(p + 'case')
        assumptions = get(p + 'assumptions') or get(p + 'description')
        if name and assumptions:
            scenarios.append(claim(assumptions, name['statement']))
    scenario = ''.join(scenarios[:1]) or gap('Bull / base / bear assumptions and comparable multi-year financials require additional evidence. The selected event window is not a complete company model.')
    bottom = claim(get('/summary/earnings_direction')) or thesis
    rows = (panel(1, 'Company & business', company + business, 'company')
            + panel(2, 'The ownership case', thesis, 'thesis')
            + panel(3, 'Financial signals', signals, 'signals')
            + panel(4, 'What remains contested', debate, 'debates')
            + panel(5, 'Next catalyst', catalyst, 'catalyst')
            + panel(6, 'What to monitor', monitor, 'monitor')
            + panel(7, 'Valuation discipline', valuation, 'valuation')
            + panel(8, 'Scenario & history gaps', scenario, 'scenarios'))
    partial = has_omissions(sources)
    supported_count = sum(valid(c) for c in citations.values())
    total = sum(not missing_value(v) for v in report_fields(run.get('state',{}).get('report',{})).values())
    source_rows = []
    short_sources = []
    for i, s in enumerate(sources, 1):
        filename = s.get('filename', 'Original')
        clean = re.sub(r'--[0-9a-f]+(?=\.)', '', filename)
        clean = re.sub(r'\.(pdf|html|txt)$', '', clean, flags=re.I).replace('_', ' ')
        url = s.get('sourceUrl', '')
        link = '<a href="' + esc(url) + '">Open original source</a>' if urlparse(url).scheme in ('https', 'http') else 'Source URL unavailable'
        source_rows.append(f'<li id="source-{i}"><b>[{i}] {esc(clean)}</b> · {link}<p>Original SHA-256: {esc(s.get("originalHash"))}</p>' + ('<p>' + esc(s['coverage'].get('limitation')) + '</p>' if s.get('coverage') else '') + '</li>')
        short_sources.append(f'[{i}] ' + clean.replace(run['ticker'] + '-', '').replace('-', ' '))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(run['ticker'])} · Investment brief</title><style>
@page{{size:A4 portrait;margin:10mm}}*{{box-sizing:border-box;color:#000;font-family:Calibri,Carlito,sans-serif}}body{{margin:0;background:#e9edeb;font-size:14px;line-height:1.28}}.sheet{{width:920px;max-width:100%;margin:24px auto;background:#fff;padding:30px 34px 24px;box-shadow:0 6px 35px #00000012}}.masthead{{display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #000;padding-bottom:9px;font-size:10px;font-weight:700;letter-spacing:1.4px}}.identity{{display:flex;align-items:baseline;gap:15px;margin-top:14px}}.ticker{{font-size:49px;line-height:1;font-weight:700;letter-spacing:-2px}}.identity p{{font-size:11px;margin:0}}h1{{font-size:29px;line-height:1.05;letter-spacing:-.6px;margin:10px 0 12px;max-width:770px}}.coverage{{display:flex;gap:0;background:#edf2ef;border-top:1px solid #c2cec6;border-bottom:1px solid #c2cec6;margin-bottom:15px}}.coverage div{{flex:1;padding:8px 12px;border-right:1px solid #c2cec6;font-size:10px}}.coverage div:last-child{{border:0}}.coverage b{{display:block;font-size:16px;margin-bottom:2px}}.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px 18px}}.panel{{min-width:0;border-top:1px solid #000;padding-top:7px;break-inside:avoid}}h2{{font-size:15px;margin:0 0 7px;display:flex;align-items:baseline;gap:8px}}h2 span{{font-size:10px;border:1px solid #000;padding:2px 4px;font-weight:400}}h3{{font-size:12px;margin:0 0 5px}}p{{margin:0 0 6px}}.claim{{margin:0 0 8px}}.claim:last-child{{margin-bottom:0}}.basis{{font-size:9px;display:block;margin-top:-2px}}sup{{font-size:9px;margin-left:3px;font-weight:700}}a{{color:#000}}.segments{{display:flex;gap:5px;flex-wrap:wrap;margin:9px 0 6px}}.segments>span{{background:#edf2ef;padding:4px 8px;font-size:11px;font-weight:700;border-left:3px solid #aebfaf}}.gap{{font-size:11px;line-height:1.35;background:#f4f3ee;border-left:3px solid #d1c7ad;padding:8px 10px}}.signals{{grid-column:1 / span 2}}.split,.debate{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}}.positive,.negative{{padding:10px 12px}}.positive{{background:#edf4ef}}.negative{{background:#f7efea}}.eyebrow{{display:block;font-size:9px;letter-spacing:.9px;text-transform:uppercase;margin-bottom:6px}}.debates{{grid-column:1 / span 2}}.debate>div{{padding:9px 12px;background:#f1f3f5;border-left:3px solid #b8c4cf}}.company,.thesis{{grid-column:auto}}.company{{grid-column:1}}.signals{{grid-row:2}}.debates{{grid-row:3}}.catalyst{{grid-column:1;grid-row:4}}.monitor{{grid-column:2;grid-row:4}}.valuation{{grid-column:1;grid-row:5}}.scenarios{{grid-column:2;grid-row:5}}.thesis{{grid-column:2;grid-row:1}}.bottom{{margin-top:13px;padding:11px 13px;background:#e7eee9;border-top:2px solid #000}}.bottom h2{{margin-bottom:5px}}.bottom p{{font-weight:700}}.footnote{{font-size:9px;line-height:1.3;margin:10px 0 0}}.sources{{font-size:8px;line-height:1.35;margin:7px 0 0;border-top:1px solid #c4c9c6;padding-top:6px}}details{{max-width:920px;margin:16px auto;padding:18px;background:white;font-size:12px}}details p{{overflow-wrap:anywhere}}summary{{cursor:pointer;font-weight:700}}li{{margin:12px 0}}.screen-note{{font-size:11px;margin:0 0 12px}}@media(max-width:650px){{.sheet{{margin:0;padding:22px 18px;box-shadow:none}}.grid,.split,.debate{{display:block}}.panel{{margin-bottom:16px}}.split>div,.debate>div{{margin-bottom:8px}}.coverage{{flex-wrap:wrap}}.coverage div{{min-width:50%}}h1{{font-size:27px}}.ticker{{font-size:42px}}details{{margin:0}}}}@media print{{body{{background:white;font-size:9pt;line-height:1.2}}.sheet{{width:auto;max-width:none;margin:0;padding:0;box-shadow:none}}.masthead{{font-size:7pt;padding-bottom:5px}}.identity{{margin-top:8px}}.ticker{{font-size:30pt}}.identity p{{font-size:7pt}}h1{{font-size:18pt;margin:6px 0 9px}}.coverage{{margin-bottom:9px}}.coverage div{{padding:5px 8px;font-size:6.5pt}}.coverage b{{font-size:10pt}}.grid{{gap:7px 12px}}h2{{font-size:10pt;margin-bottom:5px}}h2 span{{font-size:7pt}}.panel{{padding-top:5px}}.claim{{margin-bottom:4px}}.basis,.eyebrow,sup{{font-size:6pt}}.positive,.negative,.debate>div{{padding:7px 9px}}.gap{{font-size:7.5pt;padding:6px 8px}}.segments>span{{font-size:7pt}}.bottom{{padding:8px 10px;margin-top:9px}}.footnote{{font-size:6.5pt;margin-top:7px}}.sources{{font-size:6pt}}details,.screen-note{{display:none}}}}
</style></head><body><article class="sheet" data-render-version="{VERSION}"><div class="masthead"><span>CHARLIE / EQUITY RESEARCH</span><span>INVESTMENT BRIEF · REVIEW DRAFT</span></div><div class="identity"><span class="ticker">{esc(run['ticker'])}</span><p>{esc(assignment.get('since'))} — {esc(assignment.get('until'))}<br>Source window · {esc(assignment.get('horizon','12–24 months'))} investment horizon</p></div><h1>{esc(headline_text)}</h1><div class="coverage"><div><b>{len(sources)} originals</b>Selected research pack</div><div><b>{supported_count} / {total} populated fields</b>Passage-matched &amp; model-supported</div><div><b>{'Partial coverage' if partial else 'Selected coverage'}</b>{'Two-sided excerpts; omitted text not reviewed' if partial else 'Not an exhaustive company review'}</div><div><b>Investor review</b>No thesis changes applied</div></div><main class="grid">{rows}</main><section class="bottom"><h2><span>09</span>Investment takeaway</h2>{bottom}</section><p class="footnote">Source-backed draft, not independently verified. Estimates retain their source dates; the collection window is not a financial period. Unsupported fields are omitted. Read original passages before relying on the conclusions.</p><p class="sources">{' · '.join(esc(s) for s in short_sources)}</p></article><details><summary>Open source register &amp; coverage notes</summary><p>Report {esc(run['id'])} · Renderer {VERSION}. The saved research and original exports remain unchanged.</p><ol>{''.join(source_rows)}</ol></details></body></html>'''
