"""Illustrated investment infographic. Structured facts and reviewed claims stay separate."""
import html
import json
import math
from urllib.parse import urlparse
VERSION='stock-infographic-v3'
e=lambda x:html.escape(str(x if x is not None else ''),quote=True)
COLORS=['#cdeaff','#d9f0df','#eee0fb','#ffe4c6','#d7eced']

def icon(kind):
    shapes={
     'company':'<path d="M8 46V14l18-8v40M26 18h24v28M4 46h52M14 18v4m6-7v4m-6 9v4m6-7v4m-6 9v4m6-7v4M32 25h5m7 0h2m-14 8h5m7 0h2m-8 13V38"/>',
     'money':'<ellipse cx="28" cy="13" rx="19" ry="7"/><path d="M9 13v10c0 9 38 9 38 0V13M9 24v10c0 9 38 9 38 0V24M9 35v9c0 9 38 9 38 0V35"/>',
     'chart':'<path d="M6 48h46M12 43V29h7v14M27 43V19h7v24M42 43V8h7v35M8 21l15-8 12 1 13-9"/>',
     'target':'<circle cx="28" cy="29" r="21"/><circle cx="28" cy="29" r="13"/><circle cx="28" cy="29" r="4"/><path d="M28 29 50 7M41 7h9v9"/>',
     'risk':'<path d="m28 5 25 45H3ZM28 20v14M28 40v2"/>',
     'clock':'<circle cx="28" cy="28" r="22"/><path d="M28 12v17l12 7"/>',
     'balance':'<path d="M28 6v43M17 49h22M8 16h40M12 16 4 34h16ZM44 16 36 34h16Z"/>',
     'monitor':'<path d="M4 9h49v32H4ZM18 50h21M28 41v9M8 28h10l6-12 9 19 6-9h10"/>',
     'care':'<path d="M9 49V15h38v34M3 49h50M22 49V36h12v13M28 6v18M19 15h18M16 27h4m16 0h4"/>',
    }
    return '<svg class="icon" viewBox="0 0 58 58" aria-hidden="true" fill="none" stroke="#24577a" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'+shapes.get(kind,shapes['chart'])+'</svg>'

def campus():
    # Conceptual architecture, never a depiction of actual company premises.
    windows=''.join(f'<rect x="{x}" y="{y}" width="13" height="18" fill="#91bdd4" stroke="#fff"/>' for x in range(97,320,23) for y in range(50,129,27))
    return '<svg class="campus" viewBox="0 0 450 180" aria-label="Conceptual company illustration"><defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#e5f5ff"/><stop offset="1" stop-color="#fff"/></linearGradient></defs><rect width="450" height="180" fill="url(#sky)"/><path d="M0 148 210 132 450 144v36H0" fill="#dbeccf"/><path d="m75 143 40-119 235 5 34 114" fill="#a5bcc8"/><path d="M87 145V37h253v108" fill="#dae5eb"/>'+windows+'<path d="m74 37 41-15 237 6-12 10Z" fill="#7998a9"/><path d="M219 146V95h57v51" fill="#547e95"/><path d="M224 98h46v47h-46M247 98v47" fill="#bde0ec" stroke="#fff"/><path d="m210 146-45 34h145l-29-34" fill="#e9e4d8"/>'+''.join(f'<path d="M{x} 153v-36" stroke="#847051" stroke-width="5"/><circle cx="{x}" cy="111" r="22" fill="#a1c98c"/><circle cx="{x-9}" cy="102" r="17" fill="#bddba8"/>' for x in [43,65,377,409])+'</svg>'

def charts(rows):
    parts=[]
    for k,title,div,unit in [('revenue','Revenue',1e9,'$bn'),('operatingMargin','Operating margin',1,'%'),('eps','GAAP diluted EPS',1,'$'),('freeCashFlow','Free cash flow¹',1e9,'$bn'),('netIncome','GAAP net income',1e9,'$bn')]:
        pts=[(r['year'],r[k]['value']/div) for r in rows if k in r]
        if len(pts)<3:
            parts.append('<div class="chart"><h3>'+title+'</h3><p>Comparable history unavailable</p></div>');continue
        low=min(0,min(v for _,v in pts));high=max(0,max(v for _,v in pts));span=high-low or 1
        baseline=116-(0-low)/span*78;svg=f'<line x1="5" y1="{baseline}" x2="190" y2="{baseline}" stroke="#879dab"/>'
        for j,(yr,v) in enumerate(pts):
            x=10+j*37;vy=116-(v-low)/span*78;h=abs(vy-baseline)
            svg+=f'<rect x="{x}" y="{min(vy,baseline):.2f}" width="25" height="{h:.2f}" rx="2" fill="{COLORS[j%5]}" stroke="#7facc6"/><text x="{x+12}" y="{vy-5 if v>=0 else vy+12}" text-anchor="middle" font-size="10">{v:.1f}</text><text x="{x+12}" y="137" text-anchor="middle" font-size="10">{e(yr)}</text>'
        parts.append(f'<figure class="chart"><h3>{title} <small>({unit})</small></h3><svg role="img" aria-label="{title}: '+e('; '.join(f'{yr}: {v:.1f} {unit}' for yr,v in pts))+f'" viewBox="0 0 200 145">{svg}</svg></figure>')
    return ''.join(parts)

def render(run,assignment,fundamentals):
    f=fundamentals;rows=f['annuals'];q=f.get('quote') or {};sources=run.get('sources',[])
    refs={s['id']:str(i+1) for i,s in enumerate(sources)};cs=run.get('state',{}).get('citations',{})
    def get(p):
        c=cs.get(p)
        return c if c and c.get('review')=='supported' and c.get('passageMatched') and any(x.get('matched') and x.get('sourceId') in refs for x in c.get('evidence',[])) else None
    def claim(c):
        if not c:return ''
        numbers=sorted({refs[x['sourceId']] for x in c.get('evidence',[]) if x.get('matched') and x.get('sourceId') in refs})
        basis={'broker_estimate':'Broker estimate · ','broker estimate':'Broker estimate · ','interpretation':'Interpretation · '}.get(c.get('basis'),'')
        return '<p>'+('<small>'+basis+'</small>' if basis else '')+e(c['statement'])+'<sup>['+','.join(numbers)+']</sup></p>'
    def many(prefix,n=2):return ''.join(claim(get(p)) for p in [p for p in sorted(cs) if p.startswith(prefix) and get(p)][:n])
    def section(n,title,content,cls=''):
        return f'<section class="panel {cls}"><header><b>{n}</b><h2>{title}</h2></header>{content}</section>'
    def stat(label,value,kind='chart',sub=''):
        return '<div class="stat">'+icon(kind)+'<div><small>'+e(label)+'</small><strong>'+e(value)+'</strong><small>'+e(sub)+'</small></div></div>'
    latest=rows[-1];money=lambda k:('$'+format(latest[k]['value']/1e9,'.1f')+'bn') if k in latest else 'Not established'
    company=claim(get('/summary/one_liner'))
    hero='<div class="hero"><div><h1>'+e(f.get('company') or run['ticker'])+'</h1><p class="ticker">'+e(run['ticker'])+' · Company overview → investment thesis</p>'+company+'</div><div>'+campus()+'<small>Conceptual illustration</small></div><div class="facts">'+stat('Sector',q.get('sector','Not established'),'company')+stat('Market capitalisation',('$'+format(q['marketCap']/1e9,'.1f')+'bn') if q.get('marketCap') else 'Not established','money')+stat('Investment horizon',assignment.get('horizon','12–24 months'),'clock')+'</div></div>'
    # Business identity from the reviewed pack; economic flow is explicitly consolidated.
    segs=[]
    for i in range(8):
        p=f'/business/segments/{i}/';c=get(p+'name')
        if c:segs.append('<div class="business-card">'+icon('care')+'<h3>'+e(c['statement'])+'</h3>'+claim(get(p+'description') or get(p+'key_kpi'))+'</div>')
    if f.get('segments'):
        segs=['<div class="business-card">'+icon('company')+'<h3>'+e(s['name'])+'</h3><strong>$'+format(s['revenue']/1e9,'.1f')+'bn</strong><small>'+e(s['year'])+' segment revenue [F]</small></div>' for s in f['segments']]
    business='<div class="business">'+''.join(segs[:4])+'<div class="flow-arrow">→</div>'+stat('Consolidated revenue',money('revenue'),'money',latest['year'])+'<div class="flow-arrow">→</div>'+stat('Operating earnings',money('operatingIncome'),'chart',latest['year'])+'<div class="flow-arrow">→</div>'+stat('Net income',money('netIncome'),'money',latest['year'])+'</div><small>Segment amounts may include internal sales; arrows summarize group profitability, not additive segment contributions. Consolidated GAAP figures [F]. Intersegment revenue must not be added to group sales.</small>'
    history='<div class="charts">'+charts(rows)+'</div><div class="signal-strip"><div><b>Recovery signals</b>'+many('/financials/improved/',1)+'</div><div><b>Pressure points</b>'+many('/financials/weakened/',1)+'</div></div><small>[F] Annual SEC XBRL facts; latest filed values at retrieval. GAAP definitions. ¹ Free cash flow = operating cash flow − capital expenditure; margin = operating earnings ÷ revenue. These are calculations, not separately reported GAAP metrics.</small>'
    thesis=many('/summary/investment_thesis/',2)
    scenarios=[]
    for title,prefix,color in [('Upside conditions','/monitor/strengtheners/',0),('Current evidence','/summary/earnings_direction',1),('Downside conditions','/monitor/breakers/',2)]:
        content=many(prefix,1)
        scenarios.append('<div class="scenario s'+str(color)+'"><h3>'+e(title)+'</h3>'+ (content or '<p>Further evidence required</p>')+'</div>')
    thesis+='<div class="scenarios">'+''.join(scenarios)+'</div><small>Conditional evidence map, not a calibrated bull/base/bear valuation model.</small>'
    debates=''
    for i in range(10):
        p=f'/debates/{i}/';a=get(p+'market_view') or get(p+'evidence_for_market');b=get(p+'alternative_view') or get(p+'evidence_for_alternative')
        if a and b:debates+='<tr><td>'+claim(a)+'</td><td>'+claim(b)+'</td></tr>'
        if debates:break
    debate='<table><thead><tr><th>Market / recovery argument</th><th>What could be different</th></tr></thead><tbody>'+debates+'</tbody></table>' if debates else '<p>Paired debate requires further evidence.</p>'
    catalysts='<div class="risk-grid"><div><h3>'+icon('clock')+' Catalysts</h3>'+many('/catalysts/',2)+'</div><div><h3>'+icon('risk')+' Risks</h3>'+many('/financials/weakened/',2)+'</div></div>'
    valuation='<div class="valuation">'+stat('Market price',f"${q['price']:.2f}" if q.get('price') else 'Unavailable','money')+stat('Trailing P/E',f"{q['trailingPE']:.1f}×" if q.get('trailingPE') else 'Unavailable','balance')+stat('Forward P/E',f"{q['forwardPE']:.1f}×" if q.get('forwardPE') else 'Unavailable','chart')+'</div><small>[M] '+e(q.get('provider','Market data unavailable'))+' · '+e(q.get('observedAt',''))+'. Multiples use provider definitions; forward estimates are not independently reconstructed. Peer comparison and scenario price targets require matched evidence; no relative-value conclusion is implied.</small>'
    monitor='<div class="monitor-grid"><div><h3>'+icon('target')+' Thesis strengthens</h3>'+many('/monitor/strengtheners/',2)+'</div><div><h3>'+icon('risk')+' Thesis weakens</h3>'+many('/monitor/breakers/',2)+'</div><div><h3>'+icon('monitor')+' Next checkpoint</h3>'+many('/catalysts/',1)+'</div></div>'
    bottom=claim(get('/summary/earnings_direction'))
    foot=' · '.join('['+str(i+1)+'] '+e(s.get('filename','Original')) for i,s in enumerate(sources))
    details='<p>[F] <a href="'+e(f['sourceUrl'])+'">SEC company facts</a> · Retrieved '+e(f['retrievedAt'])+' · SHA-256 '+e(f['sourceHash'])+'</p>'
    if f.get('segmentSource'):details+='<p>Segment filing: <a href="'+e(f['segmentSource']['url'])+'">Original 10-K</a> · SHA-256 '+e(f['segmentSource']['sha256'])+'</p>'
    if q:details+='<p>[M] <a href="'+e(q['sourceUrl'])+'">Yahoo Finance quote</a> · Retrieved '+e(q['retrievedAt'])+'</p>'
    for i,s in enumerate(sources,1):
        url=s.get('sourceUrl','');link='<a href="'+e(url)+'">Open original</a>' if urlparse(url).scheme in ('http','https') else 'URL unavailable'
        details+='<p>['+str(i)+'] '+e(s.get('filename'))+' · '+link+'</p>'
    details+='<pre>'+e(json.dumps({'annuals':rows,'quote':q},indent=2))+'</pre>'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+e(run['ticker'])+''' · Visual investment map</title><style>
@page{size:A3 portrait;margin:9mm}*{box-sizing:border-box;font-family:Calibri,Carlito,sans-serif;color:#000}body{margin:0;background:#e8eff4;font-size:13px;line-height:1.22}.poster{max-width:1180px;margin:20px auto;padding:18px;background:white}.top{display:flex;justify-content:space-between;font-size:10px;letter-spacing:1px;border-bottom:2px solid #85bddb;padding-bottom:6px}.panel{border:1px solid #c8dfee;border-radius:7px;margin-top:8px;padding:9px;break-inside:avoid}header{display:flex;align-items:center;gap:8px;background:linear-gradient(90deg,#d6eaf8,#f5fbff);margin:-9px -9px 8px;padding:5px 8px;border-radius:7px 7px 0 0}header b{width:25px;height:25px;display:grid;place-items:center;border:1px solid #71a6c7;border-radius:50%;background:#bddcef;font-size:18px}h1{font-size:33px;line-height:1.02;margin:3px 0 7px;letter-spacing:-.6px}h2{font-size:20px;margin:0}h3{font-size:14px;margin:0 0 6px}p{margin:0 0 6px}small{display:block;font-size:9px;line-height:1.2}sup{font-size:8px;margin-left:2px}.hero{display:grid;grid-template-columns:1.1fr .9fr .65fr;gap:12px;align-items:center}.ticker{font-size:15px;font-weight:700;margin:9px 0}.campus{width:100%;display:block}.icon{width:34px;height:34px;flex-shrink:0}.stat{display:flex;align-items:center;gap:9px;padding:8px;background:#edf6fc;border:1px solid #c6dfed;border-radius:5px}.stat strong{display:block;font-size:20px;line-height:1.1;margin:3px 0}.facts{display:grid;gap:5px}.facts .stat:nth-child(2){background:#e5f3e9}.business{display:flex;gap:7px;align-items:stretch}.business .stat,.business-card{flex:1;min-width:0}.business-card{padding:7px;background:#f3effa;border:1px solid #ded1eb;border-radius:5px}.business-card p{font-size:11px}.business .stat{display:block}.flow-arrow{align-self:center;font-size:23px}.charts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}.chart{margin:0;padding:6px;border:1px solid #d4e6ef;border-radius:4px;background:#fbfdff}.chart svg{width:100%;display:block}.chart small{display:inline}.signal-strip{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:7px 0}.signal-strip>div{padding:7px;background:#e8f4e9}.signal-strip>div+div{background:#fbeeee}.signal-strip p{font-size:11px;margin:3px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}.two .panel{min-width:0}.scenarios{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:8px 0}.scenario{padding:8px;background:#e3f2e7;border-radius:5px;border-top:4px solid #8dc6a2}.scenario p{font-size:11px}.s1{background:#eaf3fb;border-color:#8abbdc}.s2{background:#faeeee;border-color:#daa3a3}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;vertical-align:top;padding:8px;border:1px solid #d4e2eb;font-size:12px}th{background:#e0effa}.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.risk-grid>div{background:#edf6ef;padding:8px}.risk-grid>div+div{background:#fcefee}h3 .icon{width:23px;height:23px;vertical-align:middle}.valuation{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-bottom:9px}.valuation .stat{display:block}.monitor-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.monitor-grid>div{padding:8px;background:#eef5f7}.monitor-grid>div:nth-child(2){background:#faeeee}.bottom{background:#e8f2ed;font-size:16px;font-weight:700}.footer{margin-top:8px;font-size:8px;line-height:1.3}.status{padding:5px 8px;background:#fff3d7;font-size:10px;margin:8px 0 0}details{max-width:1180px;margin:10px auto;padding:18px;background:white;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}@media print{body{background:white;font-size:9pt;line-height:1.18}.poster{margin:0;padding:0;max-width:none}h1{font-size:22pt}h2{font-size:14pt}.panel{margin-top:5px;padding:7px}header{margin:-7px -7px 6px;padding:3px 6px}.campus{max-height:125px}.facts .stat{padding:5px}.chart svg{max-height:112px}.top{font-size:7pt}small{font-size:7pt}.stat strong{font-size:15pt}.chart h3{font-size:10pt}.scenario p,.business-card p,.signal-strip p{font-size:8pt}th,td{font-size:9pt}.footer{font-size:6pt}details{display:none}}@media(max-width:650px){.poster{margin:0;padding:10px}.hero,.two,.scenarios,.monitor-grid{display:block}.hero>div,.scenario,.monitor-grid>div{margin-bottom:9px}.campus{max-height:150px}.facts{grid-template-columns:1fr 1fr}.business{flex-wrap:wrap}.business-card,.business .stat{min-width:40%}.flow-arrow{display:none}.charts{grid-template-columns:repeat(2,minmax(0,1fr))}.charts .chart:last-child{grid-column:1 / span 2}.signal-strip,.risk-grid{display:block}.valuation .stat{padding:5px}.valuation .stat strong{font-size:17px}.top{font-size:8px}h1{font-size:30px}.bottom{font-size:14px}}
</style><body><article class="poster" data-render-version="'''+VERSION+'''"><div class="top"><b>CHARLIE / VISUAL STOCK RESEARCH</b><span>INVESTOR REVIEW DRAFT</span></div>'''+section(1,'Company at a glance',hero)+section(2,'How the business makes money',business)+section(3,'Five-year financial snapshot',history)+'<div class="two">'+section(4,'Investment thesis',thesis)+section(5,'What the market is debating',debate)+'</div><div class="two">'+section(6,'Key catalysts and risks',catalysts)+section(7,'Valuation and market context',valuation)+'</div>'+section(8,'What to monitor',monitor)+section(9,'Bottom line',bottom,'bottom')+'<p class="status">Evidence coverage: historical filings and dated market data supplement the event window '+e(assignment.get('since'))+'–'+e(assignment.get('until'))+'. Peer comparisons and calibrated scenario prices remain incomplete. No thesis changes applied.</p><p class="footer">'+foot+' · [F] SEC annual facts · [M] Yahoo Finance. Some research originals were excerpted; omitted text was not reviewed. Illustration is conceptual. Research claims retain model-review limitations; public numerical facts are mechanically mapped and calculations are shown in the source register.</p></article><details><summary>Sources, dates, definitions and calculation inputs</summary>'+details+'</details></body></html>'
