"""Source-preserving editorial investment map. No model calls or input mutation."""
import base64
import html
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from visual_business_profiles import PROFILES

VERSION='stock-infographic-v5'
e=lambda value:html.escape(str(value if value is not None else ''),quote=True)

@lru_cache(maxsize=3)
def industry_art(kind):
    files={'healthcare':'healthcare-value-chain.png','equipment':'equipment-value-chain.png','medical-technology':'medical-technology-value-chain.png'}
    path=Path(__file__).parent/'assets/research-visuals'/files[kind]
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()

def numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def sentence_key(text):
    return set(re.findall(r'[a-z0-9]+',text.lower()))-{'the','a','and','of','to','in','with','for','by','is','at'}

class Editor:
    """Rank supported evidence by relevance and strength, then suppress repeated ideas."""
    def __init__(self,run):
        self.sources=run.get('sources',[])
        self.refs={s['id']:i+1 for i,s in enumerate(self.sources)}
        self.cs={p:c for p,c in run.get('state',{}).get('citations',{}).items() if self.valid(c)}
        self.used=[];self.used_refs=set()
    def valid(self,c):
        return bool(c and c.get('statement') and c.get('review')=='supported' and c.get('passageMatched') and any(x.get('matched') and x.get('sourceId') in self.refs for x in c.get('evidence',[])))
    def get(self,path):return self.cs.get(path)
    def fresh(self,c):
        k=sentence_key(c['statement'])
        return all(len(k & old)/max(1,len(k | old))<.62 for old in self.used)
    def score(self,item):
        p,c=item;t=c['statement']
        sources=len({x.get('sourceId') for x in c.get('evidence',[]) if x.get('matched')})
        return (sources*3+(4 if topic(c)!='Investment debate' and topic(c)==topic(self.cs.get('/infographic/headline',{'statement':''})) else 0)+min(3,len(re.findall(r'\d',t)))+(2 if c.get('basis') in ('reported_fact','reported fact') else 0),-len(t),p)
    def select(self,prefixes,limit=2,exclude=()):
        items=[x for x in self.cs.items() if any(x[0].startswith(p) for p in prefixes) and not any(x[0].endswith(z) for z in exclude)]
        selected=[]
        for _,c in sorted(items,key=self.score,reverse=True):
            if not self.fresh(c):continue
            if any(topic(c)==topic(prior) and topic(c)!='Investment debate' for prior in selected):continue
            selected.append(c);self.used.append(sentence_key(c['statement']))
            if len(selected)>=limit:break
        return selected
    def refs_html(self,c):
        ids=sorted({self.refs[x['sourceId']] for x in c.get('evidence',[]) if x.get('matched') and x.get('sourceId') in self.refs})
        self.used_refs.update(ids)
        return '<sup>'+','.join(f'<a href="#source-{n}">{n}</a>' for n in ids)+'</sup>'
    def claim(self,c,remember=False):
        if not c:return '<span class="missing">Evidence needed</span>'
        if remember:self.used.append(sentence_key(c['statement']))
        basis={'broker_estimate':'Estimate','broker estimate':'Estimate','interpretation':'Interpretation'}.get(c.get('basis'),'Source view')
        return '<p class="claim"><span class="basis">'+basis+'</span> '+e(c['statement'])+self.refs_html(c)+'</p>'

def metric_change(points,metric):
    if len(points)<2:return 'History unavailable'
    a,b=points[0][1],points[-1][1]
    if metric=='operatingMargin':return f'{b-a:+.1f} pp'
    if a>0:return f'{(b/a-1)*100:+.1f}%'
    return 'Change not comparable'

def charts(rows):
    parts=[]
    metrics=[('revenue','Revenue',1e9,'$bn','#57a7d6'),('operatingMargin','Operating margin',1,'%','#6eb59e'),('eps','Diluted EPS',1,'$','#aa93c9'),('freeCashFlow','Free cash flow',1e9,'$bn','#e3aa65'),('netIncome','Net income',1e9,'$bn','#79adb3')]
    for key,title,scale,unit,color in metrics:
        pts=[(r.get('year',''),r[key]['value']/scale) for r in rows if key in r and numeric(r[key].get('value'))]
        if len(pts)<3:
            parts.append(f'<figure class="chart"><h3>{title}</h3><p>Comparable history unavailable</p></figure>');continue
        pts=pts[-5:];low=min(0,min(v for _,v in pts));high=max(0,max(v for _,v in pts));span=high-low or 1
        y=lambda v:90-(v-low)/span*56
        bars=f'<path d="M4 {y(0):.1f}H196" stroke="#a0aab0"/>'
        for j,(year,value) in enumerate(pts):
            x=8+j*39
            bars+=f'<rect x="{x}" y="{min(y(value),y(0)):.1f}" width="26" height="{abs(y(value)-y(0)):.1f}" fill="{color}" opacity="{1 if j==len(pts)-1 else .45}"/><text x="{x+13}" y="{y(value)-5 if value>=0 else y(value)+11:.1f}">{value:.1f}</text><text x="{x+13}" y="111">{e(year)}</text>'
        title_change=metric_change(pts,key)
        parts.append(f'<figure class="chart"><h3>{title} <span>({unit})</span></h3><b class="change">{title_change}</b><small>{e(pts[0][0])}–{e(pts[-1][0])} total change</small><svg viewBox="0 0 204 118" role="img" aria-label="{e(title)}: {e(str(pts))}"><g font-family="Calibri" fill="#000" text-anchor="middle" font-size="12">{bars}</g></svg></figure>')
    return ''.join(parts)

def sensitivity(q):
    """Mechanical two-factor sensitivity; never label these cells as target prices."""
    pe=q.get('forwardPE');price=q.get('price')
    if not numeric(pe) or not numeric(price) or pe<=0 or price<=0:return '<p>Forward valuation inputs unavailable.</p>'
    implied=price/pe
    out='<table class="sensitivity"><thead><tr><th>EPS × P/E</th>'+''.join(f'<th>{pe*m:.1f}×</th>' for m in [.9,1,1.1])+'</tr></thead><tbody>'
    for a in [.9,1,1.1]:
        out+=f'<tr><th>${implied*a:.2f}</th>'
        for b in [.9,1,1.1]:
            value=price*a*b;out+=f'<td class="{"center" if a==b==1 else ""}">${value:.0f} <small>({(a*b-1)*100:+.0f}%)</small></td>'
        out+='</tr>'
    return out+'</tbody></table><p class="micro">±10% EPS / P/E; EPS = price ÷ forward P/E. Mechanical values, not forecasts. [M]</p>'

def topic(c):
    t=c['statement'].lower()
    for name,words in [('Star ratings',('star','cahps')),('Commercial pricing',('idr','commercial')),('Optum recovery',('optum','oh ')),('Medical costs',('utilization','medical')),('Medicaid',('medicaid',)),('Medicare margins',('ma margin','medicare advantage')),('Capital returns',('buyback','debt/'))]:
        if any(w in t for w in words):return name
    return 'Investment debate'

def section(n,title,body,sub=''):
    return f'<section class="panel"><header><span class="number">{n}</span><h2>{e(title)}</h2><span class="sub">{e(sub)}</span></header>{body}</section>'

def render(run,assignment,fundamentals):
    f=fundamentals;rows=f.get('annuals',[]);latest=rows[-1] if rows else {};q=f.get('quote') or {};ticker=run['ticker'];profile=PROFILES.get(ticker,{})
    ed=Editor(run);money=lambda value:'$'+format(value/1e9,'.1f')+'bn' if numeric(value) else 'Not available'
    val=lambda k:latest.get(k,{}).get('value')
    headline=ed.get('/infographic/headline')
    headline_text=re.sub(r'^'+re.escape(ticker)+r'\s*:\s*','',headline['statement']) if headline else 'The business, the evidence and the investment case'
    headline_refs=ed.refs_html(headline) if headline else ''
    if headline:ed.used.append(sentence_key(headline['statement']))
    overview=e(profile['overview'])+' <sup>[B]</sup>' if profile.get('overview') else ed.claim(ed.get('/summary/one_liner'))
    asof=e(f.get('asOf',''));window=e(assignment.get('since',''))+' — '+e(assignment.get('until',''))
    head='<div class="mast"><b>CHARLIE / INVESTMENT MAP</b><span>'+e(ticker)+' · '+asof+' · REVIEW DRAFT</span></div><div class="title"><h1>'+e(profile.get('name') or f.get('company') or ticker)+'</h1><div class="symbol">'+e(ticker)+'</div></div><h2 class="headline">'+e(headline_text)+headline_refs+'</h2>'
    facts='<div class="facts"><div><small>Market value [M]</small><b>'+money(q.get('marketCap'))+'</b></div><div><small>Share price [M]</small><b>'+('$'+format(q['price'],'.2f') if numeric(q.get('price')) else 'Unavailable')+'</b></div><div><small>Investment horizon</small><b>'+e(assignment.get('horizon','12–24 months'))+'</b></div></div>'
    company='<div class="overview"><p>'+overview+'</p>'+facts+'</div>'
    segments=f.get('segments',[])
    # Industry artwork is conceptual, never a company's actual products/facilities.
    art=profile.get('art')
    artwork='<img class="value-chain" src="'+industry_art(art)+'" alt="'+e(profile.get('industry','Industry'))+' conceptual illustration"/><span class="art-note">Conceptual industry illustration</span>' if art in ('healthcare','equipment','medical-technology') else ''
    cards=''
    for i,s in enumerate(segments[:4]):
        context=profile.get('segments',{}).get(s['name'])
        desc='<p>'+e(context[0])+'</p><small>'+e(context[1])+'</small><small><b>Watch:</b> '+e(context[2])+'</small>' if context else '<small>Reported operating segment</small>'
        cards+=f'<div class="segment c{i}"><h3>{e(s["name"])}</h3><b class="segment-value">{money(s["revenue"])}</b>{desc}</div>'
    if not cards:cards='<p>Segment economics need additional filing evidence.</p>'
    segment_scope=f' Showing {min(4,len(segments))} of {len(segments)} segments.' if len(segments)>4 else ''
    bridge='<div class="bridge"><div><small>Consolidated revenue</small><b>'+money(val('revenue'))+'</b></div><span>→</span><div><small>Operating earnings</small><b>'+money(val('operatingIncome'))+'</b></div><span>→</span><div><small>Net income</small><b>'+money(val('netIncome'))+'</b></div></div>'
    company='<div class="company-hero"><div>'+company+'</div><div>'+artwork+'</div></div>' if artwork else company
    business='<div class="segments" style="--segment-count:'+str(max(1,min(4,len(segments))))+'">'+cards+'</div>'+bridge+'<p class="micro">'+e(latest.get('year',''))+' reported figures [F].'+segment_scope+' Segment definitions may include internal sales; do not add to group revenue. '+(e(profile.get('basis',''))+' [B]' if profile else '')+'</p>'
    history='<div class="charts">'+charts(rows)+'</div><p class="micro">Reported GAAP inputs [F]. Calculated FCF = operating cash flow − capex; margin = operating income / sales. '+e(profile.get('financialNote',''))+'</p>'
    # Reserve whole paired arguments before selecting other sections. Never cross-pair topics.
    debates=[];monitor=[]
    for i in range(20):
        prefix=f'/debates/{i}/';a=ed.get(prefix+'market_view') or ed.get(prefix+'evidence_for_market');b=ed.get(prefix+'alternative_view') or ed.get(prefix+'evidence_for_alternative');resolution=ed.get(prefix+'resolution_signal')
        if a and b:
            debates.append((a,b));ed.used.extend([sentence_key(a['statement']),sentence_key(b['statement'])])
            if resolution:monitor.append((topic(a),resolution))
    debates=sorted(debates,key=lambda pair:-(ed.score(('',pair[0]))[0]+ed.score(('',pair[1]))[0]))[:3]
    debate='<table class="debates"><thead><tr><th>Market argument</th><th>Counterpoint</th></tr></thead><tbody>'+''.join('<tr><td><h3>'+e(topic(a))+'</h3>'+ed.claim(a)+'</td><td>'+ed.claim(b)+'</td></tr>' for a,b in debates)+'</tbody></table>' if debates else '<p>Paired source evidence needs review.</p>'
    thesis_claims=ed.select(['/summary/investment_thesis/','/summary/top_drivers/'],2)
    thesis='<div class="question"><b>Investment test</b><p>What must be true for the investment case to work?</p></div>'+(''.join(ed.claim(c) for c in thesis_claims) or '<p>Source-supported thesis evidence unavailable.</p>')

    catalysts=[]
    for i in range(15):
        prefix=f'/catalysts/{i}/';c=ed.get(prefix+'catalyst')
        if c and ed.fresh(c):
            impact=ed.get(prefix+'impact');timing=ed.get(prefix+'timing')
            catalysts.append(ed.claim(c,True)+(('<small>'+e(timing['statement'])+'</small>') if timing else ''))
    risks=ed.select(['/financials/weakened/','/industry/disruption_risks/','/monitor/breakers/'],2)
    triggers='<div class="triggers"><div class="positive"><h3>Upcoming checkpoints</h3>'+(''.join(catalysts[:2]) or '<p>No source-supported catalyst selected.</p>')+'</div><div class="negative"><h3>Downside exposures</h3>'+(''.join(ed.claim(c) for c in risks) or '<p>No source-supported risk selected.</p>')+'</div></div>'
    multiples=lambda quote,key:f'{quote[key]:.1f}×' if numeric(quote.get(key)) and quote[key]>0 else 'n/a'
    peers=[dict(ticker=ticker,name=profile.get('name') or f.get('company'),quote=q)]+f.get('peers',[])
    peer_table='<table class="peers"><thead><tr><th>Company</th><th>Mkt cap</th><th>Trailing P/E</th><th>Forward P/E</th></tr></thead><tbody>'
    for peer in peers:
        pq=peer['quote'];peer_table+='<tr><td><b>'+e(peer['ticker'])+'</b>'+'</td><td>'+money(pq.get('marketCap'))+'</td><td>'+multiples(pq,'trailingPE')+'</td><td>'+multiples(pq,'forwardPE')+'</td></tr>'
    peer_table+='</tbody></table><p class="micro">'+e(profile.get('peerBasis','Peer set not curated.'))+' Same-date provider multiples where supplied. [M]</p>'
    valuation='<div class="valuation"><div><h3>Market comparison</h3>'+peer_table+'</div><div><h3>Price sensitivity—not price targets</h3>'+sensitivity(q)+'</div></div>'
    monitor_rows=''.join('<tr><th>'+e(name)+'</th><td>'+ed.claim(c,True)+'</td></tr>' for name,c in monitor[:3])
    if not monitor_rows:
        monitor_rows=''.join('<tr><th>Evidence to watch</th><td>'+ed.claim(c)+'</td></tr>' for c in ed.select(['/monitor/strengtheners/','/monitor/breakers/'],2))
    watch='<table class="watch"><thead><tr><th>Decision area</th><th>Evidence that resolves the debate</th></tr></thead><tbody>'+(monitor_rows or '<tr><td colspan="2">Source-supported monitoring evidence unavailable.</td></tr>')+'</tbody></table>'
    bottom_claim=ed.get('/summary/earnings_direction')
    bottom=ed.claim(bottom_claim)
    glossary=profile.get('glossary','EPS = earnings per share · P/E = price / earnings · FCF = free cash flow')
    source_names=[]
    for n,s in enumerate(ed.sources,1):
        name=s.get('publisher') or s.get('filename','Source')
        if not s.get('publisher'):
            name=re.sub(r'--[a-f0-9]+(?=\.)','',name);name=re.sub(r'\.pdf$','',name);name=name.replace(ticker+'-','',1);name=re.sub(r'^\d{4}-','',name).replace('-',' ')
        source_names.append(f'<span>[{n}] '+e(name)+'</span>')
    footer='<div class="footer"><p><b>Dates:</b> Events '+window+' · Financials '+e(latest.get('year',''))+' · Quote '+e(q.get('observedAt','Unavailable')[:10])+'.</p><p>'+e(glossary)+'</p><p>'+ ' · '.join(source_names)+' · [F/B] SEC / company filings · [M] Yahoo Finance. Illustration is conceptual.</p></div>'
    details='<details><summary>Source register, dates and calculation inputs</summary>'
    def link(url,label):return '<a href="'+e(url)+'">'+e(label)+'</a>' if urlparse(url or '').scheme in ('https','http') else e(label)
    for n,s in enumerate(ed.sources,1):details+=f'<p id="source-{n}">[{n}] '+link(s.get('sourceUrl',''),s.get('filename','Original'))+'</p>'
    details+='<p>[F] '+link(f.get('sourceUrl',''),'SEC company facts')+' · SHA-256 '+e(f.get('sourceHash',''))+'</p>'
    if f.get('segmentSource'):details+='<p>'+link(f['segmentSource']['url'],'Segment filing')+' · SHA-256 '+e(f['segmentSource']['sha256'])+'</p>'
    if profile:details+='<p>[B] '+link(profile['source'],'Business description and dated segment structure')+'</p>'
    for peer in peers:details+='<p>[M] '+e(peer['ticker'])+' '+link(peer['quote'].get('sourceUrl',''),'Market snapshot')+' · '+e(peer['quote'].get('observedAt',''))+'</p>'
    details+='<p>AI source review does not replace investor review. Some originals were excerpted; omitted text was not reviewed. Scenario forecasts and calibrated price targets are not established by this page. Historical multiples are not supplied. Generated artwork is conceptual.</p><pre>'+e(json.dumps(f,indent=2))+'</pre></details>'
    body=head+section(1,'The company',company)+section(2,profile.get('businessTitle','How the business makes money'),business)+section(3,'Growth is only half the story',history,'Five-year financial record')+'<div class="research-grid">'+section(4,'The investment thesis',thesis)+section(5,'What the market is debating',debate)+'</div><div class="lower-grid">'+section(6,'Catalysts & risks',triggers)+section(7,'Valuation & peer context',valuation)+section(8,'What to monitor',watch)+'</div>'+section(9,'The investment view',bottom)+footer
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(ticker)+' · Investment map</title><style>'+CSS+'</style><body><main data-render-version="'+VERSION+'">'+body+'</main>'+details+'</body></html>'

CSS='''
@page{size:A4 portrait;margin:9mm}*{box-sizing:border-box;color:#000;font-family:Calibri,Carlito,sans-serif}body{margin:0;background:#edf1f3;font-size:13px;line-height:1.2}main{max-width:1000px;margin:20px auto;background:#fff;padding:24px 28px}.mast{display:flex;justify-content:space-between;font-size:10px;letter-spacing:1px;border-bottom:2px solid #75a9c7;padding-bottom:7px}.title{display:flex;align-items:center;justify-content:space-between;margin-top:9px}h1{font-size:34px;letter-spacing:-1px;line-height:1;margin:0}.symbol{font-size:25px;font-weight:700;border-bottom:4px solid #abd2e4}h2.headline{font-size:23px;line-height:1.06;font-weight:400;margin:8px 0 12px;max-width:90%}h2{font-size:16px;margin:0}h3{font-size:12px;margin:0 0 4px}p{margin:0 0 5px}small,.micro{font-size:10px;line-height:1.18}.micro{margin:4px 0 0}sup{font-size:8px;white-space:nowrap}a{color:#000}.panel{margin-top:9px;break-inside:avoid}header{display:flex;align-items:center;gap:7px;background:linear-gradient(90deg,#dfedf6,#f5f9fb);border-top:1px solid #c8dce8;padding:3px 6px;margin-bottom:6px}.number{display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;border:1px solid #7ea8c2;border-radius:50%;background:#c6dfee;width:22px;height:22px;flex-shrink:0}.sub{font-size:10px;margin-left:auto}.overview{display:grid;grid-template-columns:1fr 1fr;gap:15px;align-items:center}.overview>p{font-size:13px;margin:0}.facts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}.facts>div{background:#edf5f8;padding:7px}.facts b{display:block;font-size:17px}.facts small{display:block}.value-chain{width:100%;display:block;max-height:150px;object-fit:cover;object-position:center 48%}.art-note{display:block;font-size:9px;text-align:right;margin-top:-3px}.segments{display:grid;grid-template-columns:repeat(var(--segment-count,4),minmax(0,1fr));gap:5px;margin-top:3px}.segment{padding:6px 8px;background:#e9f4fa;border-top:3px solid #64a6c7}.segment.c1{background:#e9f3ee;border-color:#6eaa8d}.segment.c2{background:#f0eaf6;border-color:#a28bbb}.segment.c3{background:#fbf1e4;border-color:#d9aa68}.segment-value{font-size:18px;display:block;margin:3px 0}.segment p{font-size:12px}.segment small{display:block}.bridge{display:flex;align-items:center;justify-content:space-around;background:#f1f5f6;border-bottom:1px solid #c5d9e3;margin-top:5px;padding:5px}.bridge>div{display:flex;align-items:baseline;gap:7px}.bridge b{font-size:16px}.bridge>span{font-size:22px}.charts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px}.chart{margin:0;padding:5px 4px;border-right:1px solid #d6e1e8}.chart:last-child{border:0}.chart h3{font-size:12px}.chart h3 span{font-size:10px}.change{display:block;font-size:19px;line-height:1.05}.chart small{display:block;font-size:9px}.chart svg{width:100%;display:block}.research-grid{display:grid;grid-template-columns:.8fr 1.2fr;gap:10px}.research-grid>.panel{min-width:0}.question{background:#eaf3f7;border-left:3px solid #6d9fb7;padding:6px;margin-bottom:6px}.question p{margin-top:3px}.claim{margin:0 0 6px}.basis{font-weight:700;font-size:9px;letter-spacing:.2px;text-transform:uppercase}.thesis-note{background:#f6f2e6;padding:5px;font-size:10px}.triggers{display:grid;grid-template-columns:1fr 1fr;gap:8px}.triggers>div{padding:6px 8px}.positive{background:#ebf4ec}.negative{background:#faeeee}.valuation{display:grid;grid-template-columns:1.1fr 1fr;gap:10px}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px}th,td{padding:5px;vertical-align:top;text-align:left;border-bottom:1px solid #d7e2e9}th{background:#edf4f8;font-weight:700}td .claim:last-child{margin-bottom:0}.debates td{width:50%;font-size:12px}.peers th,.peers td{font-size:11px}.peers td small{display:block;font-size:9px}.peers tr:first-child td{background:#edf5f8}.sensitivity td,.sensitivity th{text-align:center;font-size:12px;padding:4px}.sensitivity small{display:block;font-size:9px}.sensitivity td.center{background:#dcefe4;font-weight:700}.watch th:first-child{width:23%}.watch .claim{margin:0}.footer{border-top:1px solid #bbcfdc;margin-top:8px;padding-top:5px;font-size:9px;line-height:1.18}.footer p{margin:0 0 3px}.missing{font-style:italic}details{max-width:1000px;margin:20px auto;background:white;padding:20px}details pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details p{overflow-wrap:anywhere}summary{cursor:pointer;font-weight:700}
@media print{body{background:white;font-size:9pt;line-height:1.14}main{margin:0;padding:0;max-width:none}.mast{font-size:7pt;padding-bottom:4px}.title{margin-top:6px}h1{font-size:25pt}.symbol{font-size:19pt}h2.headline{font-size:16pt;margin:5px 0 7px}h2{font-size:11pt}h3{font-size:9pt}.panel{margin-top:6px}header{padding:2px 4px;margin-bottom:4px}.number{width:17px;height:17px;font-size:10pt}.sub{font-size:7pt}.overview>p{font-size:9pt}.facts>div{padding:4px}.facts b{font-size:11pt}small,.micro{font-size:7pt}.value-chain{max-height:103px}.art-note{font-size:6pt}.segment{padding:4px 5px}.segment-value{font-size:12pt}.segment p{font-size:8pt}.segment small{font-size:7pt}.bridge{padding:4px;margin-top:4px}.bridge b{font-size:11pt}.bridge small{font-size:7pt}.chart h3{font-size:8pt}.chart h3 span{font-size:7pt}.change{font-size:14pt}.chart small{font-size:6pt}.chart svg{max-height:68px}.claim{margin-bottom:4px}.basis{font-size:6pt}.question{padding:4px}.thesis-note{font-size:7pt}.research-grid{gap:7px}.triggers>div{padding:4px 6px}table,.debates td{font-size:8pt}th,td{padding:4px}.peers th,.peers td{font-size:8pt}.peers td small{font-size:6pt}.sensitivity td,.sensitivity th{font-size:8pt;padding:3px}.sensitivity small{font-size:6pt}.footer{font-size:6pt;margin-top:6px}.micro{font-size:6.5pt}.watch .claim{font-size:8pt}sup{font-size:6pt}details{display:none}}
@media screen and (max-width:650px){main{margin:0;padding:15px}h1{font-size:29px}.headline{font-size:21px!important}.mast{font-size:8px;gap:8px}.overview,.research-grid,.valuation,.triggers{display:block}.facts{margin-top:8px}.facts b{font-size:15px}.segments{grid-template-columns:repeat(2,minmax(0,1fr))}.value-chain{max-height:none;object-fit:contain}.bridge{flex-wrap:wrap;gap:6px}.bridge>div{display:block}.charts{grid-template-columns:repeat(2,minmax(0,1fr))}.charts>figure:last-child{grid-column:1 / span 2;max-width:60%}.chart svg{max-height:160px}.sub{display:none}.triggers>div{margin-bottom:6px}.valuation>div{margin-bottom:10px}table,.debates td{font-size:13px}small,.micro,.footer{font-size:11px}.basis{font-size:10px}.watch th:first-child{width:28%}}
'''

CSS += """
.company-hero{display:grid;grid-template-columns:1fr 1.15fr;gap:12px;align-items:center}.company-hero .overview{display:block}.company-hero .facts{margin-top:8px}.company-hero .value-chain{max-height:none;height:auto;object-fit:contain}.lower-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.lower-grid .valuation{display:block}.lower-grid .valuation>div+div{margin-top:8px}.lower-grid .triggers{display:block}.lower-grid .triggers>div{margin-bottom:5px}.lower-grid .watch th:first-child{width:27%}
@media print{.company-hero{gap:8px}.company-hero .facts{margin-top:5px}.company-hero .facts b{font-size:10pt}.lower-grid{gap:7px}.lower-grid .valuation>div+div{margin-top:5px}.lower-grid .micro{font-size:6.5pt}.triggers .claim{font-size:8pt}.watch th{font-size:8pt}.panel{margin-top:5px}.art-note{font-size:6pt}}
@media screen and (max-width:650px){.company-hero,.lower-grid{display:block}.company-hero .value-chain{margin-top:8px}.lower-grid .triggers{display:grid;grid-template-columns:1fr}.facts b{font-size:14px}.lower-grid .watch th:first-child{width:28%}}
"""

CSS += """@media print{.title{margin-top:3px}h2.headline{margin:3px 0 5px;font-size:15pt}.panel{margin-top:4px}.company-hero{grid-template-columns:1fr .9fr}.company-hero .value-chain{max-height:80px;object-fit:contain}.company-hero .art-note{display:none}.segment small{font-size:7pt}.segment-value{margin:1px 0}.segment p{margin:1px 0}.chart svg{max-height:55px}.chart{padding:3px}.claim{margin-bottom:3px}.micro{margin-top:2px}.question{margin-bottom:4px}.triggers>div{padding:3px 5px}.sensitivity td,.sensitivity th{padding:2px}.footer{font-size:6pt}header{margin-bottom:3px}.lower-grid{grid-template-columns:1fr 1fr}.lower-grid .valuation>div+div{margin-top:3px}}"""

CSS += """@media print{.sensitivity small{display:inline;font-size:6pt}.sensitivity th,.sensitivity td{font-size:8pt;white-space:nowrap}.chart svg{max-height:51px}.footer{line-height:1.1}.company-hero .value-chain{max-height:85px}.watch .claim{font-size:8pt}.peers th,.peers td{padding:4px}.lower-grid .valuation .micro{font-size:6pt}}"""
CSS += '@media print{.panel{margin-top:2px}header{margin-bottom:2px}.footer{padding-top:3px;margin-top:3px}.micro{margin-top:1px}}'
CSS += '@media print{.company-hero .value-chain{height:72px;max-height:72px;object-fit:cover;object-position:center 50%}.mast{padding-bottom:3px}.headline{margin-bottom:4px!important}}'

CSS += '.lower-grid{column-gap:10px;row-gap:0}.lower-grid>.panel:nth-child(2){grid-column:2;grid-row:1 / span 2}.lower-grid>.panel:nth-child(3){grid-column:1;grid-row:2}'
