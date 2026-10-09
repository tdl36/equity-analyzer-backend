"""Deterministic, source-linked investment visuals saved in Charlie Studio storage."""
import base64
import html
import json
import re
from decimal import Decimal, InvalidOperation
from flask import jsonify, request
from company_research import eligible, digest
from stock_analysis import missing

RENDER_VERSION='stock-visual-v1'
PANELS=[('Company at a glance',['summary','industry']),('How the business makes money',['business']),
        ('Financial performance',['financials','earnings']),('Investment thesis and scenarios',['scenarios']),
        ('What the market is debating',['debates','expectations']),('Catalysts and risks',['catalysts','risks']),
        ('Valuation and peers',['valuation','peers']),('What to monitor',['monitor']),
        ('Bottom line and diligence',['infographic','management','diligence_questions'])]
esc=lambda v:html.escape(str(v or ''),quote=True)


def label(value):return value.replace('_',' ').capitalize()


def supported(c):return bool(c and c.get('review')=='supported' and c.get('passageMatched'))


def amount(text):
    """One explicitly labelled number. Never infer a scale or calculate a ratio."""
    match=re.fullmatch(r'\s*(USD|EUR|GBP|JPY|\$|€|£)?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|bn|[BMK]|%|x)?\s*',text or '',re.I)
    if not match:return None
    currency,number,unit=match.groups()
    if not currency and not unit:return None
    try:value=Decimal(number.replace(',',''))
    except InvalidOperation:return None
    if not value.is_finite() or abs(value)>Decimal('1e15'):return None
    return value,(currency or '')+' '+(unit or '')


def chart(report,citations):
    rows=report.get('financials',{}).get('historical',[])
    panels=[]
    for metric in ('revenue','operating_margin','eps','free_cash_flow','leverage'):
        points=[]
        for i,row in enumerate(rows):
            p=f'/financials/historical/{i}/'
            c=citations.get(p+metric)
            n=amount(row.get(metric))
            # A model review alone cannot invent numeric tokens absent from its quotations.
            if not (supported(c) and supported(citations.get(p+'period')) and n) or len(row['period'])>40:continue
            token=str(n[0]);quoted=' '.join(e['excerpt'].replace(',','') for e in c['evidence'] if e.get('matched'))
            if not re.search(r'(?<![\d.\-])'+re.escape(token)+r'(?![\d.])',quoted):continue
            points.append((row['period'],row[metric],n))
        if len(points)<2 or len({n[1] for _,_,n in points})!=1 or len({p for p,_,_ in points})!=len(points):continue
        # Positive series only: loss/recovery and mixed definitions remain in the source table.
        if any(n[0]<0 for _,_,n in points):continue
        maximum=max(n[0] for _,_,n in points)
        if not maximum:continue
        bars=''
        for i,(period,text,n) in enumerate(points):
            width=float(n[0]/maximum*360)
            y=24+i*62
            bars+=f'<text x="0" y="{y}" font-size="13">{esc(period)} · {esc(text)}</text><rect x="0" y="{y+8}" width="{width:.2f}" height="20" fill="#b9d5e4"/>'
        panels.append(f'<figure><figcaption>{esc(label(metric))} · {esc(points[0][2][1])}</figcaption><svg role="img" aria-label="{esc(label(metric))} by reported period" viewBox="0 0 540 {len(points)*62+10}" xmlns="http://www.w3.org/2000/svg">{bars}</svg></figure>')
    return ''.join(panels) or '<p>Comparable source-linked chart series unavailable. See reported values and periods below.</p>'


def document(run,visual=False):
    report=run.get('state',{}).get('report',{});citations=run.get('state',{}).get('citations',{})
    sources=run.get('sources',[]);refs={s['id']:str(i+1) for i,s in enumerate(sources)}
    def visible(value,path):
        if isinstance(value,dict):return any(visible(v,path+'/'+k) for k,v in value.items())
        if isinstance(value,list):return any(visible(v,path+'/'+str(i)) for i,v in enumerate(value))
        return not missing(value) and supported(citations.get(path))
    def render(value,path):
        if isinstance(value,dict):
            return ''.join(f'<div class="field"><h4>{esc(label(k))}</h4>{render(v,path+"/"+k)}</div>' for k,v in value.items() if not visual or visible(v,path+"/"+k))
        if isinstance(value,list):
            return '<div class="rows">'+''.join('<article>'+render(v,path+'/'+str(i))+'</article>' for i,v in enumerate(value))+'</div>' if value else '<p>Unavailable in this source pack.</p>'
        if missing(value):return '<p>Unavailable in this source pack.</p>'
        c=citations.get(path,{})
        if visual and not supported(c):return '<p>Not included · requires source review.</p>'
        numbers=', '.join(refs.get(e['sourceId'],'?') for e in c.get('evidence',[]) if e.get('matched'))
        status='Source passage matched; model reviewed' if supported(c) else 'Needs review'
        result=f'<p>{esc(value)}</p><small>{esc(c.get("basis","interpretation").replace("_"," "))} · {status}'+(f' [{numbers}]' if numbers else '')+'</small>'
        if not visual:
            result+='<details><summary>Source passages and review</summary><p>'+esc(c.get('reviewReason','Review pending'))+'</p>'+''.join(f'<blockquote>[{esc(refs.get(e["sourceId"],"?"))}] {esc(e["excerpt"])}</blockquote>' for e in c.get('evidence',[]))+'</details>'
        return result
    sections=''
    for i,(title,keys) in enumerate(PANELS):
        content=''.join(f'<div><h3>{esc(label(k))}</h3>{render(report[k],"/"+k)}</div>' for k in keys if k in report and (not visual or visible(report[k],"/"+k)))
        if i==2 and visual:content=chart(report,citations)+content
        sections+=f'<section><header><b>{i+1:02}</b><h2>{esc(title)}</h2></header>{content or "<p>Source-supported coverage unavailable in this report. Inspect the full report for gaps and unresolved fields.</p>"}</section>'
    source_html=''.join(f'<article><h3>[{i+1}] {esc(s["filename"])}</h3><p>{esc(s.get("sourceUrl") or "Observed URL unavailable")}</p><small>Original SHA-256: {esc(s["originalHash"])}<br>Extraction SHA-256: {esc(s.get("extractionHash"))}</small></article>' for i,s in enumerate(sources))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(run['ticker'])} · Stock analysis</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.55 Calibri,sans-serif;color:#000;background:#fff;max-width:1120px;margin:auto;padding:32px}}h1{{font-size:42px;line-height:1.1;margin:12px 0}}h2{{font-size:23px;margin:0}}h3{{font-size:19px}}h4{{font-size:15px;margin:10px 0 4px}}p{{margin:4px 0 9px;white-space:pre-wrap}}small{{font-size:12px;display:block}}section{{border:1px solid #ccd7de;border-radius:10px;overflow:hidden;margin:20px 0;padding:20px;break-inside:avoid}}header{{display:flex;align-items:center;gap:16px;background:#e7f1f7;margin:-20px -20px 18px;padding:14px 20px}}header b{{font-size:26px;flex-shrink:0;white-space:nowrap}}.rows{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr));gap:14px}}article{{background:#f6f8f7;border:1px solid #dce2de;padding:14px;margin:10px 0;border-radius:6px}}figure{{margin:16px 0;padding:16px;background:#f3f7fa}}svg{{max-width:650px;width:100%;font-family:Calibri,sans-serif;fill:#000}}blockquote{{border-left:2px solid #aaa;padding-left:14px}}body *{{overflow-wrap:anywhere}}@media(max-width:600px){{body{{padding:14px}}h1{{font-size:30px}}section{{padding:14px}}header{{margin:-14px -14px 14px}}}}@media print{{body{{padding:0}}details{{display:block}}}}
</style></head><body><small>CHARLIE / STOCK RESEARCH STUDIO / {esc(run['input'].get('mode','deep')).upper()}</small><h1>{esc(run['input'].get('companyName') or run['ticker'])}<br>From business model to investment thesis</h1><p>{esc(run['ticker'])} · {esc(run['created_at'])} · {esc(run['input'].get('horizon'))}</p><p>Report {esc(run['id'])} · Saved case R{run['baseline']['revision']} · {esc(run['status'])}</p><p>Draft research from selected originals. Source matching and model review do not independently establish truth, currentness or complete coverage. {'Unresolved fields are omitted from this visual; the full report retains them.' if visual else 'Read each field’s supporting passages before relying on it.'} Report creation date is not the financial observation date.</p>{sections}<section><h2>Sources and provenance</h2>{source_html}</section></body></html>'''


def register_routes(bp,get_db,ensure,read):
    @bp.get('/api/research/stock-analysis-run/<ident>/export')
    def export(ident):
        ensure();run=read(ident)
        if not run:return jsonify(error='Report not found'),404
        return jsonify(html=document(run),filename=run['ticker']+'-stock-analysis-'+ident+'.html')

    @bp.route('/api/research/stock-analysis-run/<ident>/infographic',methods=['GET','POST'])
    def infographic(ident):
        ensure();run=read(ident)
        if not run:return jsonify(error='Report not found'),404
        if run['status']!='complete':return jsonify(error='Complete and review the report first.'),409
        fingerprint=digest([RENDER_VERSION,ident,run['state'].get('report'),run['state'].get('citations')])
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('stock-visual:'+ident,))
            cur.execute("SELECT id,content FROM studio_outputs WHERE source_config->>'stockAnalysisId'=%s AND settings->>'fingerprint'=%s AND status='ready' ORDER BY id DESC LIMIT 1",(ident,fingerprint))
            existing=cur.fetchone()
            if existing:
                content=existing['content']
                if isinstance(content,str):content=json.loads(content)
                return jsonify(id=existing['id'],html=base64.b64decode(content['html']).decode(),fingerprint=fingerprint,replayed=True)
            if request.method=='GET':return jsonify(available=False)
            data=request.get_json(silent=True) or {}
            if not isinstance(data,dict) or data.get('verifiedFigures') is not True:
                return jsonify(error='Review the report’s figures, periods and source passages, then confirm before saving a visual.'),400
            from command_thesis_bridge import file_hash
            cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=ANY(%s)',(run['ticker'],run['input']['filenames']))
            docs=list(cur.fetchall())
            if len(docs)!=len(run['input']['filenames']) or any(not eligible(d.get('metadata') or {}) or file_hash(d)!=run['input']['hashes'].get(d['filename']) for d in docs):
                return jsonify(error='Original availability, content or permissions changed. Review sources before generating a visual.'),409
            output=document(run,visual=True)
            config={'stockAnalysisId':ident,'ticker':run['ticker'],'companyId':run['input']['companyId']}
            settings={'fingerprint':fingerprint,'renderVersion':RENDER_VERSION,'analystVerifiedFigures':True}
            content={'render_mode':'precise','format':'html','html':base64.b64encode(output.encode('ascii','xmlcharrefreplace')).decode()}
            cur.execute("INSERT INTO studio_outputs(title,type,status,source_config,settings,content) VALUES(%s,'infographic','ready',%s::jsonb,%s::jsonb,%s::jsonb) RETURNING id",(run['ticker']+' · Stock analysis',json.dumps(config),json.dumps(settings),json.dumps(content)))
            return jsonify(id=cur.fetchone()['id'],html=output,fingerprint=fingerprint),201
