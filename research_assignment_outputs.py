"""Source-linked derivative drafts. No additional model call or investor approval."""
import copy
import html
from stock_analysis_visual import supported


def fields(run,prefix):
    return [(p,c) for p,c in run['state'].get('citations',{}).items()
            if p.startswith(prefix) and supported(c) and c.get('statement')]


def note(run,kind,assignment):
    esc=html.escape
    groups=([('Investment view','/summary/'),('Catalysts','/catalysts/'),('Risks','/risks/'),('What to monitor','/monitor/')]
        if kind=='summary_note' else [('Company','/summary/one_liner'),('Business','/business/'),('Financials','/financials/'),('Valuation','/valuation/'),('Risks','/risks/')])
    if kind=='visual':
        groups=[('Company','/summary/one_liner'),('Business model','/business/revenue_engine'),('Financial direction','/summary/earnings_direction'),('Thesis','/summary/investment_thesis/'),('Market debate','/debates/'),('Catalysts','/catalysts/'),('Valuation','/valuation/'),('Risks','/risks/'),('Monitor','/monitor/')]
    refs={s['id']:i+1 for i,s in enumerate(run['sources'])}
    sections=[]
    for title,prefix in groups:
        chosen=fields(run,prefix)[:(2 if kind=='visual' else 12)]
        lines=[]
        for path,c in chosen:
            text=c['statement'];text=(text[:320].rsplit(' ',1)[0]+'… (see full report)') if kind=='visual' and len(text)>320 else text
            ids=sorted({refs[e['sourceId']] for e in c.get('evidence',[]) if e.get('matched') and e['sourceId'] in refs})
            field_label=' · '.join(str(int(x)+1) if x.isdigit() else x.replace('_',' ') for x in path.strip('/').split('/')[1:])
            if path.startswith('/financials/historical/'):
                period_path='/'.join(path.split('/')[:4])+'/period'
                period=run['state'].get('citations',{}).get(period_path)
                field_label+=' · '+(period['statement'] if supported(period) else 'period needs review')
            lines.append('<p><b>'+esc(field_label)+'</b><br>'+esc(text)+' <small>['+', '.join(map(str,ids))+'] · '+esc(c.get('basis','interpretation').replace('_',' '))+'</small></p>')
        sections.append('<section><h2>'+esc(title)+'</h2>'+(''.join(lines) or '<p>Source-supported coverage unavailable.</p>')+'</section>')
    title={'summary_note':'Summary note','stock_summary':'Stock summary','visual':'Visual one-pager draft'}[kind]
    coverage_warning='<p><b>Source coverage:</b> Some originals were excerpted within the fixed research context. Omitted text was not reviewed; see source provenance and the full originals.</p>' if any(s.get('coverage') for s in run['sources']) else ''
    sources=''.join('<li>'+esc(s['filename'])+' — '+esc(s.get('sourceUrl',''))+'<br><small>SHA-256 '+esc(s['originalHash'])+'</small>'+('<p>'+esc(s['coverage']['limitation'])+'</p>' if s.get('coverage') else '')+'</li>' for s in run['sources'])
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box;color:#000;font-family:Calibri,sans-serif}body{margin:28px auto;padding:0 18px;max-width:1100px;font-size:15px;line-height:1.45}h1{font-size:28px}h2{font-size:18px;margin:0 0 10px}section{background:#f4f7f8;border:1px solid #cdd8dd;padding:15px;border-radius:7px;break-inside:avoid}main{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}small{font-size:11px}p,li{overflow-wrap:anywhere}details{margin-top:20px}@media print{@page{size:A4 landscape;margin:10mm}body{font-size:10px;margin:0;padding:0}h1{font-size:20px}h2{font-size:13px}main{grid-template-columns:repeat(3,1fr);gap:7px}section{padding:8px}details{break-before:page}}@media(max-width:600px){main{grid-template-columns:1fr}}</style></head><body>'''+f'<h1>{esc(run["ticker"])} · {title}</h1><p>{esc(assignment["since"])}–{esc(assignment["until"])} · {esc(assignment["horizon"])}</p><p><b>Draft for investor review.</b> Selected original passages were matched and model-reviewed; figures are not independently certified. Missing coverage stays visible. The date range describes collection, not the observation date of every figure.</p>'+coverage_warning+'<main>'+''.join(sections)+'</main><details><summary>Sources and research provenance</summary><p>Research report '+esc(run['id'])+'</p><ol>'+sources+'</ol></details></body></html>'


def thesis(run,prepared,assignment_id):
    """Preserve existing item identities; append separately identified new evidence."""
    from thesis_imports import normalize
    p=copy.deepcopy(prepared);a=p['analysis'];suffix=assignment_id[:8]
    sources={s['id']:s for s in run['sources']}
    def refs(c):
        return [{'sourceId':'ra-'+suffix+'-'+e['sourceId'],'filename':sources[e['sourceId']]['filename'],
                 'sha256':sources[e['sourceId']]['originalHash'],'excerpt':e['excerpt'],
                 'supportType':c.get('basis','interpretation')}
                for e in c.get('evidence',[]) if e.get('matched') and e['sourceId'] in sources]
    claims=fields(run,'/summary/investment_thesis/')
    if not claims:raise ValueError('No supported thesis statements were produced. Resume explicitly to retry the first author and source-review pair; this may add two paid calls. Other completed stages and prior output are retained.')
    initial=p['baseline']['mode']=='initial'
    if initial:a={'thesis':{'summary':'','pillars':[]},'signposts':[],'threats':[]}
    if not isinstance(a.get('thesis'),dict) or not isinstance(a['thesis'].get('pillars'),list):raise ValueError('Existing thesis structure needs review before automatic draft preparation.')
    for group in ('signposts','threats'):
        if a.get(group) is None:a[group]=[]
        if not isinstance(a[group],list):raise ValueError('Existing thesis monitoring structure needs review.')
    # Keep legacy reference identities without claiming their originals were re-read.
    registered={s['id'] for s in p['sourceRegister']}
    for item in a['thesis']['pillars']+a['signposts']+a['threats']:
        for ref in item.get('sources',[]):
            sid=ref.get('sourceId')
            if sid and sid not in registered:
                p['sourceRegister'].append({'id':sid,'name':ref.get('filename') or sid,**{k:ref[k] for k in ('filename','sha256','url') if ref.get(k)},'usageStatus':'Carried forward from existing thesis; not collected or re-reviewed by this assignment'})
                registered.add(sid)
    a['thesis']['summary']='\n\n'.join(c['statement'] for _,c in claims)
    for i,(_,c) in enumerate(claims):
        a['thesis']['pillars'].append({'id':f'RA-{suffix}-P{i+1}','title':'Research update '+str(i+1),'description':c['statement'],'confidence':'Requires investor review','sources':refs(c)})
    for group,prefix,label,extra in [('signposts','/monitor/','metric',{'target':'Review the cited observation against subsequent company evidence. No additional numeric threshold inferred.','timeframe':'Next relevant company update; exact timing not established by this mapping.'}),('threats','/risks/','threat',{'triggerPoints':'Reassess if the cited risk materializes; no probability or additional trigger inferred.'})]:
        entries=fields(run,prefix)[:6]
        for i,(_,c) in enumerate(entries):a[group].append({'id':f'RA-{suffix}-{group}-{i}',label:c['statement'],**extra,'sources':refs(c)})
        if not a[group]:a[group]=[{'id':f'RA-{suffix}-{group}-gap',label:'Evidence gap: '+group,**extra,'sources':[]}]
    a['conclusion']='Proposed evidence update from research assignment '+assignment_id+'. Existing item identities were preserved; older pillars/signposts/risks are carried forward, not independently re-reviewed. New entries are additive. Review removals, contradictions and significance before approval. No live thesis has been changed.'
    if any(s.get('coverage') for s in run['sources']):a['conclusion']+=' Source coverage is partial: some originals were excerpted within the fixed context limit. Omitted text was not reviewed; inspect source provenance and originals.'
    a['researchAssignment']={'id':assignment_id,'reportId':run['id'],'reviewRequired':True,'comparison':run['state'].get('comparison',{})}
    p['analysis']=a;p['companyName']=p.get('companyName') or run['input'].get('companyName') or run['ticker']
    p['sourceRegister'] += [{'id':'ra-'+suffix+'-'+s['id'],'name':s['filename'],'filename':s['filename'],'sha256':s['originalHash'],'url':s.get('sourceUrl',''),'reviewStatus':'Passage matching and model review; physical page not established. '+s.get('coverage',{}).get('limitation','')} for s in run['sources']]
    p['provenance']={'execution':'automated-draft','authorTool':'Charlie research assignment','assignmentId':assignment_id,'reportId':run['id'],'investorApproved':False}
    return normalize(p)
