"""Source-linked derivative drafts. No additional model call or investor approval."""
from research_extraction import has_omissions
from research_source_provenance import render as source_provenance
import copy
import html
from stock_analysis_visual import supported
from stock_evidence import valid_citations, report_fields


def fields(run,prefix):
    return [(p,c) for p,c in valid_citations(run).items()
            if p.startswith(prefix) and supported(c) and c.get('statement')]


def note(run,kind,assignment):
    if kind == "visual":
        from research_onepager import render
        return render(run, assignment)
    esc=html.escape
    purpose=assignment.get('researchPurpose') or {}
    event_note=kind=='summary_note' and purpose.get('kind')=='event_note'
    groups=([('Investment view','/summary/'),('Catalysts','/catalysts/'),('Risks','/risks/'),('What to monitor','/monitor/')]
        if kind=='summary_note' else [('Company','/summary/one_liner'),('Business','/business/'),('Financials','/financials/'),('Valuation','/valuation/'),('Risks','/risks/')])
    if event_note:
        # Reuse reviewed fields from the frozen shared report. Headings organize
        # evidence; they do not establish event timing or derive new comparisons.
        groups=[
            ('Event developments and context',('/earnings/reporting_period','/earnings/headline',
                '/earnings/what_changed','/earnings/management_tone','/summary/one_liner',
                '/catalysts/')),
            ('Results and supported expectations',('/earnings/results/','/earnings/beat_miss_quality',
                '/expectations/','/summary/what_market_believes/')),
            ('Potential investment implications',('/summary/core_question','/summary/investment_thesis/',
                '/summary/variant_view/','/summary/earnings_direction','/summary/top_drivers/')),
            ('Risks and uncertainties',('/risks/','/debates/','/earnings/analyst_concerns/')),
            ('Follow-up questions and monitoring',('/diligence_questions/','/monitor/')),
        ]
    refs={s['id']:i+1 for i,s in enumerate(run['sources'])}
    sections=[]
    field_order={path:i for i,path in enumerate(report_fields(run.get('state',{}).get('report',{})))} if event_note else {}
    for title,prefix in groups:
        candidates=fields(run,prefix)
        if event_note:candidates.sort(key=lambda pair:field_order[pair[0]])
        cap=40 if event_note else 12
        chosen=candidates[:cap]
        lines=[]
        for path,c in chosen:
            text=c['statement'];text=(text[:320].rsplit(' ',1)[0]+'… (see full report)') if kind=='visual' and len(text)>320 else text
            ids=sorted({refs[e['sourceId']] for e in c.get('evidence',[]) if e.get('matched') and e['sourceId'] in refs})
            field_label=' · '.join(str(int(x)+1) if x.isdigit() else x.replace('_',' ') for x in path.strip('/').split('/')[1:])
            if event_note and path.startswith('/summary/investment_thesis/'):
                field_label='Potential implication '+str(int(path.rsplit('/',1)[1])+1)
            if path.startswith('/financials/historical/'):
                period_path='/'.join(path.split('/')[:4])+'/period'
                period=valid_citations(run).get(period_path)
                field_label+=' · '+(period['statement'] if supported(period) else 'period needs review')
            lines.append('<p><b>'+esc(field_label)+'</b><br>'+esc(text)+' <small>['+', '.join(map(str,ids))+'] · '+esc(c.get('basis','interpretation').replace('_',' '))+'</small></p>')
        if event_note and len(candidates)>cap:
            lines.append('<p><b>Coverage limit:</b> '+str(len(candidates)-cap)+' additional supported fields are not shown in this section. Review the saved full report before drawing conclusions.</p>')
        sections.append('<section><h2>'+esc(title)+'</h2>'+(''.join(lines) or '<p>Source-supported coverage unavailable.</p>')+'</section>')
    title={'summary_note':'Summary note','stock_summary':'Stock summary','visual':'Visual one-pager draft'}[kind]
    if purpose.get('kind')=='event_note':
        event=purpose['event'];title='Event note · '+esc(event['title'])
        scope='<p><b>Event date:</b> '+esc(event['date'])+'. Standalone event analysis; this does not update the thesis or mark it current. Potential implications require a separate thesis proposal and investor approval.</p>'
        if event_note:
            scope+='<p>These sections organize supported claims from the saved research report, including historical context. Placement does not establish that every claim occurred on the event date. Comparisons and consensus are shown only when supported; no new estimates or calculations are inferred. Missing coverage remains visible.</p>'
    elif purpose.get('kind')=='thesis_review':
        scope='<p><b>Longitudinal thesis review:</b> The discovery window does not define the full investment case horizon. Review missing historical evidence and all proposed changes.</p>'
    else:scope=''
    coverage_warning='<p><b>Source coverage:</b> Some originals were excerpted within the fixed research context. Omitted text was not reviewed; see source provenance and the full originals.</p>' if has_omissions(run['sources']) else ''
    window_scope=('The date range describes linked collection discovery, not the publication dates or analytical horizon of this mixed evidence pack. Supplemental history was not collected by this request.' if assignment.get('processingProposal') else 'The date range is investor context only; these are explicitly selected stored originals, not a new collection.' if assignment.get('labSelection') else 'The date range describes collection, not the observation date of every figure.')
    dates=(('Linked collection window' if assignment.get('processingProposal') else 'Research context window' if assignment.get('labSelection') else 'Collection window')+': ' if event_note else '')+esc(assignment['since'])+'–'+esc(assignment['until'])
    if not event_note:dates+=' · '+esc(assignment['horizon'])
    sources=''.join('<li>'+esc(s['filename'])+' — '+esc(s.get('sourceUrl',''))+'<br><small>SHA-256 '+esc(s['originalHash'])+'</small>'+source_provenance(s,assignment)+'</li>' for s in run['sources'])
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box;color:#000;font-family:Calibri,sans-serif}body{margin:28px auto;padding:0 18px;max-width:1100px;font-size:15px;line-height:1.45}h1{font-size:28px}h2{font-size:18px;margin:0 0 10px}section{background:#f4f7f8;border:1px solid #cdd8dd;padding:15px;border-radius:7px;break-inside:avoid}main{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}small{font-size:11px}p,li{overflow-wrap:anywhere}details{margin-top:20px}@media print{@page{size:A4 landscape;margin:10mm}body{font-size:10px;margin:0;padding:0}h1{font-size:20px}h2{font-size:13px}main{grid-template-columns:repeat(3,1fr);gap:7px}section{padding:8px}details{break-before:page}}@media(max-width:600px){main{grid-template-columns:1fr}}</style></head><body>'''+f'<h1>{esc(run["ticker"])} · {title}</h1><p>{dates}</p><p><b>Draft for investor review.</b> Selected original passages were matched and model-reviewed; figures are not independently certified. Missing coverage stays visible. '+window_scope+'</p>'+scope+coverage_warning+'<main>'+''.join(sections)+'</main><details><summary>Sources and research provenance</summary><p>Research report '+esc(run['id'])+'</p><ol>'+sources+'</ol></details></body></html>'


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
    if not claims:
        if run.get('state',{}).get('qualityRevisions'):
            raise ValueError('No supported thesis statements were produced. The bounded quality retry is exhausted. Review the saved research and prepare a separate source-checked thesis proposal in Import & review thesis. Originals and completed outputs are retained; no further automatic paid retry is available.')
        raise ValueError('No supported thesis statements were produced. Resume explicitly to retry the first author and source-review pair; this may add two paid calls. Other completed stages and prior output are retained.')
    initial=p['baseline']['mode']=='initial'
    if initial:a={'thesis':{'summary':'','pillars':[]},'signposts':[],'threats':[]}
    if not isinstance(a.get('thesis'),dict) or not isinstance(a['thesis'].get('pillars'),list):raise ValueError('Existing thesis structure needs review before automatic draft preparation.')
    for group in ('signposts','threats'):
        if a.get(group) is None:a[group]=[]
        if not isinstance(a[group],list):raise ValueError('Existing thesis monitoring structure needs review.')
    # Older native theses stored risks without the later required trigger field.
    # Preserve their text/identity and expose the gap only in this pending draft.
    legacy_gaps=0
    for item in a['threats']:
        if item.get('triggerPoints') in (None,''):
            item['triggerPoints']='Not recorded in the prior thesis. Investor review is required to define a trigger; no new threshold has been inferred.'
            legacy_gaps+=1
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
    if legacy_gaps:a['conclusion']+=' '+str(legacy_gaps)+' carried-forward risks lacked trigger points; their missing fields are marked as evidence gaps in this draft.'
    if has_omissions(run['sources']):a['conclusion']+=' Source coverage is partial: some originals were excerpted within the fixed context limit. Omitted text was not reviewed; inspect source provenance and originals.'
    a['researchAssignment']={'id':assignment_id,'reportId':run['id'],'reviewRequired':True,'comparison':run['state'].get('comparison',{})}
    p['analysis']=a;p['companyName']=p.get('companyName') or run['input'].get('companyName') or run['ticker']
    p['sourceRegister'] += [{'id':'ra-'+suffix+'-'+s['id'],'name':s['filename'],'filename':s['filename'],'sha256':s['originalHash'],'url':s.get('sourceUrl',''),'reviewStatus':'Passage matching and model review; physical page not established. '+s.get('coverage',{}).get('limitation','')} for s in run['sources']]
    p['provenance']={'execution':'automated-draft','authorTool':'Charlie research assignment','assignmentId':assignment_id,'reportId':run['id'],'investorApproved':False}
    return normalize(p)
