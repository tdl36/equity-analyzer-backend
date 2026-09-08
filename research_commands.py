"""Bounded research commands and reusable favorites; no arbitrary tool execution."""
import hashlib
import json
import re
import uuid
from datetime import date, timedelta

KEY='research_command_favorites_v1'
DEFAULTS=[
 {'id':'filing','name':'8-K + sell-side reaction','instruction':"Review {ticker}'s 8-K on {date} and sell-side reaction. Pull relevant documents, explain what changed, and propose updates to my thesis.",'kind':'filing','days':1},
 {'id':'earnings','name':'Earnings decision brief','instruction':'Review {ticker} earnings around {date}. Reconcile results and guidance against expectations, assess the sell-side reaction, and propose thesis changes.','kind':'earnings','days':7},
 {'id':'clinical','name':'Clinical catalyst review','instruction':'Investigate {ticker} clinical trial developments around {date}. Verify endpoints, effect size, safety, trial design and commercial implications; challenge the bull and bear interpretations.','kind':'event','days':7},
 {'id':'weekly','name':'Weekly evidence refresh','instruction':'Review material developments for {ticker} through {date}. Identify new evidence, contradictions, risks and signposts, and propose thesis changes.','kind':'event','days':7}]

DEFAULTS=[{**row,'coordinated':True,'autoProposal':False} for row in DEFAULTS]

def obj(v):
    return json.loads(v) if isinstance(v,str) else v


def plan(data):
    if not isinstance(data,dict):raise ValueError('A research instruction is required.')
    ticker=data.get('ticker','').strip().upper() if isinstance(data.get('ticker'),str) else ''
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',ticker):raise ValueError('Choose one ticker for this task.')
    instruction=data.get('instruction');kind=data.get('kind')
    if not isinstance(instruction,str) or not 1<=len(instruction.strip())<=3000:raise ValueError('Instruction must contain 1–3,000 characters.')
    if kind not in ('filing','earnings','event'):raise ValueError('Choose filing, earnings or event research.')
    try:until=date.fromisoformat(data.get('date',''))
    except (ValueError,TypeError):raise ValueError('Choose an explicit event date (YYYY-MM-DD).')
    from zoneinfo import ZoneInfo
    from datetime import datetime
    if until>datetime.now(ZoneInfo('America/New_York')).date():raise ValueError('The event date cannot be in the future.')
    coordinated=data.get('coordinated',False)
    if type(coordinated)!=bool:raise ValueError('Coordinated review must be enabled or disabled.')
    auto_proposal=data.get('autoProposal',False)
    if type(auto_proposal)!=bool:raise ValueError('Automatic proposal preference must be enabled or disabled.')
    days=data.get('days',1 if kind=='filing' else 7)
    if type(days)!=int or not 1<=days<=90:raise ValueError('Choose a lookback of 1–90 days.')
    meeting = None
    if data.get('meetingPrep') is not None:
        from meeting_command_plan import meeting_options
        meeting = meeting_options(data['meetingPrep'])
    return {**({'sourcePolicy':__import__('source_preferences').policy(data['sourcePolicy'])} if data.get('sourcePolicy') is not None else {}), **({'meetingPrep':meeting} if meeting else {}), 'ticker':ticker,'instruction':instruction.strip().replace('{ticker}',ticker).replace('{date}',until.isoformat()),
            'kind':kind,'coordinated':coordinated,'autoProposal':auto_proposal,'since':(until-timedelta(days=days-1)).isoformat(),'until':until.isoformat(),
            'steps':['Verify dated primary sources','Collect AlphaSense reaction','Verify iCloud handoff','Run covering analyst recap']+(['Challenge the lead draft','Address each challenge','Run final selected-claim source review'] if coordinated else [])+(['Prepare a source-checked thesis proposal automatically'] if auto_proposal else [])+['Review proposed investment implications'],
            'sources':['SEC EDGAR 8-K and exhibits','AlphaSense press releases, broker reports and transcripts'],
            'limitations':['No filing or source may exist for the chosen date. Missing evidence is reported, not invented.',
                'Public lookup covers SEC EDGAR, with worker-reviewed FDA and clinical-registry supplements for relevant events. General web/IR retrieval remains limited.',
                'Completion of collection is separate from recap completion. Thesis edits require your decision.']}


def favorites(value):
    if not isinstance(value,list) or not 1<=len(value)<=30:raise ValueError('Keep 1–30 favorites.')
    out=[];ids=set()
    for row in value:
        if not isinstance(row,dict):raise ValueError('Invalid favorite.')
        ident=row.get('id');name=row.get('name');instruction=row.get('instruction')
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9-]{1,60}',ident) or ident in ids:raise ValueError('Favorite IDs must be distinct.')
        if not isinstance(name,str) or not 1<=len(name.strip())<=80:raise ValueError('Favorite name must contain 1–80 characters.')
        if not isinstance(instruction,str) or not 1<=len(instruction.strip())<=3000:raise ValueError('Favorite instruction must contain 1–3,000 characters.')
        if row.get('kind') not in ('filing','earnings','event') or type(row.get('days'))!=int or not 1<=row['days']<=90:raise ValueError('Invalid favorite workflow/window.')
        if type(row.get('coordinated',False))!=bool:raise ValueError('Invalid coordinated review preference.')
        if type(row.get('autoProposal',False))!=bool:raise ValueError('Invalid automatic proposal preference.')
        ids.add(ident);out.append(dict(id=ident,name=name.strip(),instruction=instruction.strip(),kind=row['kind'],days=row['days'],coordinated=row.get('coordinated',False),autoProposal=row.get('autoProposal',False)))
    return out


def revision(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def create_blueprint(get_db):
    from flask import Blueprint, jsonify, request
    bp=Blueprint('research_commands',__name__)
    from source_preferences import create_routes
    create_routes(bp,get_db)
    from assignment_workspace import routes
    routes(bp,get_db)
    @bp.route('/api/research/command-favorites',methods=['GET','PUT'])
    def saved():
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(KEY,))
            cur.execute('SELECT value FROM app_settings WHERE key=%s',(KEY,));r=cur.fetchone();value=obj(r['value']) if r else DEFAULTS
            if request.method=='PUT':
                data=request.get_json(silent=True) or {}
                if not isinstance(data,dict):return jsonify(error='Expected a favorites object.'),400
                if data.get('revision')!=revision(value):return jsonify(error='Favorites changed in another session. Reload before saving.'),409
                try:value=favorites(data.get('favorites'))
                except ValueError as e:return jsonify(error=str(e)),400
                cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(value)))
        return jsonify(favorites=value,revision=revision(value))

    @bp.route('/api/research/commands',methods=['GET','POST'])
    def commands():
        if request.method=='POST':
            d=request.get_json(silent=True) or {}
            try:
                uuid.UUID(d.get('requestId',''));p=plan(d)
            except (ValueError,TypeError,AttributeError) as e:return jsonify(error=str(e)),400
            value={'action':'research_task','payload':p};jid=d['requestId']
            with get_db(commit=True) as (_,cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('command:'+jid,))
                cur.execute('SELECT input,status,stage FROM mp_jobs WHERE id=%s',(jid,));old=cur.fetchone()
                if old:
                    if old['stage']!='collection_control' or obj(old['input'])!=value:return jsonify(error='Request ID already belongs to another instruction.'),409
                    return jsonify(id=jid,status=old['status'],plan=p)
                cur.execute('SELECT id FROM analysts WHERE %s=ANY(coverage_tickers) LIMIT 1',(p['ticker'],))
                if not cur.fetchone():return jsonify(error='Assign a covering analyst to this ticker before running the workflow.'),400
                cur.execute("INSERT INTO mp_jobs(id,stage,ticker,status,input) VALUES(%s,'collection_control',%s,'queued',%s::jsonb)",(jid,p['ticker'],json.dumps(value)))
            return jsonify(id=jid,status='queued',plan=p),202
        with get_db() as (_,cur):
            cur.execute("SELECT id,ticker,status,input,result,error,created_at FROM mp_jobs WHERE stage='collection_control' AND input->>'action'='research_task' ORDER BY created_at DESC LIMIT 30")
            jobs=[dict(r) for r in cur.fetchall()]
            cur.execute("SELECT value,updated_at FROM app_settings WHERE key='collection_control_snapshot'");row=cur.fetchone();snapshot=obj(row['value']) if row else {}
            # Exact topic links only; do not infer that another same-ticker recap belongs to this task.
            for j in jobs:
                result=obj(j['result']) if j.get('result') else {}
                rid=result.get('refreshRequestId');j['collection']=next((r for r in snapshot.get('requests',[]) if r['id']==rid),None)
                topic=result.get('topic');j['reports']=[]
                cur.execute("SELECT status,result,error FROM mp_jobs WHERE stage='command_proposal_dispatch' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(j['id'],));dispatch=cur.fetchone();j['proposalAutomation']=dict(dispatch) if dispatch else None
                cur.execute("SELECT id,status,error FROM mp_jobs WHERE stage='evidence_amendment' AND input->>'commandId'=%s ORDER BY created_at DESC LIMIT 1",(j['id'],));proposal=cur.fetchone();j['proposal']=dict(proposal) if proposal else None
                if topic:
                    cur.execute("SELECT a.id,a.status,a.activity_type,a.updated_at,(a.output->>'synthesisMarkdown' IS NOT NULL) AS has_report,p.recovery_attempts,p.current_step,p.result->'coordination'->'roles' AS roles FROM analyst_activities a LEFT JOIN research_pipeline_jobs p ON p.id=(a.output->>'catalystJobId') WHERE a.ticker=%s AND a.input->>'topic'=%s ORDER BY a.created_at DESC LIMIT 10",(j['ticker'],topic));j['reports']=[dict(r) for r in cur.fetchall()]
        from research_history import timestamp
        r=jsonify(jobs=jobs,macReportedAt=timestamp(row['updated_at']) if row else None);r.headers['Cache-Control']='no-store';return r
    return bp
