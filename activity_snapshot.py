"""Read-only cross-company work inventory. Never resumes jobs or exposes inputs."""
from datetime import datetime, timezone
from flask import Blueprint, jsonify

TERMINAL = ('complete','completed','done','cancelled','stopped','applied','skipped')
ATTENTION = ('attention','needs_auth','failed','error','interrupted')
WORKFLOW_LABELS={'research_assignment':'End-to-end research','collection_control':'Collection request','auto_evidence_intake':'Evidence processing','command_meeting':'Meeting preparation','pipeline':'Meeting research','thesis':'Thesis draft','fanout':'Analyst research','orchestrate':'Research coordination'}
# Identifiers/projections are constants, never request input. No source bodies or keys.
SOURCES = [
 ('mp_jobs','Research workflows','desk',"stage, result->>'step' AS step, result->>'reportId' AS report_id, result->>'refreshRequestId' AS refresh_id, result->>'researchStages' AS stages, result->>'delayWarning' AS warning, COALESCE(input->>'assignmentId',input#>>'{payload,assignmentId}') AS assignment_id",'updated_at'),
 ('research_pipeline_jobs','Document processing','pipeline','job_type AS stage, current_step AS step','updated_at'),
 ('stock_analysis_runs','Stock analysis','stockanalysis',"jsonb_array_length(COALESCE(state->'completed','[]'::jsonb)) AS stages, state->>'inFlight' AS step",'updated_at'),
 ('company_research_runs','Company research','portfolio',"NULL::text AS step",'updated_at'),
 ('investment_committee_runs','Investment committee','portfolio',"NULL::text AS step",'updated_at'),
 ('analysis_jobs','Thesis processing','portfolio','progress AS step','updated_at'),
 ('agent_runs','Analyst team','desk','NULL::text AS step','COALESCE(completed_at,created_at)'),
 ('transcription_jobs','Audio processing','summary','progress AS step','updated_at'),
]

def stamp(value):
    if not value:return None
    if isinstance(value,(int,float)):
        try:value=datetime.fromtimestamp(value,timezone.utc)
        except (ValueError,OverflowError,OSError):return None
    if isinstance(value,str):
        try:value=datetime.fromisoformat(value.replace('Z','+00:00'))
        except ValueError:return None
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).isoformat()

def bucket(status):
    return 'recent' if status in TERMINAL else 'attention' if status in ATTENTION else 'active'

def assemble(rows, collection, now):
    parents=[r for r in rows if r.get('stage')=='research_assignment']
    reports={r.get('report_id') for r in parents}; refreshes={r.get('refresh_id') for r in parents}
    parent_ids={str(r['id']) for r in parents}
    items=[]
    for r in rows:
        if r.get('assignment_id') in parent_ids or (r['source']=='stock_analysis_runs' and str(r['id']) in reports):continue
        is_parent=r.get('stage')=='research_assignment';status=r.get('status') or 'unknown'
        stages=r.get('stages');stages=int(stages) if str(stages).isdigit() else None
        items.append(dict(id=r['source']+':'+str(r['id']),jobId=str(r['id']),ticker=r.get('ticker') or '',
            title=WORKFLOW_LABELS.get(r.get('stage'),r['label']),status=status,bucket=bucket(status),
            step=r.get('step') or ('Waiting for Mac collection' if is_parent else status.replace('_',' ')),
            stages=stages,warning=r.get('warning'),error=(r.get('error') or '')[:600],
            createdAt=stamp(r.get('created_at')),updatedAt=stamp(r.get('updated_at')),
            view='stockanalysis' if is_parent else r['view']))
    for r in (collection or {}).get('requests',[]):
        if r.get('id') in refreshes:continue
        if bucket(r.get('status'))=='recent' and (not stamp(r.get('created')) or (now-datetime.fromisoformat(stamp(r.get('created')))).total_seconds()>86400):continue
        items.append(dict(id='collection:'+r['id'],jobId=r['id'],ticker=r.get('ticker',''),title='AlphaSense collection',
            status=r.get('status','unknown'),bucket=bucket(r.get('status')),step=r.get('issue') or {'queued':'Waiting for browser collection','complete':'Originals verified'}.get(r.get('status'),'Collecting and verifying originals'),
            error=r.get('issue',''),createdAt=stamp(r.get('created')),updatedAt=None,view='desk'))
    for item in items:
        updated=item['updatedAt']; age=(now-datetime.fromisoformat(updated)).total_seconds() if updated else None
        item['quietMinutes']=int(max(0,age)/60) if age is not None else None
        item['stale']=item['bucket']=='active' and (age is None or age>900)
    items.sort(key=lambda r:({'attention':0,'active':1,'recent':2}[r['bucket']],-(datetime.fromisoformat(r['updatedAt']).timestamp() if r['updatedAt'] else 0)))
    return items

def create_blueprint(get_db):
    bp=Blueprint('activity_snapshot',__name__)
    @bp.get('/api/activity')
    def snapshot():
        rows=[]; unavailable=[]; truncated=[]; now=datetime.now(timezone.utc)
        for table,label,view,extra,updated in SOURCES:
            try:
                with get_db() as (_,cur):
                    cur.execute('SELECT to_regclass(%s) AS name',(table,))
                    if not cur.fetchone()['name']:continue
                    ticker="''::text" if table=='transcription_jobs' else 'ticker'
                    error='NULL::text' if table=='agent_runs' else 'error'
                    # Active work is selected independently of recent history: old
                    # blocked jobs must never disappear behind newer completions.
                    projection=f'id,{ticker} AS ticker,status,{error} AS error,created_at,{updated} AS updated_at,{extra}'
                    cur.execute(f'SELECT {projection} FROM {table} WHERE status NOT IN %s ORDER BY {updated} DESC LIMIT 201',(TERMINAL,))
                    active=list(cur.fetchall())
                    if len(active)>200:truncated.append(label)
                    cur.execute(f'SELECT {projection} FROM {table} WHERE status IN %s AND {updated}>NOW()-INTERVAL \'24 hours\' ORDER BY {updated} DESC LIMIT 20',(TERMINAL,))
                    for r in active[:200]+list(cur.fetchall()):rows.append({**dict(r),'source':table,'label':label,'view':view})
            except Exception:unavailable.append(label)
        collection={}; collected_at=None; agent=None
        try:
            with get_db() as (_,cur):
                cur.execute("SELECT value,updated_at FROM app_settings WHERE key='collection_control_snapshot'");r=cur.fetchone()
                if r:
                    import json
                    collection=json.loads(r['value']) if isinstance(r['value'],str) else r['value'];collected_at=stamp(r['updated_at'])
                cur.execute('SELECT MAX(last_seen) AS last_seen FROM agent_heartbeats');r=cur.fetchone();agent=stamp(r['last_seen']) if r else None
        except Exception:unavailable.append('Mac status')
        items=assemble(rows,collection,now)
        response=jsonify(items=items,counts={b:sum(r['bucket']==b for r in items) for b in ('active','attention','recent')},
            asOf=now.isoformat(),agentLastSeen=agent,collectionLastSeen=collected_at,unavailable=unavailable,truncated=truncated,
            scope='Saved research, collection, document and audio jobs. Recent history shows up to 20 results per workflow from the last 24 hours. Browser collection needs the Mac awake, Codex running and AlphaSense signed in.')
        response.headers['Cache-Control']='no-store';return response
    return bp
