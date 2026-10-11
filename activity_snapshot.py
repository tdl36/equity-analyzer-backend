"""Read-only cross-company work inventory. Never resumes jobs or exposes inputs."""
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify

TERMINAL = ('complete','completed','done','cancelled','stopped','applied','skipped','submitted')
ATTENTION = ('attention','needs_auth','failed','error','interrupted','needs_review','awaiting_approval','proposed','detected','blocked')
ACTIVE=('queued','running','pending','waiting','starting','processing','transcribing','downloading','uploading','collecting','collection_queued')
WORKFLOW_LABELS={'evidence_amendment':'Evidence proposal','catalyst_watch':'Catalyst monitoring','catalyst_signal':'Catalyst signal','research_assignment':'End-to-end research','collection_control':'Collection request','auto_evidence_intake':'Evidence processing','command_meeting':'Meeting preparation','pipeline':'Meeting research','thesis':'Thesis draft','fanout':'Analyst research','orchestrate':'Research coordination'}
# Identifiers/projections are constants, never request input. No source bodies or keys.
SOURCES = [
 ('mp_jobs','Research workflows','desk',"stage, result->>'step' AS step, result->>'reportId' AS report_id, result->>'refreshRequestId' AS refresh_id, result->>'researchStages' AS stages, result->>'delayWarning' AS warning, result->>'reason' AS reason, COALESCE(input->>'assignmentId',input#>>'{payload,assignmentId}') AS assignment_id",'updated_at'),
 ('research_pipeline_jobs','Document processing','pipeline','job_type AS stage, current_step AS step','updated_at'),
 ('stock_analysis_runs','Stock analysis','stockanalysis',"jsonb_array_length(COALESCE(state->'completed','[]'::jsonb)) AS stages, state->>'inFlight' AS step",'updated_at'),
 ('company_research_runs','Company research','portfolio',"NULL::text AS step",'updated_at'),
 ('investment_committee_runs','Investment committee','portfolio',"NULL::text AS step",'updated_at'),
 ('analysis_jobs','Thesis processing','portfolio','progress AS step','updated_at'),
 ('agent_runs','Analyst team','desk',"NULL::text AS step, logs->-1->>'ts' AS last_log_at, (status='queued' AND completed_at IS NULL AND logs @> '[{\"event\":\"execution_queued\"}]'::jsonb AND NOT logs @> '[{\"event\":\"execution_started\"}]'::jsonb) AS pending_start, (logs @> '[{\"event\":\"recover_single_v1\"}]'::jsonb OR logs @> '[{\"event\":\"batch_enrolled_v1\"}]'::jsonb) AS start_recovery",'COALESCE(completed_at,created_at)'),
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
    return 'recent' if status in TERMINAL else 'active' if status in ACTIVE else 'attention'

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
            step=r.get('step') or r.get('reason') or ('Waiting for Mac collection' if is_parent else status.replace('_',' ')),
            workflow=r.get('stage') or r['source'],
            stages=stages,warning=r.get('warning'),error=(r.get('error') or '')[:600],
            createdAt=stamp(r.get('created_at')),updatedAt=(stamp(r.get('last_log_at')) or stamp(r.get('updated_at'))) if r['source']=='agent_runs' and status in ACTIVE else stamp(r.get('updated_at')),
            pendingStart=bool(r.get('pending_start')) if r['source']=='agent_runs' else False,
            startRecovery=bool(r.get('start_recovery')) if r['source']=='agent_runs' else False,
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
        history_age=age if age is not None else ((now-datetime.fromisoformat(item['createdAt'])).total_seconds() if item['createdAt'] else None)
        if item.get('pendingStart'):
            item['step']='Waiting for first execution'
            item['warning']='Automatic start recovery is enabled; no execution has been reserved.' if item['startRecovery'] else 'No execution has been reserved. Automatic recovery is unavailable for this saved job.'
            if age is None or age<0 or age>900:
                item['bucket']='attention'
                item['warning']+=' Check backend worker availability; do not submit a duplicate.'
        if not item.get('pendingStart') and item.get('workflow')=='agent_runs' and item['status'] in ACTIVE and (age is None or age<0 or age>900):
            item['bucket']='attention'
            item['executionUnconfirmed']=True
            item['step']='Execution needs reconciliation'
            item['warning']='The saved run is unfinished, but current execution is not confirmed. Preserve its logs and check the backend/provider outcome before retrying; a retry may create another charge.'
            item['stale']=age is None or age>900
        if not item.get('pendingStart') and not item.get('executionUnconfirmed') and item['bucket']!='recent' and history_age is not None and history_age>7*86400:item['bucket']='older'
    items.sort(key=lambda r:({'attention':0,'active':1,'recent':2,'older':3}[r['bucket']],-(datetime.fromisoformat(r['updatedAt']).timestamp() if r['updatedAt'] else 0)))
    return items

def has_saved_report(value):
    """Batch enrollment occupies the legacy report field but is not research."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = json.loads(value)
    except ValueError:
        return True  # Analyst output is normally Markdown, not JSON.
    if isinstance(parsed, dict) and set(parsed).issubset({'batchId'}):
        return False
    return bool(parsed)


def create_blueprint(get_db):
    bp=Blueprint('activity_snapshot',__name__)
    @bp.get('/api/activity/analyst-run/<run_id>')
    def analyst_run(run_id):
        from uuid import UUID
        try:run_id=str(UUID(run_id))
        except ValueError:return jsonify(error='Invalid saved run.'),400
        with get_db() as (_,cur):
            cur.execute("SELECT id,ticker,status,llm_provider,llm_model,created_at,completed_at,logs, report, LENGTH(COALESCE(decision,''))>0 AS has_decision FROM agent_runs WHERE id=%s",(run_id,))
            r=cur.fetchone()
        if not r:return jsonify(error='Saved run not found.'),404
        logs=r.get('logs') or []
        if isinstance(logs,str):
            try:logs=json.loads(logs)
            except ValueError:logs=[]
        logs=logs if isinstance(logs,list) else []
        entries=[dict(time=stamp(x.get('ts')),agent=str(x.get('agent') or '')[:80],message=str(x.get('message') or '')[:1000]) for x in logs[-100:] if isinstance(x,dict)]
        response=jsonify(id=str(r['id']),ticker=r['ticker'],status=r['status'],provider=r['llm_provider'],model=r['llm_model'],
            createdAt=stamp(r['created_at']),completedAt=stamp(r['completed_at']),hasReport=has_saved_report(r['report']),hasDecision=bool(r['has_decision']),
            logs=entries,totalLogs=len(logs),truncated=len(logs)>100,asOf=datetime.now(timezone.utc).isoformat())
        response.headers['Cache-Control']='no-store'
        return response
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
                    cur.execute(f'SELECT {projection} FROM {table} WHERE status NOT IN %s ORDER BY CASE WHEN status IN %s THEN 1 ELSE 0 END,{updated} DESC LIMIT 201',(TERMINAL,ATTENTION))
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
        delivery=None
        try:
            with get_db() as (_,cur):
                cur.execute("SELECT COUNT(*) AS pending, MIN(created_at) AS oldest FROM mp_jobs WHERE stage='collection_control' AND status='queued'")
                r=cur.fetchone()
                delivery=dict(pending=int(r['pending']),oldestPendingAt=stamp(r['oldest']))
        except Exception:unavailable.append('Collection delivery')
        items=assemble(rows,collection,now)
        try:
            from research_review_due import due_items
            with get_db() as (_,cur):reviews,more=due_items(cur,now)
            items=reviews+items
            if more:truncated.append('Research issue reviews')
        except Exception:unavailable.append('Research issue reviews')
        meetings=None
        try:
            from meeting_agenda import upcoming
            with get_db() as (_,cur):meetings=upcoming(cur,now)
        except Exception:unavailable.append('Meeting preparation agenda')
        intake=[r for r in items if r.get('workflow')=='auto_evidence_intake' and r['status']=='waiting']
        if intake:
            try:
                from intake_status import annotate
                import json
                with get_db() as (_,cur):
                    cur.execute("SELECT value FROM app_settings WHERE key='research_auto_intake_v1'")
                    r=cur.fetchone();state=r['value'] if r else {}
                    if isinstance(state,str):state=json.loads(state)
                    annotate(cur,intake,state,now=now)
                for r in intake:r['step']=r['waitReason']
            except Exception:unavailable.append('Intake eligibility')
        response=jsonify(items=items,counts={b:sum(r['bucket']==b for r in items) for b in ('active','attention','recent','older')},
            meetingAgenda=meetings,asOf=now.isoformat(),agentLastSeen=agent,collectionLastSeen=collected_at,collectionDelivery=delivery,unavailable=unavailable,truncated=truncated,
            scope='Saved research, collection, document and audio jobs, plus due analyst issue reviews (New York dates). Work unchanged for more than seven days is retained under Older unfinished. Recent history shows up to 20 results per workflow from the last 24 hours. Browser collection needs the Mac awake, Codex running and AlphaSense signed in.')
        response.headers['Cache-Control']='no-store';return response
    return bp
