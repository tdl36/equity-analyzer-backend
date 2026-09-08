"""Atomic local-agent claims and bounded recovery of interrupted recap jobs."""
import json
import uuid
from flask import Blueprint, jsonify, request

LEASE_SQL="NOW()+INTERVAL '10 minutes'"


def token(value):
    try:return str(uuid.UUID(value))
    except (ValueError,TypeError,AttributeError):raise ValueError('A valid claim token is required')


def create_blueprint(get_db):
    bp=Blueprint('pipeline_recovery',__name__)
    @bp.route('/api/agent/claim',methods=['POST'])
    def claim():
        d=request.get_json(silent=True)
        if not isinstance(d,dict):return jsonify(error='Expected a claim object'),400
        # Older agents can claim atomically; automatic recovery is opt-in only
        # for the new token-aware synthesis worker.
        managed=bool(d.get('claimToken'))
        try:owner=token(d['claimToken']) if managed else None
        except ValueError as e:return jsonify(error=str(e)),400
        jid=d.get('jobId')
        with get_db(commit=True) as (_,cur):
            cur.execute('SELECT status,job_type,agent_owner FROM research_pipeline_jobs WHERE id=%s FOR UPDATE',(jid,));r=cur.fetchone()
            if not r:return jsonify(error='Job not found'),404
            if r['job_type'] not in ('note','synthesis','scan_catalysts','earnings_fetch'):return jsonify(error='Job is not assigned to the local agent'),400
            if r['status']=='running' and owner and r['agent_owner']==owner:return jsonify(claimed=True),200
            if r['status']!='queued':return jsonify(error='Job already claimed or finished'),409
            recoverable=managed and r['job_type']=='synthesis'
            cur.execute("UPDATE research_pipeline_jobs SET status='running',agent_owner=%s,agent_managed=%s,agent_lease_until="+LEASE_SQL+",updated_at=NOW(),current_step='Claimed by local agent' WHERE id=%s",(owner,recoverable,jid))
        return jsonify(claimed=True),200

    @bp.route('/api/agent/job-leases',methods=['POST'])
    def leases():
        d=request.get_json(silent=True)
        jobs=d.get('jobs') if isinstance(d,dict) else None
        if not isinstance(jobs,list) or len(jobs)>20:return jsonify(error='Expected at most 20 active jobs'),400
        try:
            claims=[(j['id'],token(j['claimToken'])) for j in jobs]
        except (KeyError,TypeError,ValueError):return jsonify(error='Invalid active claim'),400
        active=[]
        with get_db(commit=True) as (_,cur):
            for jid,owner in claims:
                cur.execute("UPDATE research_pipeline_jobs SET agent_lease_until="+LEASE_SQL+",updated_at=NOW() WHERE id=%s AND agent_owner=%s AND status='running' RETURNING id",(jid,owner))
                if cur.fetchone():active.append(jid)
        return jsonify(renewed=True,active=active)

    @bp.route('/api/agent/recover-jobs',methods=['POST'])
    def recover():
        recovered=[];exhausted=[]
        with get_db(commit=True) as (_,cur):
            cur.execute("SELECT id,recovery_attempts,steps_detail FROM research_pipeline_jobs WHERE agent_managed=TRUE AND job_type='synthesis' AND status='running' AND agent_lease_until<NOW() ORDER BY updated_at FOR UPDATE SKIP LOCKED LIMIT 10")
            for row in cur.fetchall():
                jid=row['id'];attempts=row['recovery_attempts'] or 0
                if attempts<2:
                    cur.execute("UPDATE research_pipeline_jobs SET status='queued',agent_owner=NULL,agent_lease_until=NULL,recovery_attempts=recovery_attempts+1,current_step='Interrupted worker; resuming from available source checkpoints',error=NULL,updated_at=NOW() WHERE id=%s",(jid,));recovered.append(jid)
                else:
                    cur.execute("UPDATE research_pipeline_jobs SET status='failed',agent_owner=NULL,agent_lease_until=NULL,current_step='Recovery needs attention',error='Automatic recovery limit reached after two restarts; completed checkpoints retained',updated_at=NOW() WHERE id=%s",(jid,));exhausted.append(jid)
                    detail=row['steps_detail'];detail=json.loads(detail) if isinstance(detail,str) else detail
                    if isinstance(detail,dict) and detail.get('activityId'):
                        cur.execute("UPDATE analyst_activities SET status='failed',error='Automatic recovery limit reached; inspect the retained source checkpoints and rerun when ready',updated_at=NOW() WHERE id=%s AND status='running' AND output->>'catalystJobId'=%s",(detail['activityId'],jid))
        return jsonify(recovered=recovered,exhausted=exhausted)
    return bp


def update_managed(get_db,data):
    """Returns None for legacy jobs, otherwise (body, HTTP status)."""
    with get_db(commit=True) as (_,cur):
        cur.execute('SELECT agent_managed,agent_owner,status FROM research_pipeline_jobs WHERE id=%s FOR UPDATE',(data.get('jobId'),));r=cur.fetchone()
        if not r or not r['agent_managed']:return None
        if not data.get('claimToken') or r['agent_owner']!=data['claimToken'] or r['status']!='running':return {'error':'Claim revoked or job already finished'},409
        status=data.get('status') or 'running'
        if status not in ('running','failed'):return {'error':'Complete recaps must use the durable result receipt'},400
        progress=data.get('progress')
        if progress is not None and (type(progress)!=int or not 0<=progress<=100):return {'error':'Invalid progress'},400
        cur.execute("UPDATE research_pipeline_jobs SET status=%s,current_step=%s,progress=COALESCE(%s,progress),error=%s,updated_at=NOW(),agent_lease_until="+LEASE_SQL+" WHERE id=%s",(status,str(data.get('currentStep') or '')[:1000],progress,str(data.get('error') or '')[:4000] or None,data['jobId']))
    return {'success':True},200
