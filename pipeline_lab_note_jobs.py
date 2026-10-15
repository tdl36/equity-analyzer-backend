"""Pending durable queue for Lab full notes. No scheduler or routes registered.

An expired worker becomes attention, never silently starts a new paid attempt.
"""
import uuid
from pipeline_lab_note_plan import freeze
from pipeline_lab_notes import _note_identity


def schema(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS pipeline_lab_note_jobs (
        note_id UUID PRIMARY KEY REFERENCES pipeline_lab_note_plans(note_id),
        status TEXT NOT NULL CHECK(status IN ('queued','running','attention','draft')),
        lease_id UUID, lease_until TIMESTAMPTZ, reason TEXT, revision INTEGER,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')


def enqueue(cur, plan):
    receipt=freeze(cur,plan)
    cur.execute("INSERT INTO pipeline_lab_note_jobs(note_id,status) VALUES(%s,'queued') ON CONFLICT(note_id) DO NOTHING",(receipt['noteId'],))
    return status(cur,receipt['noteId'])


def status(cur, note_id):
    cur.execute('SELECT note_id,status,reason,revision,created_at,updated_at FROM pipeline_lab_note_jobs WHERE note_id=%s',(_note_identity(note_id),))
    row=cur.fetchone()
    if not row:return None
    return {'noteId':str(row['note_id']),'status':row['status'],'reason':row['reason'],
            'revision':row['revision'],'createdAt':row['created_at'].isoformat(),
            'updatedAt':row['updated_at'].isoformat()}


def claim(cur):
    # Reap first, preserving provider journal and draft state for reconciliation.
    cur.execute("UPDATE pipeline_lab_note_jobs SET status='attention',reason='worker_interrupted',lease_id=NULL,lease_until=NULL,updated_at=NOW() WHERE status='running' AND lease_until<=NOW()")
    cur.execute("SELECT note_id FROM pipeline_lab_note_jobs WHERE status='queued' ORDER BY created_at,note_id FOR UPDATE SKIP LOCKED LIMIT 1")
    row=cur.fetchone()
    if not row:return None
    lease=str(uuid.uuid4());note=str(row['note_id'])
    cur.execute("UPDATE pipeline_lab_note_jobs SET status='running',lease_id=%s,lease_until=NOW()+INTERVAL '5 minutes',updated_at=NOW() WHERE note_id=%s",(lease,note))
    return {'noteId':note,'leaseId':lease}


def renew(cur, note_id, lease_id):
    cur.execute("UPDATE pipeline_lab_note_jobs SET lease_until=NOW()+INTERVAL '5 minutes',updated_at=NOW() WHERE note_id=%s AND lease_id=%s AND status='running' AND lease_until>NOW() RETURNING note_id",(_note_identity(note_id),_note_identity(lease_id)))
    if not cur.fetchone():raise ValueError('Worker lease is no longer current.')


def settle(cur, note_id, lease_id, result):
    if not isinstance(result,dict):raise ValueError('Bounded job result required.')
    state=result.get('status');revision=None;reason=None
    if state=='draft' and set(result)=={'status','noteId','revision'}:
        if _note_identity(result['noteId'])!=_note_identity(note_id) or type(result['revision']) is not int or result['revision']<1:
            raise ValueError('Draft receipt does not match this job.')
        revision=result['revision']
        # Do not report a saved draft on the strength of a worker assertion alone.
        cur.execute('SELECT 1 FROM pipeline_lab_note_versions WHERE note_id=%s AND revision=%s',(_note_identity(note_id),revision))
        if not cur.fetchone():raise ValueError('Draft revision is not saved.')
    elif state=='attention' and set(result)=={'status','reason'}:
        reason=result['reason']
        if reason not in ('provider_outcome_unknown','saved_reply_needs_review',
                          'merge_context_exceeds_bound','intermediate_reply_truncated',
                          'source_check_failed','worker_interrupted','preflight_failed'):
            raise ValueError('Unknown recovery reason.')
    else:raise ValueError('Unsupported job result.')
    cur.execute("UPDATE pipeline_lab_note_jobs SET status=%s,reason=%s,revision=%s,lease_id=NULL,lease_until=NULL,updated_at=NOW() WHERE note_id=%s AND lease_id=%s AND status='running' AND lease_until>NOW() RETURNING note_id",(state,reason,revision,_note_identity(note_id),_note_identity(lease_id)))
    if not cur.fetchone():raise ValueError('Worker lease is no longer current.')
    return status(cur,note_id)


def recover(cur, note_id):
    """Explicit reconciliation only: inspect durable evidence, never call a model.

    Safe unstarted/fully checkpointed work can be queued. An uncertain call cannot.
    Active leases are never stolen. The future user route must be authenticated.
    """
    from pipeline_lab_note_plan import load
    from pipeline_lab_note_response import store_draft
    from pipeline_lab_notes import read, obj
    note_id=_note_identity(note_id)
    cur.execute('SELECT status,lease_until>NOW() AS live FROM pipeline_lab_note_jobs WHERE note_id=%s FOR UPDATE',(note_id,))
    row=cur.fetchone()
    if not row:raise ValueError('Note job not found.')
    if row['status']=='running' and row['live']:
        raise ValueError('A worker still owns this note job.')
    plan=load(cur,note_id)
    draft=read(cur,note_id)
    state='queued';reason=None;revision=None
    if draft:
        if draft['ticker']!=plan['ticker'] or draft['sourcePack']!=plan['sourcePack']:
            raise ValueError('Saved draft does not match the frozen plan.')
        state='draft';revision=draft['revision']
    else:
        cur.execute('SELECT batch,status,response,request FROM pipeline_lab_note_calls WHERE note_id=%s ORDER BY batch',(note_id,))
        calls=cur.fetchall()
        mismatch = any(
            obj(call['request'])['sourcePack'] != plan['sourcePack']
            or obj(call['request'])['contextSha256'] != plan['representationSha256']
            or obj(call['request'])['modelKey'] != plan['modelKey']
            or obj(call['request'])['maxTokens'] != plan['maxTokens']
            for call in calls)
        if mismatch:
            state='attention';reason='saved_reply_needs_review'
        elif any(call['status']!='complete' for call in calls):
            state='attention';reason='provider_outcome_unknown'
        elif calls and [c['batch'] for c in calls]!=list(range(1,len(calls)+1)):
            state='attention';reason='saved_reply_needs_review'
        elif calls and calls[-1]['batch']>plan['batchCount']:
            state='attention';reason='saved_reply_needs_review'
        elif any(obj(c['response'])['stopReason'] in ('max_tokens','length','MAX_TOKENS') for c in calls if c['batch']<plan['batchCount']):
            state='attention';reason='intermediate_reply_truncated'
        elif calls and calls[-1]['batch']==plan['batchCount']:
            try:
                receipt=store_draft(cur,note_id,plan['batchCount'],plan['ticker'],plan['title'],plan['warnings'])
                state='draft';revision=receipt['revision']
            except ValueError:
                state='attention';reason='saved_reply_needs_review'
    cur.execute('UPDATE pipeline_lab_note_jobs SET status=%s,reason=%s,revision=%s,lease_id=NULL,lease_until=NULL,updated_at=NOW() WHERE note_id=%s',(state,reason,revision,note_id))
    return status(cur,note_id)
