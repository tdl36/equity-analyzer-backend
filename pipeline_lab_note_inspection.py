"""Exact, read-only note-job inspection without source bodies or provider replies."""
import re
from pipeline_lab_notes import _note_identity


def inspect(cur, note_id, ticker):
    note_id=_note_identity(note_id)
    if not isinstance(ticker,str) or not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker):
        raise ValueError('An exact company ticker is required.')
    cur.execute('''SELECT j.note_id,j.status,j.reason,j.created_at,j.updated_at,
        p.plan->>'ticker' AS ticker,p.plan->>'batchCount' AS batch_count,
        p.plan->>'modelKey' AS model_key,
        (SELECT MAX(revision) FROM pipeline_lab_note_versions v WHERE v.note_id=j.note_id) AS revision,
        (SELECT MAX(revision) FROM pipeline_lab_note_acceptances a WHERE a.note_id=j.note_id) AS accepted_revision
        FROM pipeline_lab_note_jobs j JOIN pipeline_lab_note_plans p USING(note_id)
        WHERE j.note_id=%s AND p.plan->>'ticker'=%s''',(note_id,ticker))
    row=cur.fetchone()
    if not row:return None
    count=int(row['batch_count'])
    if not 1<=count<=8:raise ValueError('Saved note plan cannot be verified.')
    cur.execute('SELECT batch,status,updated_at FROM pipeline_lab_note_calls WHERE note_id=%s ORDER BY batch LIMIT 9',(note_id,))
    calls=cur.fetchall()
    if len(calls)>8 or any(not 1<=c['batch']<=count for c in calls):raise ValueError('Saved batch identities cannot be verified.')
    return {'noteId':note_id,'ticker':row['ticker'],'status':row['status'],'reason':row['reason'],
            'revision':row['revision'],'acceptedRevision':row['accepted_revision'],
            'createdAt':row['created_at'].isoformat(),'updatedAt':row['updated_at'].isoformat(),
            'batchCount':count,'modelKey':row['model_key'],'readOnly':True,
            'batches':[{'batch':c['batch'],'status':c['status'],'updatedAt':c['updated_at'].isoformat()} for c in calls]}
