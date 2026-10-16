"""Stable, company-scoped pages of saved note jobs; no worker or model dispatch."""
import base64
import json
import uuid
from datetime import datetime


def read_page(cur, ticker, cursor=None):
    params=[ticker]
    after=''
    if cursor is not None:
        try:
            if not isinstance(cursor,str) or not 1<=len(cursor)<=512:raise ValueError()
            value=json.loads(base64.b64decode(cursor,altchars=b'-_',validate=True))
            if not isinstance(value,list) or len(value)!=3 or value[2]!=ticker:raise ValueError()
            stamp=datetime.fromisoformat(value[0]);ident=str(uuid.UUID(value[1]))
            if ident!=value[1] or stamp.tzinfo is None:raise ValueError()
        except (ValueError,TypeError,AttributeError,UnicodeError):
            raise ValueError('Invalid saved-note archive cursor.') from None
        after=' AND (j.created_at,j.note_id)<(%s,%s)';params.extend([stamp,ident])
    cur.execute('''SELECT j.note_id,j.status,j.reason,j.created_at,v.revision,
            COALESCE(v.payload->>'title',p.plan->>'title') AS title,
            (SELECT MAX(revision) FROM pipeline_lab_note_acceptances a WHERE a.note_id=j.note_id) AS accepted_revision
        FROM pipeline_lab_note_jobs j JOIN pipeline_lab_note_plans p USING(note_id)
        LEFT JOIN LATERAL (SELECT revision,payload FROM pipeline_lab_note_versions
            WHERE note_id=j.note_id ORDER BY revision DESC LIMIT 1) v ON TRUE
        WHERE p.plan->>'ticker'=%s'''+after+' ORDER BY j.created_at DESC,j.note_id DESC LIMIT 31',tuple(params))
    records=cur.fetchall()
    rows=[{'noteId':str(r['note_id']),'ticker':ticker,'status':r['status'],'reason':r['reason'],
           'revision':r['revision'],'acceptedRevision':r['accepted_revision'],'title':r['title'] or 'Untitled note',
           'createdAt':r['created_at'].isoformat()} for r in records[:30]]
    next_cursor=None
    if len(records)>30:
        row=rows[-1]
        next_cursor=base64.urlsafe_b64encode(json.dumps([row['createdAt'],row['noteId'],ticker],separators=(',',':')).encode()).decode()
    return {'ticker':ticker,'jobs':rows,'nextCursor':next_cursor,'readOnly':True}
