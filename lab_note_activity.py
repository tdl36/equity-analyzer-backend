"""Bounded read-only Lab-note inventory; never claims leases or resumes work."""
QUERY = '''WITH notes AS (
 SELECT j.note_id::text AS id, p.plan->>'ticker' AS ticker,
 CASE WHEN j.status='draft' AND v.revision IS NULL THEN 'attention'
      WHEN j.status='draft' AND a.revision IS NOT NULL THEN 'complete'
      WHEN j.status='draft' THEN 'needs_review' ELSE j.status END AS status,
 j.status AS note_status, j.reason, v.revision,
 j.created_at, GREATEST(j.updated_at,v.created_at,a.accepted_at) AS updated_at
 FROM pipeline_lab_note_jobs j JOIN pipeline_lab_note_plans p USING(note_id)
 LEFT JOIN LATERAL (SELECT revision,created_at FROM pipeline_lab_note_versions
   WHERE note_id=j.note_id ORDER BY revision DESC LIMIT 1) v ON TRUE
 LEFT JOIN pipeline_lab_note_acceptances a ON a.note_id=j.note_id AND a.revision=v.revision
) SELECT * FROM notes WHERE {condition} ORDER BY updated_at DESC,id DESC LIMIT %s'''


def inventory(cur):
    cur.execute("SELECT to_regclass('pipeline_lab_note_jobs') AS name")
    if not cur.fetchone()['name']:
        return [], []
    result=[];truncated=[]
    for condition,limit,lane in [("status='attention'",200,'attention'),
            ("status='needs_review'",200,'review'),
            ("status NOT IN ('attention','needs_review','complete')",200,'active'),
            ("status='complete' AND updated_at>NOW()-INTERVAL '24 hours'",20,'recent')]:
        cur.execute(QUERY.format(condition=condition),(limit+1,))
        rows=cur.fetchall()
        if len(rows)>limit:truncated.append('Full Lab notes ('+lane+')')
        for row in rows[:limit]:
            r=dict(row);status=r['status']
            step={'needs_review':'Draft revision '+str(r['revision'])+' awaits investor review',
                  'complete':'Current draft revision accepted', 'queued':'Waiting for note worker',
                  'running':'Writing full Lab note', 'attention':'Full Lab note needs attention'}.get(status,'Note status unavailable')
            if r['note_status']=='draft' and r['revision'] is None:step='Saved draft receipt is missing'
            result.append(dict(id=r['id'],ticker=r['ticker'] or '',status=status,step=step,
                note_status=r['note_status'],created_at=r['created_at'],updated_at=r['updated_at'],
                source='pipeline_lab_note_jobs',stage='lab_note',label='Full Lab note',view='pipelinelab',
                warning='Inspect saved progress before retrying; an uncertain provider outcome must not be repeated.' if status=='attention' else None))
    return result,truncated
