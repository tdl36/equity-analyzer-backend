"""Current layout of existing assignment evidence, without refreshing or rewriting it."""
import json
from research_onepager import render,VERSION as BRIEF_VERSION
from research_infographic import VERSION as ILLUSTRATED_VERSION


def current_view(run,row):
    if not row or row.get('ticker')!=run.get('ticker') or row.get('report_id')!=run.get('id'):
        raise ValueError('The visual does not belong to this saved report and company.')
    assignment=row.get('input') or {};fundamentals=row.get('fundamentals')
    if isinstance(assignment,str):assignment=json.loads(assignment)
    if isinstance(fundamentals,str):fundamentals=json.loads(fundamentals)
    return {'assignmentId':str(row['id']), 'html':render(run,{**assignment,'_visualFundamentals':fundamentals}),
            'rendererVersion':ILLUSTRATED_VERSION if fundamentals else BRIEF_VERSION,
            'publicDataAsOf':fundamentals.get('asOf') if fundamentals else None,
            'publicDataRetrievedAt':fundamentals.get('retrievedAt') if fundamentals else None,
            'originalUrl':f"/api/research/assignments/{row['id']}/artifact/visual?render=original",
            'replayed':True,'currentLayout':True}


def assignment_view(cur,run):
    cur.execute("SELECT to_regclass('mp_jobs') AS name")
    if not cur.fetchone()['name']:return None
    cur.execute("""SELECT id,ticker,input,result->>'reportId' AS report_id,
        result->'visualFundamentals' AS fundamentals FROM mp_jobs
        WHERE stage='research_assignment' AND ticker=%s AND result->>'reportId'=%s
        AND result->'artifacts' ? 'visual' ORDER BY created_at DESC LIMIT 2""",(run['ticker'],run['id']))
    rows=list(cur.fetchall())
    if len(rows)>1:raise ValueError('Multiple assignment visuals reference this report. Open the specific assignment to choose its visual.')
    return current_view(run,dict(rows[0])) if rows else None
