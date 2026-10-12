"""Bounded one-pager refresh provenance; never applies editorial changes."""
import re
import uuid

KEYS={'kind','assignmentId','runId','ticker','fingerprint','fundamentalsDigest','parentId'}

def validate(origin,ticker,outputs):
    if not isinstance(origin,dict) or set(origin)!=KEYS or origin.get('kind')!='onepager':
        raise ValueError('Invalid refresh origin.')
    if origin['ticker']!=ticker or 'visual' not in outputs:
        raise ValueError('One-pager refresh must retain its company and visual output.')
    for key in ('assignmentId','runId','parentId'):
        if key=='parentId' and origin[key] is None:continue
        if not isinstance(origin[key],str) or str(uuid.UUID(origin[key]))!=origin[key]:
            raise ValueError('Invalid refresh version identity.')
    for key in ('fingerprint','fundamentalsDigest'):
        if not isinstance(origin[key],str) or not re.fullmatch('[0-9a-f]{64}',origin[key]):
            raise ValueError('Invalid refresh baseline.')
    return dict(origin)

def verify(cur,origin):
    """Caller holds submission transaction; shared locks freeze report and head."""
    from onepager_edition_contract import baseline
    from onepager_decisions import read_lineage
    cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR SHARE",(origin['assignmentId'],))
    assignment=cur.fetchone()
    if not assignment:raise ValueError('The originating assignment is unavailable.')
    parent,_=read_lineage(cur,origin['assignmentId'])
    rid=(assignment.get('result') or {}).get('reportId')
    cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s FOR SHARE',(rid,))
    run=cur.fetchone()
    if not run:raise ValueError('The originating report is unavailable.')
    current={'kind':'onepager',**baseline(dict(run),dict(assignment),(assignment.get('result') or {}).get('visualFundamentals'),parent)}
    if current!=origin:raise ValueError('The originating one-pager changed. Reload its versions before starting a refresh.')
