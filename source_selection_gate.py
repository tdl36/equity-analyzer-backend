"""Fail-closed source selection before any managed broker export is staged."""
import json
from source_preferences import decision

def check(db,run,ticker,kind,publisher,url,fetch=None):
    if kind!='broker-report':return
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='refresh_requests'").fetchone():return
    row=db.execute('SELECT config FROM refresh_requests WHERE run=?',(run,)).fetchone()
    if not row:return
    cfg=json.loads(row['config']);p=cfg.get('researchCommand') or {};policy=p.get('sourcePolicy')
    if policy is None:return
    outcome=decision(policy,publisher,ticker,'meeting' if p.get('meetingPrep') else p.get('kind','event'))
    if outcome in ('excluded','reference_only'):raise ValueError('Source preference excludes this broker from AI collection. Record the exclusion; do not stage or hand off it.')
    if outcome=='include':return
    cid=cfg.get('eventId')
    if not cid:raise ValueError('Source review requires a managed command identity.')
    if fetch is None:
        import requests
        import charlie_local_agent as a
        def fetch(cid):
            r=requests.get(a.CHARLIE_API+'/api/research/commands/'+cid+'/source-shortlist',headers=a._agent_headers(),timeout=20)
            r.raise_for_status();return r.json()
    value=fetch(cid)
    found=next((x for x in value.get('candidates',[]) if x['url']==url and x['publisher'].strip().casefold()==(publisher or '').strip().casefold()),None)
    if not found or found.get('decision')!='include':raise ValueError('Source selection pending or excluded. Submit the observed document to the Command Charlie shortlist and wait for an explicit include decision.')
