#!/usr/bin/env python3
"""Check an actual assignment without dispatching work or calling a model."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from charlie_collector import DEFAULT_STATE
from meeting_assignment_verification import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command',required=True,type=lambda v:str(uuid.UUID(v)))
    parser.add_argument('--refresh',required=True,type=lambda v:str(uuid.UUID(v)))
    parser.add_argument('--recap-job',required=True,type=lambda v:str(uuid.UUID(v)))
    args=parser.parse_args()
    import requests
    from charlie_local_agent import CHARLIE_API,_agent_headers
    def get(path):
        response=requests.get(CHARLIE_API+path,headers=_agent_headers(),timeout=30)
        response.raise_for_status();return response.json()
    with sqlite3.connect((DEFAULT_STATE/'ledger.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT status,config,result FROM refresh_requests WHERE id=?',(args.refresh,)).fetchone()
        if not row or json.loads(row['config']).get('eventId')!=args.command:
            raise ValueError('Local refresh must belong to this exact command')
        collection={'status':row['status'],'result':json.loads(row['result'] or '{}')}
    matches=[r for r in get('/api/research/meeting-commands')['jobs'] if r['id']==args.command]
    if len(matches)!=1:raise ValueError('Command unavailable in recent meeting assignments')
    command=matches[0]
    recap=get('/api/catalysts/results/'+args.recap_job)
    meeting=get('/api/mp/meetings/'+str(command['meeting_id'])) if command.get('meeting_id') else {}
    job=get('/api/mp/jobs/'+command['job_id']) if command.get('job_id') else {}
    result=verify(command,collection,recap,meeting,job)
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 2

if __name__=='__main__':
    raise SystemExit(main())
