#!/usr/bin/env python3
"""Read a saved pack and report mechanical quality diagnostics without model work."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from meeting_pack_quality import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--meeting',type=int,required=True);args=p.parse_args()
    import requests
    from charlie_local_agent import CHARLIE_API,_agent_headers
    r=requests.get(CHARLIE_API+'/api/mp/meetings/'+str(args.meeting),headers=_agent_headers(),timeout=30);r.raise_for_status();d=r.json()
    q=d.get('questionSet')
    if not q:raise ValueError('No saved question pack. Running is not ready.')
    result=inspect(q.get('topics',[]),[doc['filename'] for doc in d['documents']])
    result.update(meetingId=args.meeting,ticker=d['meeting']['ticker'],version=q['version'],questionSetId=q['id'])
    print(json.dumps(result,indent=2))
    return 2 if result['unknownCitations'] or result['incompleteQuestions'] else 0
if __name__=='__main__':raise SystemExit(main())
