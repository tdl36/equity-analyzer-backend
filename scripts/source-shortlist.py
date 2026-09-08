#!/usr/bin/env python3
"""Submit observed source metadata for user selection; never downloads documents."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
import charlie_local_agent as agent
p=argparse.ArgumentParser();p.add_argument('--command',required=True);p.add_argument('--file');p.add_argument('--seal',action='store_true');args=p.parse_args()
url=agent.CHARLIE_API+'/api/research/commands/'+args.command+'/source-shortlist'
if args.file:
    value=json.loads(Path(args.file).read_text())
    r=requests.post(url,headers=agent._agent_headers(),json={'candidates':value,'sealed':args.seal},timeout=30)
else:r=requests.get(url,headers=agent._agent_headers(),timeout=30)
r.raise_for_status();print(json.dumps(r.json(),indent=2))
