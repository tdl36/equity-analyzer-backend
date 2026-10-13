"""Private resumable pending-editorial handoff; no model calls or approvals."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import uuid
import requests
from thesis_handoff import secret, save, read
from onepager_corrections import digest, submission_evidence

BASE='https://equity-analyzer-backend.onrender.com'
KEYS=('assignmentId','runId','ticker','fingerprint','fundamentalsDigest','parentId')
CLAIM_KEYS={'path','statement','basis','reason','sourceId','excerpt','locator'}


def request(path,body=None):
    if not re.fullmatch(r'/api/research/(?:assignments/[a-f0-9-]{36}/onepager-editions|stock-analysis-run/[a-f0-9-]{36}/corrections)',path):
        raise ValueError('Unsupported editorial route.')
    try:
        r=requests.request('POST' if body is not None else 'GET',BASE+path,json=body,
                           headers={'Authorization':'ApiKey '+secret()},timeout=45,allow_redirects=False)
        if not r.ok or r.is_redirect:
            raise ValueError(f'Editorial handoff returned HTTP {r.status_code}; keep this workspace and inspect the saved review before retrying.')
        return r.json()
    except (requests.RequestException,json.JSONDecodeError):
        raise ValueError('Editorial handoff unavailable; retry the same workspace. Submission identities are retained.') from None


def workspace(path):
    p=Path(path).expanduser().resolve()
    blocked=[Path(__file__).resolve().parent,Path.home()/'Library/Mobile Documents',Path.home()/'Library/CloudStorage']
    if any(p==b or b in p.parents for b in blocked):
        raise ValueError('Use a private local workspace outside the repository and watched/cloud source folders.')
    p.mkdir(parents=True,exist_ok=True,mode=0o700)
    p.chmod(0o700)
    return p


def routes(pack):
    return ('/api/research/assignments/'+pack['assignmentId']+'/onepager-editions',
            '/api/research/stock-analysis-run/'+pack['runId']+'/corrections')


def prepare(assignment,path,call=request):
    assignment=str(uuid.UUID(assignment));p=workspace(path)
    with (p/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (p/'preparation.json').exists():raise ValueError('Workspace already prepared; retain it or choose a new directory.')
        pack=call('/api/research/assignments/'+assignment+'/onepager-editions')
        if pack['assignmentId']!=assignment:raise ValueError('Assignment identity changed.')
        pack={k:pack[k] for k in KEYS}
        claims=call(routes(pack)[1])
        if claims.get('sourcePolicyReady') is not True:
            raise ValueError('Frozen source pack has missing or restricted originals, or policy preflight is unavailable. Resolve the source pack before editorial preparation.')
        if any(claims[k]!=pack[k] for k in ('runId','ticker','fingerprint')):raise ValueError('Report changed during preparation; prepare again.')
        # Original inspection is a separate human/assistant step, never inferred from inventory.
        save(p/'preparation.json',{'baseline':pack,'fields':claims['fields'],'sources':claims['sources']})
        save(p/'proposal.json',{'claims':[]})
    return {'workspace':str(p),'ticker':pack['ticker'],'permittedClaims':len(claims['fields']),
            'next':'Inspect frozen originals, then fill proposal.json with source-linked replacement claims. No approval is performed.'}


def submit(path,call=request):
    p=workspace(path)
    with (p/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prep=read(p/'preparation.json');pack=prep['baseline'];proposal=read(p/'proposal.json')
        claims=proposal.get('claims') if isinstance(proposal,dict) and set(proposal)=={'claims'} else None
        if not isinstance(claims,list) or not 1<=len(claims)<=60:raise ValueError('Supply 1–60 editorial claims.')
        paths={f['path'] for f in prep['fields']};sources={s['id'] for s in prep['sources']};seen=set()
        for c in claims:
            if not isinstance(c,dict) or set(c)-{'additionalEvidence'}!=CLAIM_KEYS:raise ValueError('Unknown or missing claim fields; review flags are forbidden.')
            if not isinstance(c['path'],str) or c['path'] not in paths or c['path'] in seen or any(e['sourceId'] not in sources for e in submission_evidence(c)):
                raise ValueError('Choose distinct permitted fields and frozen source IDs.')
            seen.add(c['path'])
        signature=digest({'preparation':prep,'proposal':proposal})
        journal=p/'submission.json'
        if journal.exists():
            state=read(journal)
            if state['digest']!=signature:raise ValueError('Submitted content changed. Retain this workspace and prepare a new proposal.')
        else:
            state={'digest':signature,'claims':[{**c,'fingerprint':pack['fingerprint'],'requestId':str(uuid.uuid4())} for c in claims],
                   'editionId':str(uuid.uuid4()),'receipts':[]}
            save(journal,state)  # Before any write: lost responses retain exact replay identities.
        edition_route,claim_route=routes(pack)
        if not state.get('editionSubmitted'):
            current=call(edition_route)
            if any(current.get(k)!=pack[k] for k in KEYS):raise ValueError('Frozen report, financials or parent changed. Prepare a new proposal; prior pending corrections are retained.')
            policy=call(claim_route)
            if policy.get('sourcePolicyReady') is not True:
                raise ValueError('Frozen source pack has missing or restricted originals. Pending receipts are retained; no new claims were submitted.')
            for c in state['claims']:
                if c['requestId'] in state['receipts']:continue
                receipt=call(claim_route,c)
                if receipt.get('id')!=c['requestId'] or receipt.get('status')!='pending':raise ValueError('Correction receipt mismatch; inspect before retrying.')
                state['receipts'].append(c['requestId']);save(journal,state)
            state['editionSubmitted']=True;save(journal,state)
        # If its response was lost, replay directly: the server resolves the prior
        # edition before checking a parent that may since have advanced.
        body={k:pack[k] for k in KEYS if k!='ticker'}
        body.update(requestId=state['editionId'],correctionIds=[c['requestId'] for c in state['claims']])
        receipt=call(edition_route,body)
        if receipt.get('id')!=state['editionId'] or receipt.get('status') not in ('pending','accepted','rejected'):
            raise ValueError('Edition receipt mismatch; inspect before retrying.')
        state['editionReceipt']=receipt;save(journal,state)
        return {'editionId':state['editionId'],'status':receipt['status'],
                'reviewUrl':'https://charlie-deployment.tonydlee.workers.dev/#view=stockanalysis&ticker='+pack['ticker']+'&section=stock-analysis&work=assignment%3A'+pack['assignmentId'],
                'message':'Edition saved for explicit investor review; original research and thesis are unchanged.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--assignment',required=True);p.add_argument('--workspace',required=True)
    p=sub.add_parser('submit');p.add_argument('--workspace',required=True)
    args=parser.parse_args()
    try:result=prepare(args.assignment,args.workspace) if args.command=='prepare' else submit(args.workspace)
    except ValueError as exc:
        parser.exit(1,str(exc)+'\n')
    except (KeyError,TypeError,OSError):
        parser.exit(1,'Handoff stopped. Retain the private workspace; check its schema, frozen baseline and saved review before retrying. No approval was performed.\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
