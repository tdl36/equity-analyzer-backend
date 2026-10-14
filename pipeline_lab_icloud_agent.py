"""One bounded Lab import per poll. No models, collection, hydration or file writes.

Injected transport uses existing Mac credentials. A server-frozen manifest survives
lost responses/restarts; retries reread only matching hashes. Not wired to agent yet.
"""
from pipeline_lab_local import read_original


def process_one(get,post,stocks,catalysts):
    pending=get('/api/agent/pipeline-lab-imports')
    rows=pending.get('imports') if isinstance(pending,dict) else None
    if not isinstance(rows,list):raise ValueError('Lab import status unavailable.')
    if not rows:return None
    request_id=rows[0]['requestId']
    job=get('/api/pipeline-lab/icloud-import/'+request_id)
    if job.get('status') not in ('queued','frozen'):return job.get('status')
    ticker=job['input']['ticker'];originals=[]
    frozen={r['reference']:r for r in job.get('manifest',[])}
    if job['status']=='frozen' and set(frozen)!=set(job['references']):raise ValueError('Frozen Lab import identities unavailable.')
    total=0
    for ref in job['references']:
        data=read_original(stocks,catalysts,ticker,ref,expected_sha256=frozen[ref]['sha256'] if frozen else None)
        total+=data['bytes']
        if total>60_000_000:raise ValueError('Selected originals exceed the 60 MB Lab limit.')
        originals.append(data)
    manifest=sorted([{k:r[k] for k in ('reference','filename','sha256','bytes')} for r in originals],key=lambda r:r['reference'])
    if job['status']=='queued':
        # Persist identities before the first body upload. An ambiguous response
        # stops this poll; the next poll reconciles server state before rereading.
        job=post('/api/agent/pipeline-lab-import/'+request_id+'/freeze',{'manifest':manifest})
    if job.get('status')!='frozen' or job.get('manifest')!=manifest:
        raise ValueError('Frozen Lab import receipt differs from the selected originals.')
    result=post('/api/agent/pipeline-lab-import/'+request_id+'/complete',{'originals':[{k:r[k] for k in ('reference','fileData')} for r in originals]})
    expected={r['reference']:(r['filename'],r['sha256'],r['bytes']) for r in manifest}
    receipts=result.get('receipts',[]) if isinstance(result,dict) else []
    if (not isinstance(result,dict) or result.get('requestId')!=request_id or result.get('status')!='complete' or len(receipts)!=len(expected)
            or {r['reference']:(r['filename'],r['sha256'],r['bytes']) for r in receipts}!=expected):
        raise ValueError('Exact Lab import completion receipt unavailable. Reconcile on the next poll.')
    return 'complete'
