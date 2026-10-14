"""Durable Lab iCloud import transactions shared by user and Mac-agent routes.

Call each mutation inside get_db(commit=True); allow ValueError to escape the
transaction so partial originals and settings roll back together. No research runs.
"""
import json
import mimetypes
import re
import uuid
from pipeline_lab_local import identity
from pipeline_lab_import import digest
from company_research import eligible

PREFIX='pipeline_lab_icloud:'


def load(cur,key):
    cur.execute('SELECT value FROM app_settings WHERE key=%s',(key,))
    row=cur.fetchone()
    return (json.loads(row['value']) if isinstance(row['value'],str) else row['value']) if row else None


def lock(cur,key):
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(key,))


def put(cur,key,value):
    cur.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW()) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()', (key,json.dumps(value)))


def job_key(request_id):
    try:return PREFIX+str(uuid.UUID(request_id))
    except (ValueError,TypeError,AttributeError):raise ValueError('Valid import identity required.')


def enqueue(cur,ticker,task,revision,request_id):
    if (not isinstance(ticker,str) or not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',ticker)
            or task not in ('thesis','note') or type(revision) is not int or revision<1):
        raise ValueError('Saved ticker, task and revision required.')
    key=job_key(request_id);lock(cur,key)
    signature={'ticker':ticker,'task':task,'revision':revision}
    old=load(cur,key)
    if old:
        if old['input']!=signature:raise ValueError('Import identity already used with different settings.')
        return old
    config_key='pipeline_lab_v1:'+ticker;lock(cur,config_key);cfg=load(cur,config_key)
    if not cfg or cfg['revision']!=revision:raise ValueError('Lab settings changed. Reload before importing.')
    from pipeline_lab import config
    selected=config(cfg['selection'])[task]
    refs=[r for r in selected if r.startswith('icloud:')]
    if not refs or len(selected)>8:raise ValueError('Save up to eight selected sources, including an iCloud original.')
    for ref in refs:identity(ref)
    # Prevent a second request ID from creating competing imports for this ticker.
    cur.execute('SELECT value FROM app_settings WHERE key LIKE %s',(PREFIX+'%',))
    for row in cur.fetchall():
        other=json.loads(row['value']) if isinstance(row['value'],str) else row['value']
        if other['input']['ticker']==ticker and other['status'] in ('queued','frozen','attention'):
            raise ValueError('An iCloud import is already pending for this stock. Resume that request.')
    job={'requestId':key[len(PREFIX):],'input':signature,'references':refs,'status':'queued','researchStarted':False}
    put(cur,key,job);return job


def freeze(cur,request_id,manifest):
    key=job_key(request_id);lock(cur,key);job=load(cur,key)
    if not job:raise ValueError('Import request not found.')
    if not isinstance(manifest,list) or len(manifest)!=len(job['references']):raise ValueError('Exact original manifest required.')
    normalized=[]
    for item in manifest:
        if not isinstance(item,dict) or set(item)!={'reference','filename','sha256','bytes'}:raise ValueError('Invalid original manifest.')
        _,name=identity(item['reference'])
        if (item['filename']!=name or not isinstance(item['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',item['sha256'])
                or type(item['bytes']) is not int or not 0<item['bytes']<=15_000_000):raise ValueError('Invalid original identity, size or hash.')
        normalized.append(dict(item))
    normalized.sort(key=lambda r:r['reference'])
    if (sorted(r['reference'] for r in normalized)!=sorted(job['references'])
            or sum(r['bytes'] for r in normalized)>60_000_000
            or len({r['filename'] for r in normalized})!=len(normalized)):
        raise ValueError('Manifest has missing identities, duplicate filenames or exceeds the pack limit.')
    if 'manifest' in job:
        if job['manifest']!=normalized:raise ValueError('Frozen original manifest changed. No substitution is allowed.')
        return job
    if job['status']!='queued':raise ValueError('Import is no longer pending.')
    job={**job,'manifest':normalized,'status':'frozen'};put(cur,key,job);return job


def complete(cur,request_id,originals):
    key=job_key(request_id);lock(cur,key);job=load(cur,key)
    if not job or job['status'] not in ('frozen','complete'):raise ValueError('Freeze exact originals before importing.')
    # A receipt replay is read-only and never changes a later configuration.
    if job['status']=='complete':return job
    if not isinstance(originals,list) or len(originals)!=len(job['manifest']):raise ValueError('Every frozen original is required.')
    encoded={}
    for item in originals:
        if not isinstance(item,dict) or set(item)!={'reference','fileData'} or not isinstance(item['reference'],str) or item['reference'] in encoded:
            raise ValueError('Distinct original bodies required.')
        encoded[item['reference']]=item['fileData']
    if set(encoded)!=set(job['references']):raise ValueError('Original identities differ from the frozen pack.')
    for item in job['manifest']:
        sha,size=digest(encoded[item['reference']])
        if sha!=item['sha256'] or size!=item['bytes']:raise ValueError('Original bytes differ from the frozen pack.')
    tk=job['input']['ticker'];cfgkey='pipeline_lab_v1:'+tk;lock(cur,cfgkey);cfg=load(cur,cfgkey)
    if not cfg or cfg['revision']!=job['input']['revision']:raise ValueError('Lab settings changed; originals were not imported. Review the pending request.')
    receipts=[];mapping={}
    for item in job['manifest']:
        name=item['filename'];metadata={'source':'pipeline_lab_icloud','requestId':job['requestId'],'reference':item['reference'],'sha256':item['sha256'],'usage':'research'}
        cur.execute('''INSERT INTO document_files(ticker,filename,file_data,file_type,mime_type,file_size,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT(ticker,filename) DO NOTHING''',
            (tk,name,encoded[item['reference']],name.rsplit('.',1)[-1].lower(),mimetypes.guess_type(name)[0] or 'application/octet-stream',item['bytes'],json.dumps(metadata)))
        cur.execute('SELECT id,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',(tk,name));stored=cur.fetchone()
        if not stored or digest(stored['file_data'])!=(item['sha256'],item['bytes']):raise ValueError('Different original uses this filename; all existing bytes were preserved.')
        if not eligible(stored.get('metadata') or {}):raise ValueError('Matching original is excluded or reference-only.')
        target='charlie:thesis_'+str(stored['id']);mapping[item['reference']]=target
        receipts.append({**item,'target':target})
    task=job['input']['task'];selection=cfg['selection'];selection[task]=sorted(set(mapping.get(r,r) for r in selection[task]))
    put(cur,cfgkey,{'revision':cfg['revision']+1,'selection':selection})
    job={**job,'status':'complete','receipts':receipts,'completedRevision':cfg['revision']+1};put(cur,key,job)
    return job


def cancel(cur,request_id):
    key=job_key(request_id);lock(cur,key);job=load(cur,key)
    if not job:raise ValueError('Import request not found.')
    if job['status']=='complete':return job
    job={**job,'status':'cancelled'};put(cur,key,job);return job

ISSUES={
    'source_unavailable':'Original unavailable or changed at its exact iCloud path. Check download status, file identity and Mac permissions, then retry or cancel.',
    'import_unavailable':'Import connection or receipt unavailable. Retry to reconcile the saved request; existing originals remain preserved.',
    'settings_conflict':'Saved settings or stored originals conflict with this request. Review the sources and cancel this request before starting a new import.',
}

def attention(cur,request_id,code):
    if not isinstance(code,str) or code not in ISSUES:raise ValueError('Unknown import issue.')
    key=job_key(request_id);lock(cur,key);job=load(cur,key)
    if not job:raise ValueError('Import request not found.')
    if job['status'] in ('complete','cancelled'):return job
    job={**job,'status':'attention','issueCode':code,'issue':ISSUES[code]};put(cur,key,job);return job


def retry(cur,request_id):
    key=job_key(request_id);lock(cur,key);job=load(cur,key)
    if not job:raise ValueError('Import request not found.')
    if job['status']!='attention':return job
    job={k:v for k,v in job.items() if k not in ('issue','issueCode')}
    job['status']='frozen' if 'manifest' in job else 'queued';put(cur,key,job);return job
