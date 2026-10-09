"""Durable, bounded import of collector-approved STOCKS originals into Charlie."""
import base64
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

MAX_BYTES = 60_000_000


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS original_imports (
        document TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'queued',
        attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
        owner TEXT, lease_until REAL, issue TEXT, receipt TEXT, updated REAL NOT NULL)''')


def enqueue(collector, document):
    """Caller holds collector transaction. No historic scan or model calls."""
    row = collector.db.execute('SELECT d.*,r.topic FROM documents d JOIN runs r ON r.id=d.run WHERE d.id=?', (document,)).fetchone()
    if not row or row['topic'] or row['usage'] != 'research' or row['status'] not in ('handed_off','duplicate'):
        return False
    collector.db.execute('INSERT OR IGNORE INTO original_imports(document,updated) VALUES(?,?)', (document,time.time()))
    return True


def snapshot(collector):
    return [dict(r) for r in collector.db.execute('''SELECT d.ticker,d.run,d.id AS document,
        d.filename,q.status,q.attempts,q.issue,q.updated FROM original_imports q
        JOIN documents d ON d.id=q.document ORDER BY q.updated DESC LIMIT 100''')]


def payload(collector, row):
    if row['topic'] or row['usage'] != 'research' or row['status'] not in ('handed_off','duplicate'):
        raise ValueError('Original is no longer eligible for automatic import.')
    db=collector.db
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='refresh_requests'").fetchone():
        request=db.execute('SELECT status,config FROM refresh_requests WHERE run=?',(row['run'],)).fetchone()
        if request:
            if request['status']=='cancelled':raise ValueError('Collection was cancelled; original retained locally.')
            cfg=json.loads(request['config'])
            policy=db.execute('SELECT config FROM refresh_policies WHERE ticker=?',(row['ticker'],)).fetchone()
            if policy and not json.loads(policy['config']).get('enabled') and not cfg.get('manual'):
                raise ValueError('Collection policy is paused; resume it to import originals.')
    from source_selection_gate import check
    check(db,row['run'],row['ticker'],row['kind'],row['publisher'],row['source_url'],analyst=row['analyst'],author_evidence=row['author_evidence'])
    root=collector.stocks/row['ticker'];path=Path(row['destination'] or '')
    if (root.is_symlink() or path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve()) or not root.resolve().is_relative_to(collector.stocks.resolve())):
        raise ValueError('Original is missing or outside its confirmed company folder.')
    if path.stat().st_size>MAX_BYTES:raise ValueError('Original exceeds the 60 MB cloud import limit.')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=row['sha256']:raise ValueError('Original changed after collection; inspect it before retrying.')
    return dict(ticker=row['ticker'],collectionRun=row['run'],documentId=row['id'],filename=path.name,
        fileData=base64.b64encode(raw).decode('ascii'),sha256=digest,usage='research',
        sourceUrl=row['source_url'],publisher=row['publisher'],published=row['published'])


def sync(collector=None, post=None, limit=3, clock=time.time, only_run=None):
    """Retry exact-byte imports after restart/timeouts; never retry research generation."""
    from charlie_collector import Collector
    own=collector is None;c=collector or Collector();results=[]
    if post is None:
        import requests
        from charlie_local_agent import CHARLIE_API,_agent_headers
        def post(body):
            response=requests.post(CHARLIE_API+'/api/agent/collection-original',json=body,
                headers=_agent_headers(),timeout=45,allow_redirects=False)
            if response.status_code!=200:
                raise ValueError('Cloud import returned HTTP %s; original retained, automatic retry pending.' % response.status_code)
            return response.json()
    try:
        for _ in range(min(max(limit,0),10)):
            owner=uuid.uuid4().hex;now=clock()
            with c.lock():
                q=c.db.execute("SELECT * FROM original_imports WHERE status!='imported' AND next_attempt<=? AND COALESCE(lease_until,0)<=? AND (? IS NULL OR document IN (SELECT id FROM documents WHERE run=?)) ORDER BY next_attempt,updated LIMIT 1",(now,now,only_run,only_run)).fetchone()
                if not q:break
                c.db.execute("UPDATE original_imports SET status='uploading',owner=?,lease_until=?,attempts=attempts+1,updated=? WHERE document=?",(owner,now+180,now,q['document']))
                row=c.db.execute('SELECT d.*,r.topic FROM documents d JOIN runs r ON r.id=d.run WHERE d.id=?',(q['document'],)).fetchone()
            try:
                body=payload(c,row);receipt=post(body)
                if (receipt.get('imported') is not True or receipt.get('filename')!=body['filename']
                        or receipt.get('sha256')!=body['sha256']):
                    raise ValueError('Cloud receipt did not confirm the exact original; automatic verification retry pending.')
                receipt={k:receipt[k] for k in ('imported','filename','sha256')}
                status,issue='imported',None
            except Exception as exc:
                # Only our bounded actionable messages, never provider bodies or credentials.
                status='attention';issue=str(exc) if isinstance(exc,ValueError) and str(exc).startswith(('Original ','Collection ','Cloud ')) else 'Original import unavailable; check Mac/network/source permissions. Automatic retry pending.'
                receipt=None
            with c.lock():
                c.db.execute('''UPDATE original_imports SET status=?,issue=?,receipt=?,owner=NULL,
                    lease_until=NULL,next_attempt=?,updated=? WHERE document=? AND owner=?''',
                    (status,issue,json.dumps(receipt) if receipt else None,clock()+min(3600,60*2**min(q['attempts'],6)),clock(),q['document'],owner))
            results.append({'document':q['document'],'status':status,'issue':issue})
        return results
    finally:
        if own:c.db.close()


def validate(data):
    if not isinstance(data,dict):raise ValueError('Expected an original object')
    if not isinstance(data.get('ticker'),str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}',data['ticker']):raise ValueError('Confirmed ticker required')
    for key,n in [('collectionRun',12),('documentId',16)]:
        if not isinstance(data.get(key),str) or not re.fullmatch('[0-9a-f]{%s}'%n,data[key]):raise ValueError('Collection identity required')
    from charlie_collector import source_url
    source_url(data.get('sourceUrl',''))
    if not str(data.get('filename','')).lower().endswith('.pdf'):raise ValueError('Expected an original PDF')
    from command_source_import import validate_original
    return validate_original(data)


def create_blueprint(get_db,is_agent):
    from flask import Blueprint,jsonify,request
    bp=Blueprint('collection_original_sync',__name__)
    @bp.post('/api/agent/collection-original')
    def receive():
        if not is_agent():return jsonify(error='Local agent authentication required.'),403
        data=request.get_json(silent=True)
        try:raw,digest=validate(data)
        except (ValueError,TypeError,AttributeError):return jsonify(error='Invalid collection original, usage or hash.'),400
        tk,name=data['ticker'],data['filename']
        with get_db(commit=True) as (_,cur):
            # Share the lock namespace with command imports and preserve existing bytes.
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('command-source:'+tk+':'+name,))
            cur.execute('SELECT file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR UPDATE',(tk,name));old=cur.fetchone()
            if old:
                try:
                    same=hashlib.sha256(base64.b64decode(old['file_data'],validate=True)).hexdigest()==digest
                    meta=json.loads(old['metadata']) if isinstance(old['metadata'],str) else old['metadata'] or {}
                except (ValueError,TypeError):same=False;meta={}
                if not same:return jsonify(error='Different original already stored under this name; retained existing bytes.'),409
                from company_research import eligible
                if not eligible(meta):return jsonify(error='Existing original is restricted; restriction retained.'),409
            else:
                meta={k:data.get(k) for k in ('collectionRun','documentId','sourceUrl','publisher','published')}
                meta.update(source='alphasense_collection',sha256=digest,usage='research')
                cur.execute('''INSERT INTO document_files(ticker,filename,file_data,file_type,mime_type,file_size,metadata)
                    VALUES(%s,%s,%s,'pdf','application/pdf',%s,%s::jsonb)
                    ON CONFLICT(ticker,filename) DO NOTHING RETURNING id''',(tk,name,data['fileData'],len(raw),json.dumps(meta)))
                if not cur.fetchone():return jsonify(error='Concurrent import; retry to verify original.'),409
        return jsonify(imported=True,filename=name,sha256=digest)
    return bp
