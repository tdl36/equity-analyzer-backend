"""Import only registered, eligible command originals; never overwrite research."""
import base64
import hashlib
import json
import mimetypes
import re
import uuid
from pathlib import Path

MAX_BYTES = 60_000_000


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('Expected a source object')
    uuid.UUID(data.get('commandId', ''))
    name = data.get('filename', '')
    if (not isinstance(name, str) or not 1 <= len(name) <= 255 or Path(name).name != name
            or name.startswith('.') or '\\' in name or Path(name).suffix.lower() not in ('.pdf','.txt','.html','.htm')):
        raise ValueError('Unsupported original filename')
    if data.get('usage') != 'research':
        raise ValueError('Restricted originals cannot be imported')
    encoded = data.get('fileData')
    if not isinstance(encoded, str) or len(encoded) > MAX_BYTES * 4 // 3 + 4:
        raise ValueError('Source exceeds the 60 MB import limit')
    raw = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if not raw or len(raw) > MAX_BYTES or data.get('sha256') != digest:
        raise ValueError('Original source hash or size verification failed')
    if name.lower().endswith('.pdf') and not raw.startswith(b'%PDF-'):
        raise ValueError('Expected a PDF original')
    url = data.get('sourceUrl', '')
    if not isinstance(url, str) or not url.startswith('https://') or len(url) > 3000:
        raise ValueError('Verified source URL required')
    return raw, digest


def create_blueprint(get_db):
    from flask import Blueprint, jsonify, request
    bp = Blueprint('command_source_import', __name__)
    @bp.route('/api/agent/command-source', methods=['POST'])
    def receive():
        data = request.get_json(silent=True)
        try:raw, digest = validate(data)
        except (ValueError,TypeError,AttributeError) as exc:return jsonify(error=str(exc)),400
        with get_db(commit=True) as (_,cur):
            cur.execute("SELECT ticker,input,result FROM mp_jobs WHERE id=%s AND stage='collection_control'", (data['commandId'],))
            command = cur.fetchone()
            def obj(value):return json.loads(value) if isinstance(value,str) else (value or {})
            if (not command or obj(command['input']).get('action') != 'research_task'
                    or obj(command['result']).get('topic') != data.get('topic') or command['ticker'] != data.get('ticker')):
                return jsonify(error='Command receipt, ticker and event folder must match before import'),409
            tk, name = command['ticker'], data['filename']
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('command-source:'+tk+':'+name,))
            cur.execute('SELECT file_data FROM document_files WHERE ticker=%s AND filename=%s FOR UPDATE', (tk,name))
            old = cur.fetchone()
            if old:
                try:same = hashlib.sha256(base64.b64decode(old['file_data'],validate=True)).hexdigest()==digest
                except (ValueError,TypeError):same=False
                if not same:return jsonify(error='A different original already uses this filename; existing document retained'),409
            else:
                metadata={'source':'managed_command','commandId':data['commandId'],'topic':data['topic'],
                          'sourceUrl':data['sourceUrl'],'sha256':digest,'usage':'research'}
                cur.execute('INSERT INTO document_files(ticker,filename,file_data,file_type,mime_type,file_size,metadata) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT(ticker,filename) DO NOTHING RETURNING id',
                    (tk,name,data['fileData'],Path(name).suffix.lower().lstrip('.'),mimetypes.guess_type(name)[0] or 'text/plain',len(raw),json.dumps(metadata)))
                if not cur.fetchone():return jsonify(error='Another import created this filename; retry to verify its contents'),409
        return jsonify(imported=True,filename=name,sha256=digest)
    return bp


def import_sources(manager, row, post=None):
    from research_task_sources import active
    cfg=json.loads(row['config']);rid=row['id'];owner=row['owner']
    if not cfg.get('researchCommand'):return 0
    root=manager.c.catalysts/cfg['ticker']/cfg['topic']
    sources=[]
    for doc in manager.c.status(row['run'])['documents']:
        if doc['usage']=='research' and doc['status']=='handed_off':
            sources.append({'path':Path(doc['destination']),'sha256':doc['sha256'],'url':doc['source_url']})
    saved=manager.db.execute('SELECT manifest FROM research_public_sources WHERE request_id=?',(rid,)).fetchone()
    if not saved or not json.loads(saved['manifest']).get('complete'):
        raise ValueError('Complete public source collection before importing command sources')
    for doc in json.loads(saved['manifest'])['documents']:
        sources.append({'path':root/doc['filename'],'sha256':doc['sha256'],'url':doc['url']})
    if len(sources)>60:raise ValueError('Command import exceeds 60 originals; narrow the collection')
    if sources and post is None:
        import requests
        from charlie_local_agent import CHARLIE_API,_agent_headers
        def post(payload):
            response=requests.post(CHARLIE_API+'/api/agent/command-source',headers=_agent_headers(),json=payload,timeout=45)
            if not response.ok:raise ValueError('Cloud source import failed (%s); preserve originals and retry collection' % response.status_code)
            return response.json()
    for source in sources:
        with manager.c.lock():
            active(manager,rid,owner)
            path=source['path']
            if (root.is_symlink() or path.is_symlink() or not path.is_file()
                    or path.parent.resolve()!=root.resolve() or not root.resolve().is_relative_to(manager.c.catalysts.resolve())):
                raise ValueError('Source is not an original in this command event folder')
            if path.stat().st_size>MAX_BYTES:raise ValueError('Original exceeds 60 MB comparison import limit')
            raw=path.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=source['sha256']:raise ValueError('Registered original changed before cloud import')
            receipt=post({'commandId':cfg['eventId'],'ticker':cfg['ticker'],'topic':cfg['topic'],
                          'filename':path.name,'fileData':base64.b64encode(raw).decode(),'sha256':digest,
                          'sourceUrl':source['url'],'usage':'research'})
            if receipt.get('imported') is not True or receipt.get('filename')!=path.name or receipt.get('sha256')!=digest:
                raise ValueError('Cloud import did not confirm the exact original; safe to retry')
    return len(sources)
