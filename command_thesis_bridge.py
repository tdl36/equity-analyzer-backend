"""Resolve a completed command to its exact recap inputs, never ticker guesses."""
import base64
import hashlib
import json
import re
import uuid


def obj(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def file_hash(document):
    raw = document.get('file_data') or ''
    if len(raw) > 80_000_000:
        raise ValueError('Source exceeds comparison verification limit')
    return hashlib.sha256(base64.b64decode(raw, validate=True)).hexdigest()


def match_sources(sources, documents):
    ready, blocked = [], []
    names = [s.get('filename') for s in sources if isinstance(s, dict)]
    for source in sources:
        name = source.get('filename') if isinstance(source, dict) else None
        digest = (source.get('originalSha256') or source.get('sha256', '')) if isinstance(source, dict) else ''
        reason = None
        matches = [d for d in documents if d['filename'] == name]
        if not name or names.count(name) != 1 or not re.fullmatch(r'[a-f0-9]{64}', digest):
            reason = 'Source identity is missing or ambiguous'
        elif len(matches) != 1:
            reason = 'Import the original source into Charlie; iCloud inventory alone is insufficient'
        else:
            try:
                if file_hash(matches[0]) != digest:
                    reason = 'Imported bytes differ from recap input; transformed text needs a separate verified comparison'
            except (ValueError, TypeError):
                reason = 'Stored source could not be verified'
        if reason:
            blocked.append({'filename': name or 'Unknown source', 'reason': reason})
        else:
            ready.append({'filename': name, 'sha256': digest})
    return ready, blocked


def resolve(cur, command_id, ticker=None):
    uuid.UUID(command_id)
    cur.execute("SELECT ticker,input,result FROM mp_jobs WHERE id=%s AND stage='collection_control'", (command_id,))
    command = cur.fetchone()
    if not command or obj(command['input']).get('action') != 'research_task':
        raise ValueError('Research command not found')
    if ticker and command['ticker'] != ticker:
        raise ValueError('Command belongs to another ticker')
    topic = obj(command['result']).get('topic')
    if not topic:
        raise ValueError('Collection has not created the command event folder yet')
    cur.execute("SELECT id,output FROM analyst_activities WHERE ticker=%s AND input->>'topic'=%s ORDER BY created_at DESC LIMIT 1", (command['ticker'], topic))
    activity = cur.fetchone()
    output = obj(activity['output']) if activity else {}
    if not output.get('synthesisMarkdown'):
        raise ValueError('The latest linked analyst activity has no completed recap yet')
    sources = obj(output.get('evidenceSnapshot')).get('sources', [])
    if not isinstance(sources, list) or not 1 <= len(sources) <= 100:
        raise ValueError('The recap has no bounded input source register; rerun with source tracking')
    names = [s.get('filename') for s in sources if isinstance(s, dict) and isinstance(s.get('filename'), str)]
    cur.execute('SELECT filename,file_data FROM document_files WHERE ticker=%s AND filename=ANY(%s)', (command['ticker'], names))
    ready, blocked = match_sources(sources, list(cur.fetchall() or []))
    identity = {'commandId': command_id, 'activityId': str(activity['id']), 'sources': sources}
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {'commandId': command_id, 'activityId': str(activity['id']), 'ticker': command['ticker'],
            'topic': topic, 'revision': revision, 'ready': ready, 'blocked': blocked,
            'instructions': obj(command['input']).get('payload', {}).get('instruction', '')}
