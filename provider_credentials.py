"""Shared provider credentials. Never return secrets to browser Settings."""
import os
import requests
from flask import Blueprint, jsonify, request, Response, stream_with_context

FIELDS = {'anthropic': ('apiKey', ('ANTHROPIC_API_KEY',)),
          'gemini': ('geminiApiKey', ('GEMINI_API_KEY', 'GOOGLE_API_KEY')),
          'openai': ('openaiApiKey', ('OPENAI_API_KEY',))}
SECRET_FIELDS = {v[0] for v in FIELDS.values()}
_db = None

def configure(get_db):
    global _db
    _db = get_db

def get(provider, get_db=None):
    field, names = FIELDS[provider]
    db = get_db or _db
    # Legacy browser-synced values were not validated. Preserve deployment
    # credentials until an explicit validated save selects the shared value.
    saved = ''
    preferred = False
    if db:
        with db() as (_, cur):
            cur.execute('SELECT value FROM app_settings WHERE key=%s', (field,))
            row = cur.fetchone()
            if row and isinstance(row.get('value'), str): saved = row['value'].strip()
            cur.execute('SELECT value FROM app_settings WHERE key=%s', ('_provider_preference:'+provider,))
            preference = cur.fetchone()
            preferred = bool(preference and preference.get('value') == 'validated_saved')
    if saved and preferred: return saved
    return next((os.environ[n].strip() for n in names if os.environ.get(n, '').strip()), '') or saved

def status():
    return {p: {'configured': bool(get(p))} for p in FIELDS}

def validate(provider, key):
    headers = {'x-api-key': key, 'anthropic-version': '2023-06-01'} if provider == 'anthropic' else ({'x-goog-api-key': key} if provider == 'gemini' else {'Authorization': 'Bearer '+key})
    url = {'anthropic':'https://api.anthropic.com/v1/models', 'gemini':'https://generativelanguage.googleapis.com/v1beta/models', 'openai':'https://api.openai.com/v1/models'}[provider]
    try:
        response = requests.get(url, headers=headers, timeout=20)
        return response.status_code
    except requests.RequestException:
        return None

def create_blueprint(get_db, is_agent):
    configure(get_db)
    bp = Blueprint('provider_credentials', __name__)
    @bp.after_request
    def private(response):
        response.headers['Cache-Control'] = 'private, no-store'
        return response
    @bp.get('/api/provider-credentials')
    def inspect():
        return jsonify(providers=status())
    @bp.put('/api/provider-credentials/<provider>')
    def save(provider):
        if provider not in FIELDS: return jsonify(error='Unknown provider.'), 404
        data = request.get_json(silent=True) or {}
        key = data.get('key')
        if not isinstance(key,str) or not 10 <= len(key.strip()) <= 1000 or any(c.isspace() for c in key.strip()):
            return jsonify(error='Enter a complete provider API key.'), 400
        key = key.strip()
        code = validate(provider,key)
        if code != 200:
            return jsonify(error='The provider could not validate this key. The existing shared key was kept.', providerStatus=code), 422
        with get_db(commit=True) as (_,cur):
            cur.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW()) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()', (FIELDS[provider][0],key))
            cur.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW()) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()', ('_provider_preference:'+provider,'validated_saved'))
        return jsonify(provider=provider, configured=True)
    @bp.get('/api/agent/provider-credentials/<provider>')
    def agent_key(provider):
        if not is_agent(): return jsonify(error='Local agent authentication required.'),403
        if provider not in FIELDS: return jsonify(error='Unknown provider.'),404
        return jsonify(key=get(provider))
    @bp.post('/api/provider/anthropic/messages')
    def messages():
        # Existing browser analysis now streams through authenticated Charlie.
        # No retries or credential fallback after an uncertain paid request.
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or data.get('stream') is not True:
            return jsonify(error='A streaming analysis request is required.'),400
        data = {k:v for k,v in data.items() if k not in SECRET_FIELDS}
        key=get('anthropic')
        if not key:return jsonify(error={'message':'Configure Claude in Settings → Connections.'}),422
        try:
            upstream=requests.post('https://api.anthropic.com/v1/messages',json=data,
                headers={'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'},
                stream=True,timeout=(15,300))
        except requests.RequestException:
            return jsonify(error={'message':'Provider connection interrupted. Check the saved outcome before retrying.'}),502
        if upstream.status_code != 200:
            code=upstream.status_code;upstream.close()
            return jsonify(error={'message':f'Provider rejected analysis (HTTP {code}). Check shared provider settings.'}),422
        def chunks():
            try:
                yield from upstream.iter_content(chunk_size=8192)
            finally: upstream.close()
        return Response(stream_with_context(chunks()),mimetype='text/event-stream',headers={'X-Accel-Buffering':'no'})
    return bp
