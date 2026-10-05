"""Model-free, durable external thesis drafts and transactional investor approval."""
import copy
import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, current_app
import psycopg2

SCHEMA = 'charlie.external-thesis-draft.v1'
MAX_BYTES = 2_000_000
CORE = ('thesis', 'signposts', 'threats', 'conclusion')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def ticker_value(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}', value.strip().upper()):
        raise ValueError('Enter a valid ticker (up to 20 letters, numbers, dots or hyphens).')
    return value.strip().upper()


def text(value, label, required=True, limit=40000):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f'{label} must be text' + (' and cannot be empty.' if required else '.'))
    return value.strip()


def tree(value, depth=0):
    if depth > 12:
        raise ValueError('The file is nested too deeply.')
    if isinstance(value, dict):
        if len(value) > 100:
            raise ValueError('Too many fields in a record.')
        for key, item in value.items():
            if not isinstance(key, str) or key in ('__proto__', 'constructor', 'prototype'):
                raise ValueError('Unsupported field name.')
            tree(key, depth + 1)
            tree(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 500:
            raise ValueError('A list exceeds the 500-item limit.')
        for item in value:
            tree(item, depth + 1)
    elif isinstance(value, str):
        if '\x00' in value or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise ValueError('Text contains an unsupported character.')
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError('Use JSON values only.')


def normalize(package):
    if not isinstance(package, dict):
        raise ValueError('Upload a Charlie thesis draft JSON object.')
    try:
        encoded = json.dumps(package, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError):
        raise ValueError('The draft contains invalid JSON values.')
    if len(encoded) > MAX_BYTES:
        raise ValueError('The draft exceeds 2 MB. Include references, not original document bodies.')
    tree(package)
    if package.get('schema') not in (SCHEMA, 'charlie.external-thesis-draft.proposed.v1'):
        raise ValueError('Unsupported draft format. Download Prepare for ChatGPT for the supported template.')
    a = copy.deepcopy(package.get('analysis'))
    if not isinstance(a, dict):
        raise ValueError('The draft must contain an analysis object.')
    ticker = ticker_value(package.get('ticker'))
    if ticker_value(a.get('ticker', ticker)) != ticker:
        raise ValueError('The package ticker and analysis ticker do not match.')
    company = text(package.get('companyName') or a.get('company'), 'Company name', limit=255)
    if a.get('company') and a['company'] != company:
        raise ValueError('The package and analysis company names do not match.')
    if not isinstance(a.get('thesis'), dict):
        raise ValueError('Investment thesis is required.')
    a['thesis']['summary'] = text(a['thesis'].get('summary'), 'Thesis summary')
    a['conclusion'] = text(a.get('conclusion', ''), 'Conclusion', required=False)
    for section, items, label in (
        ('pillars', a['thesis'].get('pillars'), 'title'),
        ('signposts', a.get('signposts'), 'metric'),
        ('threats', a.get('threats'), 'threat'),
    ):
        if not isinstance(items, list) or not 1 <= len(items) <= 100:
            raise ValueError(f'{section} must contain between 1 and 100 entries.')
        ids = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f'Each {section} entry must be an object.')
            item[label] = text(item.get(label), f'{section}: {label}', limit=4000)
            if section == 'pillars':
                item['description'] = text(item.get('description'), 'Pillar description')
            if section == 'signposts':
                item['target'] = text(item.get('target'), 'Signpost target')
                item['timeframe'] = text(item.get('timeframe'), 'Signpost timeframe')
            if section == 'threats':
                item['triggerPoints'] = text(item.get('triggerPoints'), 'Risk trigger points')
            ident = str(item.get('id') or f'{ticker}-{section}-{digest(item[label])[:12]}')
            if len(ident) > 200 or ident in ids:
                raise ValueError(f'Duplicate or invalid ID in {section}.')
            item['id'] = ident
            ids.add(ident)
            # Native thesis views render these as text; reject objects rather than crashing later.
            for field in ('description', 'confidence', 'category', 'likelihood', 'impact', 'baseline', 'reviewTrigger', 'thresholdBasis', 'claimBasis'):
                if field in item:
                    text(item[field], f'{section}: {field}', required=False)
            sources = item.get('sources', [])
            if not isinstance(sources, list) or len(sources) > 100 or any(not isinstance(s, dict) for s in sources):
                raise ValueError('Sources must be a list of reference objects.')
            for source in sources:
                for field in ('filename', 'sourceId', 'excerpt', 'url', 'sha256', 'supportType'):
                    if field in source:
                        text(source[field], 'Source ' + field, required=False)
                pages = source.get('pdfPages', [])
                if not isinstance(pages, list) or any(type(p) is not int or p < 1 for p in pages):
                    raise ValueError('Source PDF pages must be positive whole numbers.')
    register = package.get('sourceRegister', [])
    if not isinstance(register, list) or any(not isinstance(s, dict) for s in register):
        raise ValueError('Source register must be a list of reference objects.')
    registry = {}
    for source in register:
        ident = source.get('id') or source.get('sourceId')
        if not isinstance(ident, str) or not ident or ident in registry:
            raise ValueError('Every source register entry needs a unique id.')
        registry[ident] = source
    for item in a['thesis']['pillars'] + a['signposts'] + a['threats']:
        for s in item.get('sources', []):
            if s.get('sourceId') and register:
                entry = registry.get(s['sourceId'])
                if entry is None:
                    raise ValueError('A citation refers to a missing source register ID.')
                if s.get('sha256') and entry.get('sha256') and s['sha256'] != entry['sha256']:
                    raise ValueError('Citation and source register hashes disagree.')
                if type(entry.get('pages')) is int and any(p > entry['pages'] for p in s.get('pdfPages', [])):
                    raise ValueError('A citation exceeds the registered PDF page count.')
    history = a.get('documentHistory', [])
    if not isinstance(history, list) or any(not isinstance(h, dict) for h in history):
        raise ValueError('Document history must contain reference objects.')
    # A reference cannot claim that the original has been uploaded to Charlie.
    for document in history:
        document.pop('stored', None)
    baseline = package.get('baseline', {})
    if not isinstance(baseline, dict):
        raise ValueError('Invalid baseline.')
    if 'expectedNoExistingThesis' in baseline and type(baseline['expectedNoExistingThesis']) is not bool:
        raise ValueError('The initial-thesis flag must be true or false.')
    if baseline.get('mode') not in (None, 'initial', 'upgrade'):
        raise ValueError('Unknown baseline mode.')
    if baseline.get('hash') and (not isinstance(baseline['hash'], str) or not re.fullmatch('[a-f0-9]{64}', baseline['hash'])):
        raise ValueError('Invalid baseline hash.')
    return {'schema': SCHEMA, 'ticker': ticker, 'companyName': company,
            'analysis': {**{k: a[k] for k in CORE}, 'ticker': ticker, 'company': company, 'documentHistory': history},
            'baseline': baseline, 'sourceRegister': copy.deepcopy(register),
            'provenance': copy.deepcopy(package.get('provenance', {}))}


def row_hash(row):
    return digest({'analysis': row['analysis'], 'company': row['company'], 'updated': str(row['updated_at'])}) if row else digest(None)


def current(cur, ticker):
    cur.execute('SELECT ticker,company,analysis,updated_at FROM portfolio_analyses WHERE ticker=%s', (ticker,))
    row = cur.fetchone()
    if row:
        row = dict(row)
        value = row['analysis']
        if isinstance(value, str):
            value = json.loads(value)
        while isinstance(value, dict) and 'thesis' not in value and isinstance(value.get('analysis'), dict):
            value = value['analysis']
        if not isinstance(value, dict):
            raise ValueError('The saved thesis has an unsupported shape. Resolve it before importing an upgrade.')
        row['analysis'] = value
    return row


def sections(before, after):
    before = before or {}
    return [{'name': label, 'key': key, 'before': before.get(key), 'after': after.get(key),
             'changed': before.get(key) != after.get(key)}
            for key, label in [('thesis', 'Investment thesis'), ('signposts', 'Signposts'), ('threats', 'Risks'), ('conclusion', 'Conclusion')]]


def candidate(previous, package, ident):
    a = copy.deepcopy(previous or {})
    a.update(copy.deepcopy(package['analysis']))
    for key in ('_pipelineChanges', '_factCorrections'):
        a.pop(key, None)
    # Preserve all prior documents, and do not trust imported history/approval claims.
    old_docs = (previous or {}).get('documentHistory') or []
    a['documentHistory'] = list({digest(d): d for d in old_docs + a['documentHistory']}.values())
    history = list((previous or {}).get('history') or [])
    if previous:
        history.append({'timestamp': previous.get('updatedAt'), **{k: previous.get(k) for k in CORE}})
    a['history'] = history[-20:]
    a['updatedAt'] = datetime.now(timezone.utc).isoformat()
    a['externalDraft'] = {'id': ident, 'sourceRegister': package['sourceRegister'],
        'provenance': package['provenance'], 'evidenceStatus': 'References supplied by author; originals not verified by import',
        'execution': 'external', 'approvedAt': a['updatedAt']}
    return a


def create_blueprint(get_db, invalidate=lambda: None):
    bp = Blueprint('thesis_imports', __name__)
    ready = False
    lock = threading.Lock()

    def ensure():
        nonlocal ready
        with lock:
            if ready:
                return
            import thesis_history
            thesis_history.ensure_schema(get_db)
            with get_db(commit=True) as (_, cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('external-thesis-drafts-schema',))
                cur.execute('''CREATE TABLE IF NOT EXISTS external_thesis_drafts (
                    id TEXT PRIMARY KEY, ticker VARCHAR(20) NOT NULL, fingerprint TEXT UNIQUE NOT NULL,
                    package JSONB NOT NULL, baseline JSONB, baseline_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', revision_id INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW())''')
            ready = True

    @bp.after_request
    def no_cache(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    @bp.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @bp.errorhandler(psycopg2.Error)
    def database_failure(exc):
        # SQL exception details can contain private research. Log only the class.
        current_app.logger.error('External thesis database failure: %s', type(exc).__name__)
        return jsonify(error='Charlie could not finish this operation. Refresh the draft status before retrying. Thesis and approval changes are saved together or rolled back together.'), 503

    @bp.get('/api/thesis-imports/prepare/<ticker>')
    def prepare(ticker):
        ticker = ticker_value(ticker)
        with get_db() as (_, cur):
            row = current(cur, ticker)
        old = row['analysis'] if row else {}
        package = {'schema': SCHEMA, 'ticker': ticker, 'companyName': row['company'] if row else '',
            'baseline': {'hash': row_hash(row), 'mode': 'upgrade' if row else 'initial', 'expectedNoExistingThesis': not bool(row)},
            'analysis': {k: old.get(k) for k in CORE} if row else {
                'thesis': {'summary': '', 'pillars': [{'id': ticker + '-P1', 'title': '', 'description': '', 'confidence': '', 'sources': []}]},
                'signposts': [{'id': ticker + '-S1', 'metric': '', 'target': '', 'timeframe': '', 'sources': []}],
                'threats': [{'id': ticker + '-R1', 'threat': '', 'triggerPoints': '', 'sources': []}], 'conclusion': ''},
            'sourceRegister': old.get('externalDraft', {}).get('sourceRegister', []),
            'provenance': {'execution': 'subscription-assisted', 'authorTool': ''}}
        return jsonify(package=package, instructions=(
            'Prepare an investment thesis draft using the attached source documents and this JSON template. '
            'Return one JSON file using this exact schema; retain ticker, baseline and existing item IDs. '
            'Fill companyName and every section, distinguish reported facts, forecasts and interpretation, '
            'include specific signpost targets/timeframes and risk triggerPoints, and use text for these fields. '
            'Citations belong in each item.sources: sourceId, filename, pdfPages (physical page numbers), excerpt '
            '(only a short exact quote or empty for paraphrase), supportType and optional sha256. '
            'sourceRegister is a list of objects with unique id, name, optional date/pages/hash/usageStatus. '
            'Preserve source restrictions; never invent sources, hashes or independent validation. '
            'Include uncertainty and contrary evidence. Do not include original document bodies or credentials. '
            'This is a draft for investor review, not approval. Charlie will not run paid models on import.'))

    @bp.route('/api/thesis-imports', methods=['GET', 'POST'])
    def drafts():
        ensure()
        if request.method == 'GET':
            ticker = request.args.get('ticker')
            with get_db() as (_, cur):
                cur.execute('''SELECT id,ticker,status,created_at,updated_at,revision_id,
                    package->>'companyName' AS company FROM external_thesis_drafts ''' +
                    ('WHERE ticker=%s ' if ticker else '') + 'ORDER BY created_at DESC LIMIT 100',
                    (ticker_value(ticker),) if ticker else ())
                rows = [dict(r) for r in cur.fetchall()]
            return jsonify(drafts=rows)
        if request.content_length and request.content_length > MAX_BYTES:
            raise ValueError('The draft exceeds 2 MB.')
        package = normalize(request.get_json(silent=True))
        with get_db(commit=True) as (_, cur):
            row = current(cur, package['ticker'])
            fingerprint = digest(package)
            cur.execute('SELECT id FROM external_thesis_drafts WHERE fingerprint=%s', (fingerprint,))
            existing = cur.fetchone()
            if existing:
                return jsonify(id=existing['id'], replayed=True)
            baseline = package['baseline']
            if (baseline.get('hash') and baseline['hash'] != row_hash(row)) or ((baseline.get('expectedNoExistingThesis') or baseline.get('mode') == 'initial') and row):
                return jsonify(error='The saved thesis differs from the draft’s starting version. Prepare a fresh package and reconcile your draft before importing.'), 409
            ident = str(uuid.uuid4())
            cur.execute('''INSERT INTO external_thesis_drafts(id,ticker,fingerprint,package,baseline,baseline_hash)
                VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s) ON CONFLICT(fingerprint) DO NOTHING''',
                (ident, package['ticker'], fingerprint, json.dumps(package), json.dumps(row['analysis'] if row else None), row_hash(row)))
            cur.execute('SELECT id FROM external_thesis_drafts WHERE fingerprint=%s', (fingerprint,))
            ident = cur.fetchone()['id']
        return jsonify(id=ident), 201

    @bp.get('/api/thesis-imports/<ident>')
    def detail(ident):
        ensure()
        with get_db() as (_, cur):
            cur.execute('SELECT * FROM external_thesis_drafts WHERE id=%s', (ident,))
            row = cur.fetchone()
            if not row:
                return jsonify(error='Draft not found.'), 404
            latest = current(cur, row['ticker'])
        d = dict(row)
        d['stale'] = row_hash(latest) != row['baseline_hash'] and row['status'] == 'pending'
        d['sections'] = sections(row['baseline'], row['package']['analysis'])
        d['warnings'] = ['Import checks structure and reference consistency, not the truth of claims or the contents of originals.',
            'Approving replaces the detailed thesis, signposts and risks. Existing scorecards, full/condensed theses and generated reports are not regenerated; review them separately.']
        if not row['package']['baseline'].get('hash'):
            d['warnings'].append('This draft has no Charlie preparation receipt. Its comparison baseline was captured at import; carefully reconcile any earlier edits.')
        if not row['package']['sourceRegister']:
            d['warnings'].append('No source register supplied. Evidence remains unverified.')
        return jsonify(d)

    @bp.post('/api/thesis-imports/<ident>/dismiss')
    def dismiss(ident):
        ensure()
        with get_db(commit=True) as (_, cur):
            cur.execute("UPDATE external_thesis_drafts SET status='dismissed',updated_at=NOW() WHERE id=%s AND status='pending' RETURNING id", (ident,))
            row = cur.fetchone()
        return jsonify(success=True) if row else (jsonify(error='Only pending drafts can be dismissed.'), 409)

    @bp.post('/api/thesis-imports/<ident>/approve')
    def approve(ident):
        ensure()
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise ValueError('Invalid approval request.')
        if body.get('confirm') is not True:
            raise ValueError('Explicit investor approval is required.')
        with get_db(commit=True) as (_, cur):
            cur.execute('SELECT * FROM external_thesis_drafts WHERE id=%s FOR UPDATE', (ident,))
            row = cur.fetchone()
            if not row:
                return jsonify(error='Draft not found.'), 404
            if body.get('ticker') != row['ticker'] or body.get('fingerprint') != row['fingerprint']:
                return jsonify(error='Reload this draft and confirm its ticker before approval.'), 409
            if row['status'] == 'approved':
                return jsonify(success=True, ticker=row['ticker'], revisionId=row['revision_id'], replayed=True)
            if row['status'] != 'pending':
                return jsonify(error='This draft is no longer pending.'), 409
            # Legacy writers do not share advisory locks. A brief table lock also fences
            # concurrent initial INSERTs and all existing save/restore UPDATE paths.
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute('LOCK TABLE portfolio_analyses IN SHARE ROW EXCLUSIVE MODE')
            latest = current(cur, row['ticker'])
            if row_hash(latest) != row['baseline_hash']:
                return jsonify(error='The thesis changed after this draft was imported. Nothing was applied. Prepare again and reconcile against the latest version.'), 409
            previous = latest['analysis'] if latest else None
            a = candidate(previous, row['package'], ident)
            import onepager
            diff = onepager.diff_thesis(previous or {}, a)
            old_summary = ((previous or {}).get('thesis') or {}).get('summary', '')
            diff['summary'] = {'before': old_summary, 'after': a['thesis']['summary'],
                               'changed': old_summary != a['thesis']['summary']}
            if diff['summary']['changed']:
                diff['counts']['summary'] = 1
                diff['has_changes'] = True
            # Save a restorable pre-import snapshot even for legacy theses without a journal.
            if previous:
                cur.execute('''INSERT INTO thesis_revisions(ticker,source,summary,snapshot)
                    VALUES(%s,'import_baseline','State before external draft approval',%s::jsonb)''',
                    (row['ticker'], json.dumps(previous)))
            cur.execute('''INSERT INTO portfolio_analyses(ticker,company,analysis,updated_at)
                VALUES(%s,%s,%s::jsonb,NOW()) ON CONFLICT(ticker) DO UPDATE SET
                company=EXCLUDED.company,analysis=EXCLUDED.analysis,updated_at=NOW()''',
                (row['ticker'], row['package']['companyName'], json.dumps(a)))
            cur.execute('''INSERT INTO thesis_revisions(ticker,source,summary,counts,diff,snapshot)
                VALUES(%s,'external_import','Investor approved external thesis draft',%s::jsonb,%s::jsonb,%s::jsonb) RETURNING id''',
                (row['ticker'], json.dumps(diff.get('counts', {})), json.dumps(diff), json.dumps(a)))
            revision = cur.fetchone()['id']
            cur.execute('''INSERT INTO thesis_snapshots(ticker,snapshot_type,thesis_summary,pillar_count,conviction,raw_snapshot)
                VALUES(%s,'external_import',%s,%s,%s,%s::jsonb)''',
                (row['ticker'], a['thesis']['summary'], len(a['thesis']['pillars']), a['conclusion'], json.dumps({'analysis': a, 'scorecard_data': None})))
            cur.execute("UPDATE external_thesis_drafts SET status='approved',revision_id=%s,updated_at=NOW() WHERE id=%s", (revision, ident))
        invalidate()
        return jsonify(success=True, ticker=row['ticker'], revisionId=revision)

    return bp
