"""Lease-fenced primary-source supplements for Command Charlie.

Discovery and relevance assessment belong to the browser worker. This adapter
records that assessment and saves bounded, allowlisted public originals.
"""
import hashlib
import json
import re
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

import requests
from research_task_sources import active


def endpoint(url):
    u = urlsplit(url)
    if (u.scheme != 'https' or u.username or u.password or u.port not in (None, 443)
            or u.query or u.fragment or '%' in u.path or '..' in u.path
            or '\\' in url):
        raise ValueError('Use a canonical HTTPS primary-source URL')
    if u.hostname == 'clinicaltrials.gov' and re.fullmatch(r'/study/NCT\d{8}', u.path):
        return 'https://clinicaltrials.gov/api/v2/studies/' + u.path.rsplit('/', 1)[-1], 'registry'
    if u.hostname == 'www.fda.gov' and re.fullmatch(
            r'/(?:news-events/press-announcements|drugs|medical-devices)/[A-Za-z0-9/_-]+', u.path):
        return 'https://www.fda.gov' + u.path, 'fda'
    raise ValueError('Supported supplements are FDA pages and ClinicalTrials.gov study records')


def retrieve(url):
    with requests.get(url, timeout=25, stream=True, allow_redirects=False,
                      headers={'User-Agent': 'CharlieResearch/1.0', 'Accept': 'application/json,text/html'}) as response:
        if response.status_code != 200:
            raise ValueError('Primary-source retrieval failed (%s); not an empty search' % response.status_code)
        data = bytearray()
        for block in response.iter_content(65536):
            data.extend(block)
            if len(data) > 15_000_000:
                raise ValueError('Primary source exceeds 15 MB')
        if not data:
            raise ValueError('Primary source is empty')
        return bytes(data)


def register(manager, rid, owner, record, fetch=retrieve):
    row = active(manager, rid, owner)
    cfg = json.loads(row['config'])
    command = cfg.get('researchCommand')
    if not command:
        raise ValueError('A managed research command is required')
    url, kind = endpoint(record['url'])
    day = record.get('sourceDate', '')
    if date.fromisoformat(day).isoformat() != day or not command['since'] <= day <= command['until']:
        raise ValueError('Verified publication/update date must be inside the task window')
    for key in ('title', 'relevance', 'dateEvidence'):
        if not isinstance(record.get(key), str) or not 10 <= len(record[key].strip()) <= 2000:
            raise ValueError('Provide bounded title, company relevance and observed date evidence')
    root = manager.c.catalysts / cfg['ticker'] / cfg['topic']
    def validate_root():
        if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(manager.c.catalysts.resolve()):
            raise ValueError('Invalid catalyst destination')
    validate_root()
    data = fetch(url)
    if kind == 'registry':
        study = json.loads(data)
        protocol = study.get('protocolSection', {})
        if protocol.get('identificationModule', {}).get('nctId') != url.rsplit('/', 1)[-1]:
            raise ValueError('Registry identity does not match the requested study')
        actual_day = protocol.get('statusModule', {}).get('lastUpdatePostDateStruct', {}).get('date')
        if actual_day != day:
            raise ValueError('Registry last posted update differs from the observed source date')
        suffix = '.json.txt'
    else:
        if not re.search(br'<(?:!doctype\s+html|html)\b', data[:4096], re.I):
            raise ValueError('Expected an FDA HTML original')
        if record['dateEvidence'].encode() not in data:
            raise ValueError('Observed date evidence is absent from the downloaded FDA page')
        suffix = '.html'
    digest = hashlib.sha256(data).hexdigest()
    name = '%s_%s_%s%s' % (kind.upper(), day, hashlib.sha256(url.encode()).hexdigest()[:20], suffix)
    with manager.c.lock():
        active(manager, rid, owner)
        validate_root()
        saved = manager.db.execute('SELECT manifest FROM research_public_sources WHERE request_id=?', (rid,)).fetchone()
        if not saved or not json.loads(saved['manifest']).get('complete'):
            raise ValueError('Finish the resumable SEC lookup before registering supplements')
        manifest = json.loads(saved['manifest'])
        existing = next((d for d in manifest['documents'] if d['url'] == url), None)
        target = root / name
        if target.is_symlink() or (target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError('Existing original differs; retained for inspection')
        if existing:
            if existing['sha256'] != digest or not target.is_file():
                raise ValueError('Registered source changed or is missing; original receipt retained')
            return existing
        if len(manifest['documents']) >= 40:
            raise ValueError('Public document limit reached; narrow the task')
        if not target.exists():
            temp = root / ('.' + name + '.part')
            if temp.is_symlink():
                raise ValueError('Unsafe staging path')
            temp.write_bytes(data)
            temp.replace(target)
        document = {**{k: record[k].strip() for k in ('title', 'relevance', 'dateEvidence')},
                    'url': url, 'observedUrl': record['url'], 'sourceDate': day,
                    'sourceKind': kind, 'filename': name, 'sha256': digest,
                    'retrievedAt': datetime.now(timezone.utc).isoformat(),
                    'limitations': ('Registry update date is not the date of a clinical readout. Sponsor relevance was reviewed by the worker.'
                                    if kind == 'registry' else 'Publication date and company relevance were reviewed by the worker; not independently inferred.')}
        manifest['documents'].append(document)
        manager.db.execute('UPDATE research_public_sources SET manifest=? WHERE request_id=?', (json.dumps(manifest), rid))
        return document
