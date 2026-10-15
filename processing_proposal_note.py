"""Freeze optional full-note settings without token counting or model calls."""
import hashlib
import json
import re
from research_purpose import context, note_prompt


def freeze(co, plan):
    if not plan.get('labFullNote'):
        return None
    if not callable(co.lab_note_defaults) or not callable(co.lab_note_dispatch) or not callable(co.lab_note_stop):
        raise ValueError('Full Lab note integration unavailable. No processing was requested.')
    defaults = co.lab_note_defaults()
    model = defaults.get('modelKey')
    prompt = defaults.get('promptTemplate')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,120}', model):
        raise ValueError('Full-note model configuration is invalid.')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Full-note instructions are unavailable.')
    purpose = plan['researchPurpose']
    prompt = note_prompt(purpose) if purpose['kind'] == 'event_note' else prompt + '\n\n' + context(purpose)
    if plan.get('instruction'):
        prompt += '\nInvestor focus (data): ' + json.dumps(plan['instruction'])
    if len(prompt.encode()) > 100000:
        raise ValueError('Full-note instructions exceed the supported size.')
    return dict(modelKey=model, promptTemplate=prompt, allowPdfText=plan['labFullNote']['allowPdfText'], maxTokens=32000, maxCalls=8)


def public(config):
    if not config:
        return None
    return {**{k: config[k] for k in ('modelKey', 'allowPdfText', 'maxTokens', 'maxCalls')},
            'promptSha256': hashlib.sha256(config['promptTemplate'].encode()).hexdigest(),
            'sourcePreparation': config.get('sourcePreparation'),
            'configDigest': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def source_plan(cur, ticker, sources, config):
    """Local extraction/planning only; freeze hashes and warnings, never source text."""
    from pipeline_lab_note_sources import prepare
    ids = [s['documentId'] for s in sources]
    cur.execute('''WITH selected AS MATERIALIZED (
        SELECT id,filename,metadata,file_type,file_data FROM document_files
        WHERE ticker=%s AND id::text=ANY(%s) FOR SHARE)
        SELECT id::text AS id,filename,metadata,file_type,
        CASE WHEN octet_length(file_data)<=20000000
          AND sum(octet_length(file_data)) OVER ()<=80000000 THEN file_data ELSE NULL END AS file_data
        FROM selected''', (ticker, ids))
    documents = {'charlie:thesis_' + r['id']: dict(r) for r in cur.fetchall()}
    for document in documents.values():
        check_pdf(document, config['allowPdfText'])
    pack = [{k: s[k] for k in ('reference', 'filename', 'sha256', 'bytes')} for s in sources]
    prepared = prepare(ticker, pack, documents, allow_pdf_text=config['allowPdfText'])
    return {**{k: prepared[k] for k in ('sourcePack', 'representationSha256', 'warnings')},
            'batchCount': len(prepared['batches'])}


def verify_sources(cur, ticker, sources, config):
    if config.get('sourcePreparation') is None:
        return  # Legacy approved settings keep their original contract.
    if source_plan(cur, ticker, sources, config) != config['sourcePreparation']:
        raise ValueError('Full-note source representation changed. Save a fresh proposal before processing.')


def check_pdf(document, allow_text):
    """Validate PDF containers; never start OCR during an interactive approval check."""
    if not (document['filename'].lower().endswith('.pdf') or 'pdf' in str(document.get('file_type') or '').lower()):
        return
    import base64
    import io
    from PyPDF2 import PdfReader
    from notegen import MAX_TOKENS_PER_BATCH, TOKENS_PER_PDF_PAGE
    try:
        raw = base64.b64decode(document['file_data'], validate=True)
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted or not len(reader.pages):
            raise ValueError('encrypted or empty PDF')
        oversized = len(reader.pages) * TOKENS_PER_PDF_PAGE > MAX_TOKENS_PER_BATCH
    except Exception:
        raise ValueError('PDF '+document['filename']+' cannot be read locally or is encrypted. Choose a readable original.') from None
    if oversized:
        if not allow_text:
            raise ValueError('PDF '+document['filename']+' requires text-only conversion. Explicitly allow extracted PDF text or select a different pack.')
        from pdf_text import extract
        def no_ocr(*args):
            raise ValueError('OCR must be prepared separately')
        try:
            extract(raw, ocr=no_ocr, engine_id='processing-proposal-no-ocr')
        except Exception:
            raise ValueError('PDF '+document['filename']+' needs OCR or has unreadable text. Prepare a readable source separately; this approval check does not run OCR.') from None
