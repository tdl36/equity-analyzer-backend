"""Validate a future explicit investor decision, without recording or applying it.

This module is not an approval endpoint. A future caller must authorize the
investor, lock the assignment's accepted lineage and all evidence, check durable
decision replay/conflicts, and commit the decision and pointer atomically. The
returned intent is not accepted evidence and cannot be passed to a renderer.
"""
from onepager_corrections import digest
from onepager_edition_contract import _uuid, prepare_edition
from company_research import eligible
from research_originals import verified_original


_IDENTITY = ('assignmentId', 'runId', 'fingerprint', 'fundamentalsDigest',
             'parentId', 'correctionIds', 'selectionDigest')
_KEYS = set(_IDENTITY) | {'requestId', 'editionId', 'action'}


def validate_decision(run, assignment, record, body, corrections, documents,
                      financials, *, parent_id=None):
    """Return a bounded decision intent only after fresh complete-pack checks.

    record/corrections/documents/financials and parent_id must be server loaded.
    Both accept and reject require the exact reviewed baseline. A rejection never
    advances lineage. Editions are complete correction selections against the
    original report, not implicit cumulative patches from their parent.
    """
    if not isinstance(body, dict) or set(body) != _KEYS:
        raise ValueError('Name the exact edition and decision; review flags are not accepted.')
    request_id = _uuid(body['requestId'])
    edition_id = _uuid(body['editionId'])
    if body['action'] not in ('accept', 'reject'):
        raise ValueError('Choose accept or reject explicitly.')
    if not isinstance(record, dict) or record.get('status') != 'pending':
        raise ValueError('Only an undecided pending edition can be reviewed.')
    if record.get('id') != edition_id:
        raise ValueError('The decision names a different edition.')
    if assignment.get('status') in ('stopped', 'cancelled'):
        raise ValueError('This assignment was stopped. No decision can be recorded.')
    for key in _IDENTITY:
        if body[key] != record.get(key):
            raise ValueError('The reviewed selection changed. Reload the exact edition.')
    selection = {key: record.get(key) for key in _IDENTITY
                 if key != 'selectionDigest'}
    selection['requestId'] = edition_id
    checked = prepare_edition(run, assignment, selection, corrections, documents,
                              financials, parent_id=parent_id)
    if checked != record:
        raise ValueError('Saved edition evidence changed. Prepare a new edition.')
    # Unchanged claims also appear on the proposed page, so checking only the
    # selected corrections' originals is insufficient for a review decision.
    sources = run.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('The frozen source pack is unavailable.')
    seen = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get('id'), str) or not source['id']:
            raise ValueError('Invalid frozen source identity.')
        if source['id'] in seen:
            raise ValueError('Ambiguous frozen source identity.')
        seen.add(source['id'])
        filename = source.get('filename')
        if not isinstance(filename, str) or not filename:
            raise ValueError('Invalid frozen original filename.')
        document = documents.get(filename)
        if not isinstance(document, dict) or not eligible(document.get('metadata') or {}):
            raise ValueError('A frozen original is missing or restricted. Inspect its usage terms.')
        verified_original(source, document, (run.get('input', {}).get('hashes') or {}).get(filename))
    return {'requestId': request_id, 'editionId': edition_id,
            'action': body['action'], 'assignmentId': checked['assignmentId'],
            'runId': checked['runId'], 'parentId': checked['parentId'],
            'selectionDigest': checked['selectionDigest'],
            'editionDigest': digest(checked), 'payloadDigest': digest(body)}
