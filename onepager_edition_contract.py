"""Pure pending-edition contract; no persistence, acceptance or rendering.

Inputs must be server-loaded saved records. The caller must lock/reload these
records in its transaction before using this preparation at any later decision.
A successful preparation is evidence identity checking, never investor approval.
"""
import uuid
from onepager_corrections import digest, preparation, prepare


_BODY_KEYS = {'requestId', 'assignmentId', 'runId', 'fingerprint',
              'fundamentalsDigest', 'parentId', 'correctionIds'}


def _uuid(value):
    if not isinstance(value, str):
        raise ValueError('Invalid edition identity.')
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise ValueError('Invalid edition identity.') from None


def baseline(run, assignment, fundamentals, parent_id=None):
    """Return bounded identities, without private source text or report bodies."""
    pack = preparation(run)
    result = assignment.get('result') or {}
    inputs = assignment.get('input') or {}
    if (str(result.get('reportId')) != pack['runId'] or
            assignment.get('ticker') != pack['ticker'] or
            inputs.get('ticker', pack['ticker']) != pack['ticker']):
        raise ValueError('The assignment does not belong to this saved report.')
    if not isinstance(fundamentals, dict) or not fundamentals:
        raise ValueError('The saved public-data snapshot is unavailable.')
    return {'assignmentId': _uuid(assignment.get('id')),
            'runId': pack['runId'], 'ticker': pack['ticker'],
            'fingerprint': pack['fingerprint'],
            # Binding the frozen assignment input also prevents output-policy drift.
            'fundamentalsDigest': digest({'input': inputs, 'fundamentals': fundamentals,
                                          'vintage': result.get('visualFundamentals')}),
            'parentId': _uuid(parent_id) if parent_id is not None else None}


def prepare_edition(run, assignment, body, corrections, documents, fundamentals, parent_id=None):
    """Revalidate an ordered selection of persisted T202 pending corrections.

    documents maps frozen filename to the current server-loaded document row.
    fundamentals is the full saved financial snapshot, not only its vintage
    metadata in assignment.result.visualFundamentals; callers must load it.
    Returns a new immutable-by-convention pending record; never alters inputs.
    This is not an endpoint and must not be used as an accepted evidence adapter.
    """
    if not isinstance(body, dict) or set(body) != _BODY_KEYS:
        raise ValueError('Invalid edition selection; review flags are not accepted.')
    identity = _uuid(body['requestId'])
    current = baseline(run, assignment, fundamentals, parent_id)
    for key in _BODY_KEYS - {'requestId', 'correctionIds'}:
        if body[key] != current[key]:
            raise ValueError('The report, assignment, public data or parent changed. Prepare a new edition.')
    selected = body['correctionIds']
    if not isinstance(selected, list) or not 1 <= len(selected) <= 60:
        raise ValueError('Select between one and sixty corrections.')
    ids = [_uuid(value) for value in selected]
    if len(set(ids)) != len(ids):
        raise ValueError('A correction may appear only once in an edition.')
    if not isinstance(corrections, list) or not isinstance(documents, dict):
        raise ValueError('Saved correction evidence is unavailable.')
    lookup = {}
    for row in corrections:
        if not isinstance(row, dict):
            raise ValueError('Invalid saved correction.')
        key = str(row.get('id'))
        if key in lookup:
            raise ValueError('Ambiguous saved correction identity.')
        lookup[key] = row
    claims, paths = [], set()
    for key in ids:
        row = lookup.get(key)
        if (not row or str(row.get('run_id')) != current['runId'] or
                row.get('fingerprint') != current['fingerprint']):
            raise ValueError('A selected correction is missing, unrelated or stale.')
        proposal = row.get('proposal')
        if not isinstance(proposal, dict):
            raise ValueError('Invalid saved correction evidence.')
        if not isinstance(proposal.get('path'), str):
            raise ValueError('Invalid saved correction path.')
        if proposal['path'] in paths:
            raise ValueError('Select only one correction for each claim.')
        replay = {name: proposal.get(name) for name in
                  ('path', 'statement', 'basis', 'reason', 'sourceId', 'excerpt', 'locator')}
        if 'additionalEvidence' in proposal:
            replay['additionalEvidence']=[{k:e.get(k) for k in ('sourceId','excerpt','locator')} for e in proposal['additionalEvidence']]
        replay.update(requestId=key, fingerprint=current['fingerprint'])
        _, signature, checked = prepare(run, replay, documents)
        if row.get('payload_hash') != signature or proposal != checked:
            raise ValueError('Saved correction evidence changed. Prepare a new correction.')
        paths.add(checked['path'])
        claims.append({'correctionId': key, 'proposalDigest': digest(checked),
                       'proposal': checked})
    return {'id': identity, **current, 'correctionIds': ids,
            'selectionDigest': digest({'baseline': current, 'claims': claims}),
            'claims': claims, 'status': 'pending',
            'caveat': 'Source identity and passages were checked. Investor review of meaning and completeness is still required.'}
