"""Explicit pending-preview evidence, isolated from the shared original gate.

This adapter is not installed in a renderer or endpoint yet. Its claims are
pending editorial proposals, never model-supported or investor-accepted claims.
"""
import copy
from dataclasses import dataclass
from onepager_edition_contract import prepare_edition
from stock_evidence import valid_citations, report_fields


@dataclass(frozen=True)
class PendingEditionEvidence:
    edition_id: str
    run: dict
    claims: dict
    correction_ids: tuple
    source_fingerprint: str
    financial_digest: str
    label: str = 'Pending editorial preview — not accepted'


def _replace(report, path, statement):
    """Replace one prevalidated scalar using canonical JSON pointer segments."""
    parts = [part.replace('~1', '/').replace('~0', '~') for part in path.split('/')[1:]]
    current = report
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    if isinstance(current, list):
        current[int(parts[-1])] = statement
    else:
        current[parts[-1]] = statement


def pending_evidence(run, assignment, record, corrections, documents, financials):
    """Revalidate a persisted pending selection and return an isolated preview.

    All arguments must be server-loaded. Future routes must lock/reload them in
    the same read transaction. No client review flags are evidence of a decision.
    The returned run keeps its original citations; only the explicit claims map
    may be consumed by a renderer displaying the mandatory pending label.
    """
    if not isinstance(record, dict) or record.get('status') != 'pending':
        raise ValueError('Only a saved pending edition can be previewed here.')
    body = {key: record.get(key) for key in
            ('assignmentId', 'runId', 'fingerprint', 'fundamentalsDigest', 'parentId', 'correctionIds')}
    body['requestId'] = record.get('id')
    checked = prepare_edition(run, assignment, body, corrections, documents, financials)
    if checked != record:
        raise ValueError('The saved edition evidence changed. Prepare a new edition.')
    derived = copy.deepcopy(run)
    claims = copy.deepcopy(valid_citations(run))
    fields = report_fields(run['state']['report'])
    for entry in checked['claims']:
        p = entry['proposal']; path = p['path']
        if fields.get(path) != p['before']:
            raise ValueError('The original claim changed. Prepare a new edition.')
        _replace(derived['state']['report'], path, p['statement'])
        # Deliberately no review='supported' or passageMatched promotion. The
        # source match is retained as evidence metadata, separate from meaning.
        claims[path] = {
            'statement': p['statement'], 'basis': p['basis'],
            'reviewOrigin': 'pending_editorial', 'editionId': checked['id'],
            'correctionId': entry['correctionId'],
            'evidence': [{'sourceId': p['sourceId'], 'excerpt': p['excerpt'],
                          'locator': p['locator'], 'originalHash': p['sha256'],
                          'matched': True}],
            'caveat': p['caveat'],
        }
    return PendingEditionEvidence(checked['id'], derived, claims,
                                  tuple(checked['correctionIds']),
                                  checked['fingerprint'], checked['fundamentalsDigest'])
