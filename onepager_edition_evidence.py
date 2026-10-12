"""Explicit pending-preview evidence, isolated from the shared original gate.

Renderers consume this adapter only through an explicit typed argument. Its claims are
pending editorial proposals, never model-supported or investor-accepted claims.
"""
import copy
from dataclasses import dataclass
from onepager_edition_contract import prepare_edition
from onepager_corrections import digest
from stock_evidence import valid_citations, report_fields


@dataclass(frozen=True)
class PendingEditionEvidence:
    edition_id: str
    run: dict
    claims: dict
    correction_ids: tuple
    source_fingerprint: str
    financial_digest: str
    assignment_digest: str
    render_input_digest: str
    content_digest: str
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
                                  checked['fingerprint'], checked['fundamentalsDigest'],
                                  digest(assignment['input']),
                                  digest({'assignment': assignment['input'], 'financials': financials}),
                                  digest({'run': derived, 'claims': claims}))


def render_evidence(run, preview=None):
    """Resolve only explicit server context; ordinary and thesis gates stay intact."""
    if preview is None:
        return run, valid_citations(run)
    from onepager_corrections import preparation
    if type(preview) is not PendingEditionEvidence:
        raise ValueError('Preview requires a validated pending edition context.')
    if preparation(run)['fingerprint'] != preview.source_fingerprint:
        raise ValueError('The preview report changed. Prepare a new edition.')
    if digest({'run': preview.run, 'claims': preview.claims}) != preview.content_digest:
        raise ValueError('The pending preview context changed. Reload validated evidence.')
    return preview.run, preview.claims


def pending_label(claim):
    return 'Pending editorial proposal — not accepted' if claim and claim.get('reviewOrigin') == 'pending_editorial' else ''


def labeled_statement(claim):
    label = pending_label(claim)
    return (label + ': ' if label else '') + claim['statement']


def preview_notice(preview, run):
    """Print-visible provenance, with no claim of a decision or source refresh."""
    if preview is None:
        return ''
    import html
    return ('<p class="footnote micro edition-notice"><b>' + html.escape(preview.label) +
            '</b> · Edition ' + html.escape(preview.edition_id) +
            ' · Report ' + html.escape(run['id']) +
            '. Proposed wording requires investor review. Original research and financial snapshot retained.</p>')


def check_render_inputs(run, assignment, fundamentals, preview):
    if preview is None:
        return
    render_evidence(run, preview)
    inputs = {k:v for k,v in assignment.items() if k != '_visualFundamentals'}
    if digest(inputs) != preview.assignment_digest or (fundamentals is not None and digest({'assignment': inputs, 'financials': fundamentals}) != preview.render_input_digest):
        raise ValueError('Preview assignment or financial snapshot changed. Reload the saved inputs.')
