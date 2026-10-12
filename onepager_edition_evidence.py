"""Explicit pending-preview evidence, isolated from the shared original gate.

Renderers consume this adapter only through an explicit typed argument. Its claims are
pending editorial proposals, never model-supported or investor-accepted claims.
"""
import copy
from dataclasses import dataclass
from onepager_edition_contract import prepare_edition
from onepager_corrections import digest, evidence_items
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


@dataclass(frozen=True)
class AcceptedEditionEvidence(PendingEditionEvidence):
    decision_id: str = ''
    decided_at: str = ''
    label: str = 'Investor-accepted editorial edition'


def accepted_evidence(run, assignment, record, corrections, documents, financials, *, decision):
    """Revalidate an explicitly named retained acceptance, never a client flag.

    Caller loads the decision and edition under the assignment lock. Historical
    editions retain their own parent; a later acceptance does not rewrite them.
    Current source policy and exact bytes still gate access to every version.
    """
    from onepager_decision_contract import validate_decision
    if not isinstance(decision, dict) or decision.get('action') != 'accept' or not decision.get('decidedAt'):
        raise ValueError('A retained investor acceptance is required.')
    body = {k: record.get(k) for k in ('assignmentId','runId','fingerprint','fundamentalsDigest','parentId','correctionIds','selectionDigest')}
    body.update(requestId=decision.get('requestId'), editionId=record.get('id'), action='accept')
    intent = validate_decision(run, assignment, record, body, corrections, documents, financials, parent_id=record.get('parentId'))
    for key in ('requestId','editionId','editionDigest'):
        if decision.get(key) != intent[key]:
            raise ValueError('Saved acceptance does not match this edition.')
    pending = pending_evidence(run, assignment, record, corrections, documents, financials, parent_id=record.get('parentId'))
    claims = copy.deepcopy(pending.claims)
    for claim in claims.values():
        if claim.get('reviewOrigin') == 'pending_editorial':
            claim.update(reviewOrigin='investor', decisionId=intent['requestId'], decidedAt=decision['decidedAt'])
    values = {k:getattr(pending,k) for k in PendingEditionEvidence.__dataclass_fields__ if k != 'label'}
    values.update(claims=claims, content_digest=digest({'run':pending.run,'claims':claims}))
    return AcceptedEditionEvidence(**values, decision_id=intent['requestId'], decided_at=decision['decidedAt'])


def editorial_label(claim):
    if claim and claim.get('reviewOrigin') == 'investor':
        return 'Investor-reviewed editorial wording'
    return pending_label(claim)


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


def pending_evidence(run, assignment, record, corrections, documents, financials, *, parent_id=None):
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
    checked = prepare_edition(run, assignment, body, corrections, documents, financials, parent_id=parent_id)
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
            'evidence': [{'sourceId': e['sourceId'], 'excerpt': e['excerpt'],
                          'locator': e['locator'], 'originalHash': e['sha256'],
                          'matched': True} for e in evidence_items(p)],
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
    if type(preview) not in (PendingEditionEvidence, AcceptedEditionEvidence):
        raise ValueError('Preview requires a validated pending edition context.')
    if preparation(run)['fingerprint'] != preview.source_fingerprint:
        raise ValueError('The preview report changed. Prepare a new edition.')
    if digest({'run': preview.run, 'claims': preview.claims}) != preview.content_digest:
        raise ValueError('The pending preview context changed. Reload validated evidence.')
    return preview.run, preview.claims


def pending_label(claim):
    return 'Pending editorial proposal — not accepted' if claim and claim.get('reviewOrigin') == 'pending_editorial' else ''


def labeled_statement(claim):
    label = editorial_label(claim)
    return (label + ': ' if label else '') + claim['statement']


def preview_notice(preview, run):
    """Print-visible provenance, with no claim of a decision or source refresh."""
    if preview is None:
        return ''
    import html
    return ('<p class="footnote micro edition-notice"><b>' + html.escape(preview.label) +
            '</b> · Edition ' + html.escape(preview.edition_id) +
            ' · Report ' + html.escape(run['id']) +
            ('. Decision ' + html.escape(preview.decision_id) + ' · Accepted ' + html.escape(preview.decided_at) + '. Acceptance does not certify source accuracy or approve a thesis.' if type(preview) is AcceptedEditionEvidence else '. Proposed wording requires investor review.') + ' Original research and financial snapshot retained.</p>')


def check_render_inputs(run, assignment, fundamentals, preview):
    if preview is None:
        return
    render_evidence(run, preview)
    inputs = {k:v for k,v in assignment.items() if k != '_visualFundamentals'}
    if digest(inputs) != preview.assignment_digest or (fundamentals is not None and digest({'assignment': inputs, 'financials': fundamentals}) != preview.render_input_digest):
        raise ValueError('Preview assignment or financial snapshot changed. Reload the saved inputs.')
