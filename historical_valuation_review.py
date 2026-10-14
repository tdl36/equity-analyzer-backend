"""Pure pending-review snapshots; no persistence, approval, rendering or dispatch."""
import hashlib
import json
import re
import uuid
from historical_valuation_evidence import check_market_supplement


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _baseline(value):
    fields = {'assignmentId', 'parentEditionId', 'fundamentalsDigest', 'sourcePackFingerprint', 'parentEditionDigest'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('The assignment, parent edition and frozen source baseline are required.')
    for key in ('assignmentId', 'parentEditionId'):
        raw = value[key]
        if not isinstance(raw, str) or str(uuid.UUID(raw)) != raw:
            raise ValueError('Canonical assignment and parent edition identities are required.')
    for key in ('fundamentalsDigest', 'sourcePackFingerprint', 'parentEditionDigest'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('Exact frozen baseline digests are required.')
    return dict(value)


def baseline_from_edition(run, assignment, fundamentals, edition, *, current_parent_id):
    """Bind server-loaded target contents and its current accepted lineage.

    This verifies linkage, not the edition's substantive claims or investor
    decision. Callers still lock these records and separately load originals.
    A pending edition based on an older accepted parent cannot acquire a fresh
    financial supplement without first being reconciled to the current lineage.
    """
    from onepager_edition_contract import baseline as edition_baseline
    from onepager_corrections import digest
    if not isinstance(edition, dict):
        raise ValueError('A server-loaded target edition is required.')
    if assignment.get('status') in ('cancelled', 'stopped'):
        raise ValueError('The coordinating assignment was stopped.')
    # An accepted target is itself the current parent. Its original parent field
    # describes its ancestry and must remain intact for baseline comparison.
    target_id = edition.get('id')
    parent = edition.get('parentId') if target_id == current_parent_id else current_parent_id
    expected = edition_baseline(run, assignment, fundamentals, parent_id=parent)
    if any(edition.get(key) != value for key, value in expected.items()):
        raise ValueError('Target edition no longer matches its assignment, report or accepted lineage.')
    return _baseline({'assignmentId': expected['assignmentId'],
        'parentEditionId': target_id, 'parentEditionDigest': digest(edition),
        'fundamentalsDigest': expected['fundamentalsDigest'],
        'sourcePackFingerprint': expected['fingerprint']})


def prepare_pending(supplement, originals, baseline, *, cik, symbol, price_retrieval_dates):
    """Bind diagnostics to their intended edition without altering that edition.

    Caller supplies the live baseline and trusted receipt context, never values
    accepted directly from a proposal. Original bytes remain in their source store.
    Matching amounts remain diagnostics, not approval or corporate-action evidence.
    """
    frozen_baseline = _baseline(baseline)
    diagnosis = check_market_supplement(supplement, originals, cik, symbol, price_retrieval_dates)
    payload = {'version': 1, 'status': 'pending', 'baseline': frozen_baseline,
               'proposal': supplement, 'diagnosis': diagnosis}
    encoded = _json(payload)
    if len(encoded) > 500000:
        raise ValueError('Pending historical review exceeds 500 KB.')
    # A detached JSON snapshot prevents later caller mutation of the proposal,
    # baseline or retrieval-date inventory from changing this pending version.
    return {'reviewDigest': hashlib.sha256(encoded).hexdigest(), 'payload': json.loads(encoded)}


def replay_pending(snapshot, originals, current_baseline, *, cik, symbol, price_retrieval_dates):
    """Recompute before use; changed baseline, bytes or context requires a new version.

    Successful replay still grants no permission to publish, apply or approve.
    Missing sources can be retained as gaps, but their later arrival changes the
    evidence identity and requires preparing a new pending snapshot.
    """
    if not isinstance(snapshot, dict) or set(snapshot) != {'reviewDigest', 'payload'}:
        raise ValueError('A complete pending review snapshot is required.')
    try:
        encoded = _json(snapshot['payload'])
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Invalid pending review JSON.') from exc
    if len(encoded) > 500000 or hashlib.sha256(encoded).hexdigest() != snapshot['reviewDigest']:
        raise ValueError('Pending review snapshot was changed.')
    payload = snapshot['payload']
    if not isinstance(payload, dict) or payload.get('baseline') != _baseline(current_baseline):
        raise ValueError('The target edition or frozen source baseline has changed.')
    result = prepare_pending(payload.get('proposal'), originals, current_baseline,
                             cik=cik, symbol=symbol, price_retrieval_dates=price_retrieval_dates)
    if result != snapshot:
        raise ValueError('Evidence, diagnostics or pending status changed; prepare a new review version.')
    return result
