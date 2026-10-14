"""Read-only candidate assembly inside a caller-owned authenticated transaction.

No schema creation, saves, commits, provider calls or approval. The assignment
lock and original row locks remain held through assembly/replay. Original bodies
are never returned; only the bounded candidate snapshot leaves this adapter.
"""
from historical_valuation_context import load_context
from historical_valuation_review import _baseline
from historical_valuation_source_store import read_records
from historical_valuation_candidates import assemble_candidates


def assemble_from_storage(cur, assignment_id, edition_id, selections, expected_baseline):
    expected = _baseline(expected_baseline)
    baseline, identity = load_context(cur, assignment_id, edition_id)
    if baseline != expected:
        raise ValueError('The saved target changed; reload its evidence before selecting inputs.')
    # Validate selection shape/size and conflicting receipts before reading bytes.
    assemble_candidates(selections, [], baseline, **identity)
    refs = [selection[field] for selection in selections
            for field in ('priceSource', 'earningsSource')]
    records = read_records(cur, refs, assignment_id=baseline['assignmentId'])
    return assemble_candidates(selections, records, baseline, **identity)


def replay_from_storage(cur, assignment_id, edition_id, snapshot):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('payload'), dict):
        raise ValueError('A complete saved candidate snapshot is required.')
    payload = snapshot['payload']
    current = assemble_from_storage(cur, assignment_id, edition_id,
                                    payload.get('selections'), payload.get('baseline'))
    # Verify the supplied digest, all qualifiers and the entire prior snapshot;
    # never return a refreshed version under an earlier identity.
    if current != snapshot:
        raise ValueError('Candidate evidence changed; prepare a new version.')
    return current


def plan_from_storage(cur, assignment_id, edition_id, price_date, expected_baseline):
    """Plan from the complete bounded inventory, never an arbitrary recent slice."""
    from historical_valuation_selection import plan_selection
    expected = _baseline(expected_baseline)
    baseline, identity = load_context(cur, assignment_id, edition_id)
    if baseline != expected:
        raise ValueError('The saved target changed; reload its evidence before planning inputs.')
    # Validate identity/date before inventory access. Missing coverage is explicit.
    plan_selection([], baseline, price_date=price_date, **identity)
    records = _complete_inventory(cur, baseline['assignmentId'])
    return plan_selection(records, baseline, price_date=price_date, **identity)


def _complete_inventory(cur, assignment_id):
    """Caller holds assignment lock, shared with trusted ingestion writers."""
    cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
    records = []
    if cur.fetchone()['name']:
        cur.execute('''SELECT source_id,sha256 FROM historical_valuation_originals
            WHERE assignment_id=%s ORDER BY source_id LIMIT 61 FOR SHARE''', (assignment_id,))
        rows = cur.fetchall()
        if len(rows)>60:
            raise ValueError('Retained inventory exceeds the planning bound; source selection requires review.')
        if rows:
            refs = [{'id':row['source_id'],'sha256':row['sha256']} for row in rows]
            records = read_records(cur, refs, assignment_id=assignment_id)
    return records


def readiness_from_storage(cur, assignment_id, edition_id, dates, expected_baseline):
    """One locked inventory snapshot for the entire frozen date batch."""
    from historical_valuation_readiness import readiness
    expected = _baseline(expected_baseline)
    baseline, identity = load_context(cur, assignment_id, edition_id)
    if baseline != expected:
        raise ValueError('The saved target changed; reload before checking readiness.')
    # Fail malformed batches before fetching any original bodies.
    readiness([], baseline, dates=dates, **identity)
    records = _complete_inventory(cur, baseline['assignmentId'])
    return readiness(records, baseline, dates=dates, **identity)


def replay_readiness_from_storage(cur, assignment_id, edition_id, snapshot):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('payload'), dict):
        raise ValueError('A complete historical readiness checkpoint is required.')
    payload = snapshot['payload']
    current = readiness_from_storage(cur, assignment_id, edition_id,
                                     payload.get('dates'), payload.get('baseline'))
    if current != snapshot:
        raise ValueError('Historical evidence or target changed; prepare a new checkpoint.')
    return current


def edition_readiness_from_storage(cur, assignment_id, edition_id, expected_baseline):
    """Derive calendar anchors from the edition's frozen financials, then inspect.

    No caller-supplied amounts/dates, fresh quote, inferred trading session or
    persisted coordinator mutation. load_context keeps assignment input/vintage
    locked while the date policy verifies the same combined financial digest.
    """
    from historical_valuation_dates import frozen_dates
    from historical_valuation_readiness import readiness
    expected = _baseline(expected_baseline)
    baseline, identity = load_context(cur, assignment_id, edition_id)
    if baseline != expected:
        raise ValueError('The saved target changed; reload its evidence.')
    cur.execute('SELECT input,result FROM mp_jobs WHERE id=%s', (baseline['assignmentId'],))
    row = cur.fetchone()
    policy = frozen_dates(row['result']['visualFundamentals'], baseline,
                          assignment_input=row['input'])
    result = {'datePolicy': policy, 'valuationReady': False, 'dispatchAllowed': False}
    if policy['payload']['dates']:
        records = _complete_inventory(cur, baseline['assignmentId'])
        result['readiness'] = readiness(records, baseline, **identity, dates=policy['payload']['dates'])
    return result
