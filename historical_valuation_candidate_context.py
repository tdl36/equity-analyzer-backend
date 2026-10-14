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
