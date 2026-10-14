"""Load a coherent historical-review target from saved server records.

Caller owns transaction/authentication; no commits, schema creation or approval.
The assignment lock serializes with edition decisions and trusted source ingestion.
"""
from historical_valuation_store import _identity
from historical_valuation_review import baseline_from_edition
from onepager_decisions import read_lineage
from onepager_corrections import digest


def load_context(cur, assignment_id, edition_id):
    aid, eid = _identity(assignment_id), _identity(edition_id)
    cur.execute("SELECT id,ticker,status,input,result FROM mp_jobs WHERE id=%s AND stage='research_assignment' FOR UPDATE", (aid,))
    row = cur.fetchone()
    if not row:
        raise ValueError('Saved assignment not found.')
    assignment = dict(row)
    if assignment['status'] in ('stopped', 'cancelled'):
        raise ValueError('The coordinating assignment was stopped.')
    cur.execute("SELECT to_regclass('onepager_editions') AS name")
    if not cur.fetchone()['name']:
        raise ValueError('Saved edition not found.')
    cur.execute('SELECT run_id,record FROM onepager_editions WHERE id=%s AND assignment_id=%s FOR SHARE', (eid, aid))
    saved = cur.fetchone()
    result = assignment.get('result') or {}
    if not saved or str(saved['run_id']) != str(result.get('reportId')):
        raise ValueError('Saved edition does not belong to this assignment report.')
    edition = saved['record']
    if not isinstance(edition, dict) or edition.get('id') != eid:
        raise ValueError('Saved edition identity changed.')
    parent, decisions = read_lineage(cur, aid)
    decision = decisions.get(eid)
    if decision and (decision['action'] == 'reject' or decision['editionDigest'] != digest(edition)):
        raise ValueError('Target edition was rejected or its decided content changed.')
    cur.execute('SELECT id,ticker,status,input,baseline,sources,state FROM stock_analysis_runs WHERE id=%s FOR SHARE', (saved['run_id'],))
    row = cur.fetchone()
    if not row:
        raise ValueError('Saved report not found.')
    fundamentals = result.get('visualFundamentals')
    baseline = baseline_from_edition(dict(row), assignment, fundamentals, edition,
                                     current_parent_id=parent)
    # Issuer identity comes from the saved public-data snapshot, not the proposal.
    cik = fundamentals.get('cik')
    if isinstance(cik, bool) or not isinstance(cik, int) or not 0 < cik < 10000000000:
        raise ValueError('The saved fundamentals lack a verified issuer CIK.')
    return baseline, {'cik': cik, 'symbol': assignment['ticker']}
