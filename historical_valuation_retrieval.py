"""Durable retrieval reservation; interrupted requests never replay providers."""
import json
from historical_valuation_store import _identity
from historical_valuation_context import load_context


def retrieve_once(get_db, aid, eid, request_id, selection, fetch):
    request_id = _identity(request_id)
    # This commits before network access. Process death leaves a visible reservation.
    with get_db(commit=True) as (_, cur):
        baseline, _ = load_context(cur, aid, eid)
        frozen = dict(baseline=baseline, selection=selection)
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('historical-retrieval-schema'))")
        cur.execute('''CREATE TABLE IF NOT EXISTS historical_valuation_retrievals (
            id UUID PRIMARY KEY, frozen JSONB NOT NULL, status TEXT NOT NULL,
            result JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
        cur.execute('''INSERT INTO historical_valuation_retrievals(id,frozen,status)
            VALUES(%s,%s::jsonb,'reserved') ON CONFLICT DO NOTHING RETURNING id''',
            (request_id,json.dumps(frozen,allow_nan=False)))
        inserted = cur.fetchone() is not None
        cur.execute('SELECT frozen,status,result FROM historical_valuation_retrievals WHERE id=%s FOR UPDATE',(request_id,))
        saved = cur.fetchone()
        if saved['frozen'] != frozen:
            raise ValueError('Retrieval identity belongs to a different target or selection.')
        if not inserted:
            if saved['status'] == 'retained':
                return saved['result']
            raise ValueError('Previous retrieval remains unresolved; inspect retained originals before starting another request.')
        # A new request UUID must not bypass an ambiguous provider outcome.
        # The assignment row lock above serializes different identities too.
        # Compact daily-price requests fetch the same response for all chosen
        # dates. Changing a date must not bypass an unresolved provider request.
        cur.execute("""SELECT id FROM historical_valuation_retrievals
            WHERE frozen->'baseline'->>'assignmentId'=%s
              AND (frozen->'selection'=%s::jsonb
                   OR (%s='daily_prices' AND frozen->'selection'->>'kind'='daily_prices'))
              AND status!='retained'
              AND id!=%s LIMIT 1""", (aid,json.dumps(selection,allow_nan=False),selection.get('kind'),request_id))
        if cur.fetchone():
            raise ValueError('An earlier retrieval for this assignment and selection remains unresolved; inspect its provider outcome and originals before another request.')
    # Exceptions intentionally leave the reservation. No inferred provider retry.
    result = fetch()
    with get_db(commit=True) as (_, cur):
        current, _ = load_context(cur, aid, eid)
        if current != baseline:
            raise ValueError('Target changed; inspect retained originals before another request.')
        cur.execute('SELECT status,result FROM historical_valuation_retrievals WHERE id=%s FOR UPDATE',(request_id,))
        saved=cur.fetchone()
        if saved and saved['status']=='retained':
            if saved['result']!=result:raise ValueError('Retained ingestion receipt changed.')
            return saved['result']
        cur.execute("UPDATE historical_valuation_retrievals SET status='retained',result=%s::jsonb WHERE id=%s AND status='reserved'",
                    (json.dumps(result,allow_nan=False),request_id))
        if cur.rowcount != 1:
            raise ValueError('Retrieval reservation changed; inspect retained evidence.')
    return result


def inspect_retrievals(cur, aid, eid):
    """Read target-scoped reservations without source bodies or provider access."""
    baseline, _ = load_context(cur, aid, eid)
    cur.execute("SELECT to_regclass('historical_valuation_retrievals') AS name")
    if not cur.fetchone()['name']:
        return {'retrievals': [], 'truncated': False, 'valuationReady': False, 'baseline': baseline}
    cur.execute('''SELECT id,frozen,status,created_at FROM historical_valuation_retrievals
        WHERE frozen->'baseline'->>'assignmentId'=%s
          AND (frozen->'baseline'->>'parentEditionId'=%s OR status!='retained')
        ORDER BY (status!='retained') DESC,created_at DESC,id LIMIT 61''', (aid,eid))
    rows=cur.fetchall(); items=[]
    for row in rows[:60]:
        frozen=row['frozen']; selection=frozen.get('selection') or {}
        retained=row['status']=='retained'
        items.append({'id':str(row['id']),'kind':selection.get('kind'),
                      'parentEditionId':frozen.get('baseline',{}).get('parentEditionId'),
                      'priceDate':selection.get('priceDate'),
                      'status':'evidence_retained' if retained else 'unresolved',
                      'createdAt':row['created_at'].isoformat(),
                      'baselineChanged':frozen.get('baseline')!=baseline,
                      'retryAllowed':False,
                      'nextAction':'Inspect the retained original inventory; evidence still requires review.' if retained else
                      'Inspect provider outcome and retained originals. Do not start another retrieval to bypass this reservation.'})
    return {'retrievals':items,'truncated':len(rows)>60,'valuationReady':False,'baseline':baseline}


def preflight_receipt(cur, request_id, baseline, selection):
    """Reject a mismatched reservation before trusted ingestion contacts a provider."""
    request_id = _identity(request_id)
    cur.execute('SELECT frozen,status FROM historical_valuation_retrievals WHERE id=%s FOR UPDATE', (request_id,))
    row = cur.fetchone()
    if not row or row['frozen'].get('baseline') != baseline:
        raise ValueError('Original receipt does not belong to this frozen retrieval.')
    if row['frozen'].get('selection') != selection:
        raise ValueError('Original receipt does not match the reserved source selection.')
    if row['status'] != 'reserved':
        raise ValueError('Retrieval already has an outcome; replay its saved receipt without provider access.')


def retain_receipt(cur, request_id, baseline, receipt):
    """Commit an ingestion receipt in the same transaction as its original bytes.

    Trusted ingestion calls this after source verification and target recheck.
    It is not an endpoint and cannot settle an unknown provider outcome by age.
    """
    request_id=_identity(request_id)
    cur.execute('SELECT frozen,status,result FROM historical_valuation_retrievals WHERE id=%s FOR UPDATE',(request_id,))
    row=cur.fetchone()
    if not row or row['frozen'].get('baseline')!=baseline:
        raise ValueError('Original receipt does not belong to this frozen retrieval.')
    selection=row['frozen'].get('selection') or {}
    if (receipt.get('kind')!=selection.get('kind') or
            (selection.get('kind')=='daily_prices' and receipt.get('priceDate')!=selection.get('priceDate'))):
        raise ValueError('Original receipt does not match the reserved source selection.')
    if row['status']=='retained':
        if row['result']!=receipt:raise ValueError('Retrieval already has a different receipt.')
        return
    if row['status']!='reserved':raise ValueError('Retrieval reservation cannot retain this receipt.')
    cur.execute("UPDATE historical_valuation_retrievals SET status='retained',result=%s::jsonb WHERE id=%s",(json.dumps(receipt,allow_nan=False),request_id))
