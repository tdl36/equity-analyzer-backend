"""Read-only explanation of the existing collector's claim eligibility.

No claims, scheduling, lease renewal or policy changes occur here. Positions describe
an untargeted claim at the sampled instant; an explicitly targeted claim can differ.
"""

def queue_diagnostics(db, stamp):
    rows = [dict(r) for r in db.execute('''
        SELECT q.id,q.ticker,q.status,q.created,q.lease_until,
            p.ticker AS policy_ticker,json_extract(p.config,'$.enabled') AS enabled,
            json_extract(q.config,'$.manual') AS manual,COALESCE(c.turn,0) AS turn
        FROM refresh_requests q LEFT JOIN refresh_policies p ON p.ticker=q.ticker
        LEFT JOIN refresh_claim_order c ON c.request_id=q.id
        WHERE q.status NOT IN ('complete','cancelled')
        ORDER BY COALESCE(c.turn,0),q.created,q.id''')]
    active = [r for r in rows if r['status'] in ('collecting','verifying')
              and r['lease_until'] is not None and r['lease_until'] > stamp]
    eligible = [r for r in rows if r['status'] in ('queued','collecting','verifying')
                and (r['lease_until'] is None or r['lease_until'] <= stamp)
                and r['policy_ticker'] is not None
                and (r['enabled'] == 1 or r['manual'] == 1)]
    positions = {r['id']: n+1 for n,r in enumerate(eligible)}
    blockers = [{'requestId':r['id'],'ticker':r['ticker'],'leaseUntil':r['lease_until']}
                for r in active[:3]]
    result = {}
    for r in rows:
        position = positions.get(r['id'])
        if r in active:
            state = 'leased'
        elif r['status'] in ('attention','needs_auth'):
            state = 'needs_attention'
        elif r['policy_ticker'] is None:
            state = 'missing_policy'
        elif r['enabled'] != 1 and r['manual'] != 1:
            state = 'policy_paused'
        elif r['lease_until'] is not None and r['lease_until'] > stamp:
            state = 'lease_wait'
        elif position:
            state = 'waiting_for_slot' if active else 'eligible'
        else:
            state = 'not_claimable'
        result[r['id']] = dict(asOf=stamp,requestStatus=r['status'],state=state,
            defaultPosition=position,eligibleCount=len(eligible),
            activeCount=len(active),active=blockers,
            leaseUntil=r['lease_until'] if r['lease_until'] and r['lease_until'] > stamp else None)
    return result
