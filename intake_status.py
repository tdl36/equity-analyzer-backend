"""Read-only explanations of existing automatic-intake eligibility gates."""
from datetime import datetime,timezone,timedelta

def reason(state, *, now=None, has_thesis=None, blocking=None, has_key=None, budget=None):
    now=now or datetime.now(timezone.utc)
    if not state.get('enabled'):return 'paused','Automatic intake is paused. Existing sources are retained; enable intake when ready.'
    if has_key is False:return 'credentials','A server-side model key is required for unattended comparisons.'
    if budget:return 'budget',str(budget)
    if state.get('day')==now.date().isoformat() and state.get('used',0)>=state.get('dailyLimit',2):
        reset=(now+timedelta(days=1)).date().isoformat()
        return 'daily_limit',f'Daily comparison limit reached. Eligibility resets at {reset} 00:00 UTC; pending sources remain queued.'
    if has_thesis is False:return 'baseline','This company needs a saved thesis before automatic comparison can run.'
    if blocking=='awaiting_approval':return 'review','An evidence proposal is awaiting your review. Resolve that proposal before another automatic comparison.'
    if blocking:return 'in_progress','A comparison or import for this company is already in progress. The new source remains queued.'
    return 'worker','Waiting for the next worker eligibility check. Import availability, credentials and budget are checked before dispatch.'

def annotate(cur,events,state,**checks):
    """No writes or model calls; match the worker's company-level eligibility query."""
    tickers=sorted({r['ticker'] for r in events if r.get('status')=='waiting'})
    eligibility={}
    for ticker in tickers:
        cur.execute("""SELECT EXISTS(SELECT 1 FROM portfolio_analyses WHERE ticker=%s) AS has_thesis,
          (SELECT status FROM mp_jobs WHERE ticker=%s AND
            ((stage='evidence_amendment' AND status IN ('queued','running','awaiting_approval')) OR
             (stage='auto_evidence_intake' AND status='processing'))
           ORDER BY CASE WHEN status='awaiting_approval' THEN 0 ELSE 1 END LIMIT 1) AS blocking""",(ticker,ticker))
        eligibility[ticker]=dict(cur.fetchone())
    for event in events:
        if event.get('status')=='waiting':
            code,message=reason(state,**eligibility[event['ticker']],**checks)
            event.update(waitCode=code,waitReason=message)
