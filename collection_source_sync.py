"""Frozen scheduled-source defaults and narrowly scoped selection recovery."""
import json
from source_preferences import policy

def reconcile(manager,source_policy,shortlists):
    checked=policy(source_policy);resumed=[]
    with manager.c.lock():
        manager.db.execute('CREATE TABLE IF NOT EXISTS source_policy_cache(id INTEGER PRIMARY KEY,value TEXT)')
        manager.db.execute('INSERT OR REPLACE INTO source_policy_cache VALUES(1,?)',(json.dumps(checked),))
        for row in manager.db.execute("SELECT * FROM refresh_requests WHERE status='attention' AND issue LIKE 'Source selection:%'").fetchall():
            cfg=json.loads(row['config']);cid=cfg.get('sourceReviewId') or (cfg.get('eventId') if cfg.get('researchCommand') else None)
            matches=[s for s in shortlists if s.get('commandId')==cid and s.get('ticker')==row['ticker']]
            if len(matches)!=1:continue
            s=matches[0]
            if not s.get('sealed') or not s.get('candidates') or any(c.get('decision') not in ('include','exclude') for c in s['candidates']):continue
            active=manager.db.execute('SELECT config FROM refresh_policies WHERE ticker=?',(row['ticker'],)).fetchone()
            if not cfg.get('manual') and (not active or not json.loads(active['config']).get('enabled')):continue
            manager._retry(row['id']);manager.c.event(row['run'],'source_selection_resumed',selection=s.get('revision'),commandId=cid)
            resumed.append(row['id'])
    return resumed


def frozen_default(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_policy_cache'").fetchone():return None
    row=db.execute('SELECT value FROM source_policy_cache WHERE id=1').fetchone()
    return policy(json.loads(row['value'])) if row else None
