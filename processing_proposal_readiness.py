"""On-demand inspection of one frozen proposal; no worker wake or paid work."""
from datetime import datetime, timezone
from processing_proposals import STAGE, obj, verify_evidence


def inspect(co, collection_id, ident):
    with co.db() as (_, cur):
        cur.execute('SELECT * FROM mp_jobs WHERE id=%s AND stage=%s', (ident, STAGE))
        row = cur.fetchone()
        if not row or obj(row['input']).get('collectionId') != collection_id:
            return None
        frozen = obj(row['input']); ticker = row['ticker']; blockers = []
        cur.execute("SELECT ticker,status,input FROM mp_jobs WHERE id=%s AND stage='research_assignment'", (collection_id,))
        parent = cur.fetchone()
        if not parent or parent['ticker'] != ticker:
            return None
        if row['status'] != 'pending':
            blockers.append('This proposal is already decided. Refresh saved proposals and inspect its saved job.')
        else:
            if parent['status'] != 'complete' or not obj(parent['input']).get('collectionOnly'):
                blockers.append('The source-only collection is not complete. Inspect its saved progress.')
            if not co.has_key():
                blockers.append('Configure the research model key in Settings before starting.')
            cur.execute("SELECT id FROM mp_jobs WHERE ticker=%s AND stage='research_assignment' AND status IN ('queued','running','attention') LIMIT 1", (ticker,))
            if cur.fetchone():
                blockers.append('An assignment for this ticker is unfinished. Resume or stop it first.')
            cur.execute("SELECT to_regclass('investment_case_versions') AS name")
            revision = 0
            if cur.fetchone()['name']:
                cur.execute('SELECT revision FROM investment_case_versions WHERE ticker=%s ORDER BY revision DESC LIMIT 1', (ticker,))
                case = cur.fetchone(); revision = case['revision'] if case else 0
            if revision != frozen['caseRevision']:
                blockers.append('The research baseline changed. Save a fresh processing proposal.')
            if frozen['plan'].get('labFullNote') and (not callable(co.lab_note_dispatch) or not callable(co.lab_note_stop)):
                blockers.append('Full Lab note integration is unavailable. The proposal remains pending.')
            try:
                verify_evidence(cur, ticker, frozen)
            except ValueError as exc:
                blockers.append(str(exc))
        return dict(proposalId=ident, collectionId=collection_id, ticker=ticker,
                    revision=1, digest=frozen['digest'], status=row['status'],
                    ready=not blockers, blockers=blockers, readOnly=True,
                    checkedAt=datetime.now(timezone.utc).isoformat())
