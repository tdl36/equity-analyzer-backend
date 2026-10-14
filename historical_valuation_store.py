"""Transaction-scoped pending supplement storage, not an approval API.

The caller must lock and load the assignment, parent edition, frozen fundamentals,
source pack and original receipts in the same transaction. These functions never
commit, modify the baseline, fetch sources, dispatch models or accept research.
"""
import json
import uuid
from historical_valuation_review import prepare_pending, replay_pending


def _identity(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError('A canonical review submission identity is required.')
    return value


def save_pending(cur, request_id, supplement, originals, baseline, **context):
    ident = _identity(request_id)
    snapshot = prepare_pending(supplement, originals, baseline, **context)
    # Transaction lock also protects first-use schema creation across processes.
    cur.execute("SELECT pg_advisory_xact_lock(hashtext('historical-valuation-review-schema'))")
    cur.execute('''CREATE TABLE IF NOT EXISTS historical_valuation_reviews (
        id UUID PRIMARY KEY, assignment_id UUID NOT NULL, parent_edition_id UUID NOT NULL,
        review_digest TEXT NOT NULL, snapshot JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
    cur.execute('''INSERT INTO historical_valuation_reviews
        (id,assignment_id,parent_edition_id,review_digest,snapshot)
        VALUES(%s,%s,%s,%s,%s::jsonb) ON CONFLICT(id) DO NOTHING RETURNING id''',
        (ident, baseline['assignmentId'], baseline['parentEditionId'],
         snapshot['reviewDigest'], json.dumps(snapshot)))
    inserted = cur.fetchone() is not None
    cur.execute('SELECT assignment_id,parent_edition_id,review_digest,snapshot FROM historical_valuation_reviews WHERE id=%s', (ident,))
    stored = cur.fetchone()
    if (str(stored['assignment_id']) != baseline['assignmentId']
            or str(stored['parent_edition_id']) != baseline['parentEditionId']
            or stored['review_digest'] != snapshot['reviewDigest']
            or stored['snapshot'] != snapshot):
        raise ValueError('Submission identity already belongs to different historical review evidence.')
    return {'id': ident, 'status': 'pending', 'reused': not inserted,
            'reviewDigest': snapshot['reviewDigest']}


def read_pending(cur, request_id, originals, baseline, **context):
    """Read without creating tables; replay against the current locked context."""
    ident = _identity(request_id)
    cur.execute("SELECT to_regclass('historical_valuation_reviews') AS name")
    if not cur.fetchone()['name']:
        return None
    cur.execute('SELECT assignment_id,parent_edition_id,review_digest,snapshot FROM historical_valuation_reviews WHERE id=%s FOR SHARE', (ident,))
    stored = cur.fetchone()
    if not stored:
        return None
    if (str(stored['assignment_id']) != baseline.get('assignmentId')
            or str(stored['parent_edition_id']) != baseline.get('parentEditionId')
            or stored['review_digest'] != stored['snapshot'].get('reviewDigest')):
        raise ValueError('Saved historical review belongs to a different or changed baseline.')
    return replay_pending(stored['snapshot'], originals, baseline, **context)
