"""Operational readiness receipts; never a score of investment correctness."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def timestamp(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            result = parsedate_to_datetime(value)
    else:
        raise ValueError('Missing timestamp')
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def summarize(assignments, now=None):
    now = now or datetime.now(timezone.utc)
    rows = []
    ids = set()
    for data in assignments:
        command = data['command']
        if command['id'] in ids:
            raise ValueError('Duplicate command in batch receipt')
        ids.add(command['id'])
        preparation = data.get('preparation') or {}
        collection = data.get('collection') or {}
        versions = data.get('versions') or []
        ready = (preparation.get('status') == 'done' and bool(versions)
                 and bool(data.get('documents')) and bool(data.get('meetingId')))
        problem = preparation.get('error') or collection.get('issue') or command.get('error')
        age = None
        first_pack_seconds = None
        try:
            submitted = timestamp(command['created_at'])
            age = max(0, int((now - submitted).total_seconds()))
            if ready:
                first_saved = min(timestamp(v['created_at']) for v in versions)
                first_pack_seconds = max(0, int((first_saved - submitted).total_seconds()))
        except (ValueError, TypeError, KeyError, OverflowError):
            pass
        rows.append({'commandId': command['id'], 'ticker': command['ticker'],
                     'readyForReview': ready, 'collectionStatus': collection.get('status', 'unknown'),
                     'preparationStatus': preparation.get('status', 'not_started'),
                     'preparationStage': preparation.get('stage'), 'issue': problem,
                     'meetingOriginals': len(data.get('documents') or []),
                     'collectionSourceProgress': collection.get('sourceProgress'),
                     'cachedAnalyses': preparation.get('cachedSources', 0),
                     'savedVersions': len(versions), 'meetingId': data.get('meetingId'),
                     'secondsSinceSubmitted': age, 'secondsToFirstSavedPack': first_pack_seconds})
    return {'status': 'ready_for_review' if rows and all(r['readyForReview'] for r in rows) else 'incomplete',
            'companies': rows, 'ready': sum(r['readyForReview'] for r in rows), 'total': len(rows),
            'checkedAt': now.isoformat(),
            'limitations': ['Readiness and wall-clock latency only; original-to-output acceptance checks and analyst review remain required.',
                            'First-pack time includes queueing, collection and generation; it is not model execution time.']}
