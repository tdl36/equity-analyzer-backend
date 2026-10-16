"""Display-only terminal collection receipts through the existing agent sync.

No source bytes, processing authorization, worker dispatch or job mutations.
"""
import json
import math
import re
import uuid

PREFIX = 'collection_terminal_receipt:'
CHECK = 'collection_history_checked:'
LIMIT = 50


def identity(row):
    if not isinstance(row, dict):
        return None
    try:
        ident = str(uuid.UUID(row.get('id', '')))
    except (ValueError, TypeError, AttributeError):
        return None
    ticker = row.get('ticker')
    if not isinstance(ticker, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}', ticker):
        return None
    return ident, ticker


def terminal(row):
    ids = identity(row)
    if not ids or row.get('status') not in ('complete', 'cancelled'):
        return None
    created = row.get('created')
    if type(created) not in (float, int) or not math.isfinite(created) or created <= 0:
        return None
    return dict(id=ids[0], ticker=ids[1], status=row['status'], created=created,
                historicalReceipt=True, receiptOnly=True)


def remember(cur, rows):
    for row in rows:
        value = terminal(row)
        if value:
            # Terminal requests cannot be retried in place. Retain the first
            # exact receipt; repeated syncs never rewrite that evidence.
            cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING',
                        (PREFIX + value['id'], json.dumps(value)))


def lookup(cur, snapshot, request_id, ticker):
    ids = identity({'id': request_id, 'ticker': ticker})
    if not ids:
        return None
    for row in snapshot.get('requests', []):
        if identity(row) == ids:
            return row
    cur.execute('SELECT value FROM app_settings WHERE key=%s', (PREFIX + ids[0],))
    row = cur.fetchone()
    if not row:
        return None
    try:
        value = json.loads(row['value']) if isinstance(row['value'], str) else row['value']
        value = terminal(value)
    except (ValueError, TypeError):
        return None
    return value if value and identity(value) == ids else None


def requested_history(cur):
    # Unknown local requests rotate behind unchecked requests, so an absent
    # historical row cannot starve later ones. Reads do not advance the queue.
    cur.execute('''SELECT j.result->>'refreshRequestId' AS id, j.ticker
        FROM mp_jobs j
        LEFT JOIN app_settings saved ON saved.key=%s || (j.result->>'refreshRequestId')
        LEFT JOIN app_settings checked ON checked.key=%s || (j.result->>'refreshRequestId')
        WHERE j.stage='collection_control' AND j.result->>'refreshRequestId' IS NOT NULL
          AND saved.key IS NULL
        GROUP BY j.result->>'refreshRequestId',j.ticker,checked.updated_at
        ORDER BY checked.updated_at ASC NULLS FIRST, MIN(j.created_at),j.result->>'refreshRequestId',j.ticker
        LIMIT 50''', (PREFIX, CHECK))
    return [dict(id=ids[0], ticker=ids[1]) for row in cur.fetchall() if (ids := identity(row))]


def local_history(db, requests):
    """Exact SQLite reads only; never invokes collection or recreates a request."""
    values = []
    for requested in requests[:LIMIT]:
        ids = identity(requested)
        if not ids:
            continue
        row = db.execute('SELECT id,ticker,status,created FROM refresh_requests WHERE id=? AND ticker=?', ids).fetchone()
        value = terminal(dict(row)) if row else None
        values.append(value or dict(id=ids[0], ticker=ids[1], status='unavailable'))
    return values


def record_history_checks(cur, rows):
    for row in rows:
        ids = identity(row)
        if not ids:
            continue
        # Accept only references actually retained by an existing command.
        cur.execute("SELECT 1 FROM mp_jobs WHERE stage='collection_control' AND ticker=%s AND result->>'refreshRequestId'=%s LIMIT 1", (ids[1], ids[0]))
        if not cur.fetchone():
            continue
        remember(cur, [row])
        cur.execute('''INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s,NOW())
            ON CONFLICT(key) DO UPDATE SET updated_at=EXCLUDED.updated_at''',
                    (CHECK + ids[0], json.dumps({'id': ids[0], 'ticker': ids[1]})))
