"""Durable Telegram issue delivery; no processing retries or research side effects."""
import hashlib
import time
import uuid
from audio_intake_receipts import connection, DEFAULT_PATH


def setup(db):
    db.execute('''CREATE TABLE IF NOT EXISTS audio_issue_notifications (
        id TEXT PRIMARY KEY, message TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
        next_attempt REAL NOT NULL DEFAULT 0, claim TEXT)''')


def enqueue(filename, reason, job_id='', path=DEFAULT_PATH):
    # One notice per issue/job; different issues remain visible. Never use raw
    # exception text, headers, transcripts or provider responses in messages.
    filename=str(filename)[:500]; reason=str(reason)[:800]; job_id=str(job_id or '')[:200]
    ident=hashlib.sha256(repr((filename,reason,job_id)).encode()).hexdigest()
    message=('Charlie processing needs attention\n'+filename+'\n'+reason+
             ('\nJob: '+job_id if job_id else '')+
             '\nOriginal preserved. No automatic paid retry. Open Charlie Alerts for details.')
    with connection(path) as db, db:
        setup(db)
        db.execute('INSERT OR IGNORE INTO audio_issue_notifications(id,message) VALUES(?,?)',(ident,message))
    return ident


def deliver(sender, path=DEFAULT_PATH, now=None):
    """Bounded delivery with a durable lease; success requires acknowledged send.

    Telegram has no idempotency key. An accepted send whose response is lost can
    duplicate on retry; ordinary polling/restarts cannot duplicate confirmed sends.
    """
    now=time.time() if now is None else now
    for _ in range(3):
        claim=str(uuid.uuid4())
        with connection(path) as db, db:
            setup(db)
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT id,message FROM audio_issue_notifications WHERE delivered=0 AND next_attempt<=? ORDER BY rowid LIMIT 1',(now,)).fetchone()
            if row is None:return
            db.execute('UPDATE audio_issue_notifications SET next_attempt=?,claim=? WHERE id=?',(now+300,claim,row['id']))
        try: ok=sender(row['message']) is True
        except Exception: ok=False
        if ok:
            with connection(path) as db, db:
                db.execute('UPDATE audio_issue_notifications SET delivered=1 WHERE id=? AND claim=?',(row['id'],claim))
