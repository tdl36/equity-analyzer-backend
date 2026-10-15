"""Durable, fail-closed guard for automatic SUMMARIES uploads; never retries POSTs."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import sqlite3
import uuid

DEFAULT_PATH = Path.home() / 'Library/Application Support/Charlie/audio-intake.sqlite3'

@contextmanager
def connection(path=DEFAULT_PATH):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=10);db.row_factory=sqlite3.Row
    try:
        os.chmod(path,0o600)
        db.execute('PRAGMA synchronous=FULL')
        db.execute('''CREATE TABLE IF NOT EXISTS intake (
            id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, digest TEXT NOT NULL,
            filename TEXT NOT NULL, status TEXT NOT NULL, job_id TEXT, reason TEXT,
            alerted INTEGER NOT NULL DEFAULT 0, UNIQUE(endpoint,digest))''')
        db.commit()
        yield db
    finally:db.close()

def prepare(filename, data, endpoint, history_reader, path=DEFAULT_PATH, allow_new=True):
    digest=hashlib.sha256(data).hexdigest()
    with connection(path) as db, db:
        db.execute('INSERT OR IGNORE INTO intake(id,endpoint,digest,filename,status) VALUES(?,?,?,?,?)',
                   (str(uuid.uuid4()),endpoint,digest,filename,'checking'))
        row=dict(db.execute('SELECT * FROM intake WHERE endpoint=? AND digest=?',(endpoint,digest)).fetchone())
    if row['status']!='checking':return row,False
    if not allow_new:
        with connection(path) as db, db:
            db.execute("UPDATE intake SET status='attention',reason=? WHERE id=? AND status='checking'",
                       ('Original was present when the agent started without a dispatch receipt. Review prior processing before a new attempt.',row['id']))
            row=dict(db.execute('SELECT * FROM intake WHERE id=?',(row['id'],)).fetchone())
        return row,False
    # A filename match cannot prove byte identity: retain for review, never assume
    # either completion or permission to repeat an older attempt.
    history=history_reader(filename)
    if (not isinstance(history,dict) or history.get('filename')!=filename
        or not isinstance(history.get('history'),list) or type(history.get('truncated')) is not bool):
        raise ValueError('Prior processing history could not be verified; no upload sent.')
    if history['history'] or history['truncated']:
        with connection(path) as db, db:
            db.execute("UPDATE intake SET status='attention',reason=? WHERE id=? AND status='checking'",
                       ('Prior processing exists for this filename. Review saved jobs before any new processing.',row['id']))
            row=dict(db.execute('SELECT * FROM intake WHERE id=?',(row['id'],)).fetchone())
        return row,False
    with connection(path) as db, db:
        changed=db.execute("UPDATE intake SET status='reserved' WHERE id=? AND status='checking'",(row['id'],)).rowcount
        row=dict(db.execute('SELECT * FROM intake WHERE id=?',(row['id'],)).fetchone())
    return row,changed==1

def update(ident,status,job_id=None,reason=None,path=DEFAULT_PATH):
    if status not in ('submitted','complete','attention'):raise ValueError('Invalid intake state')
    if status=='submitted' and (not isinstance(job_id,str) or not job_id or job_id=='?' or len(job_id)>200):
        raise ValueError('Upload receipt has no valid saved job identity')
    with connection(path) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM intake WHERE id=?',(ident,)).fetchone()
        if row is None:raise ValueError('Missing durable intake receipt')
        # A late error must not erase confirmed completion. A late successful
        # upload receipt may resolve an uncertain reservation, but cannot change
        # a previously recorded job identity.
        if job_id and row['job_id'] and job_id!=row['job_id']:
            raise ValueError('Conflicting saved job identity')
        if row['status']!='complete':
            if status=='complete' and not row['job_id']:
                raise ValueError('Completion requires a saved job identity')
            db.execute('UPDATE intake SET status=?,job_id=COALESCE(?,job_id),reason=? WHERE id=?',
                       (status,job_id,reason,ident))
        return dict(db.execute('SELECT * FROM intake WHERE id=?',(ident,)).fetchone())

def reserve_alert(ident,path=DEFAULT_PATH):
    with connection(path) as db, db:
        return db.execute('UPDATE intake SET alerted=1 WHERE id=? AND alerted=0',(ident,)).rowcount==1

def same_original(path,digest):
    try:return hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    except OSError:return False
