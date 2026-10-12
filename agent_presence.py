"""Run-scoped worker reporting; never grants ownership, cancellation or replay."""
import json
import threading
from datetime import datetime, timezone
from uuid import UUID


def key(run_id):
    return 'analyst_presence_v1:' + str(UUID(str(run_id)))


class Presence:
    def __init__(self, get_db, run_id, execution_id):
        self.db, self.key = get_db, key(run_id)
        self.execution_id = str(UUID(str(execution_id)))
        self.stop = threading.Event()
        self.write('reporting')  # Fail before model work if initial evidence cannot persist.
        self.thread = threading.Thread(target=self.loop, daemon=True, name='analyst-presence')
        self.thread.start()

    def write(self, state):
        value = dict(executionId=self.execution_id, state=state)
        with self.db(commit=True) as (_, cur):
            cur.execute('''INSERT INTO app_settings(key,value,updated_at) VALUES(%s,%s::jsonb,NOW() AT TIME ZONE 'UTC')
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=EXCLUDED.updated_at''',
                (self.key, json.dumps(value)))

    def loop(self):
        while not self.stop.wait(30):
            try:
                self.write('reporting')
            except Exception:
                # A failed heartbeat must expire visibly, never imply a safe restart.
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            return  # A pending DB write must not overwrite a later stopped receipt.
        try:
            self.write('exited')
        except Exception:
            pass  # Preserve the job outcome; existing presence ages into unknown.


def inspect(get_db, run_id, reservation_ids):
    with get_db() as (_, cur):
        cur.execute('SELECT value,updated_at FROM app_settings WHERE key=%s', (key(run_id),))
        row = cur.fetchone()
    return describe(row, reservation_ids)


def describe(row, reservation_ids, now=None):
    unknown = dict(state='unknown', lastSeen=None)
    if not row:
        return unknown
    value = row.get('value')
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return unknown
    if not isinstance(value, dict) or value.get('executionId') not in reservation_ids:
        return unknown
    seen = row.get('updated_at')
    if not isinstance(seen, datetime):
        return unknown
    seen = seen.replace(tzinfo=timezone.utc) if seen.tzinfo is None else seen
    age = ((now or datetime.now(timezone.utc)) - seen).total_seconds()
    state = value.get('state')
    if state not in ('reporting', 'exited') or age < 0:
        return unknown
    return dict(state='stale' if state == 'reporting' and age > 90 else state,
                lastSeen=seen.isoformat())
