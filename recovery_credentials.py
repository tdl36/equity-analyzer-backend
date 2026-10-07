"""Resolve credentials for already-authorized background recovery only."""
import os


def anthropic_recovery_key(get_db):
    """Prefer the server key; otherwise use Charlie's existing saved Settings key.

    Resolve on every sweep so rotation/removal takes effect without a restart.
    Database failures propagate to the caller's type-only recovery log; never
    include credential values in diagnostics or persist them on individual jobs.
    """
    key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if key:
        return key
    with get_db() as (_, cur):
        cur.execute('SELECT value FROM app_settings WHERE key=%s', ('apiKey',))
        row = cur.fetchone()
    value = row.get('value') if row else None
    return value.strip() if isinstance(value, str) else ''
