"""Bounded connection waiting; never retries an application transaction."""
import time
from contextlib import contextmanager
from psycopg2.pool import PoolError


def acquire_connection(pool, timeout=8.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return pool.getconn()
        except PoolError as exc:
            # A closed/misconfigured pool will not recover merely by waiting.
            if 'exhausted' not in str(exc).lower():
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PoolError(f'Database is busy; no connection became available within {timeout:g} seconds. Please retry.') from exc
            time.sleep(min(0.05, remaining))


@contextmanager
def pooled_cursor(pool, commit=False):
    conn = acquire_connection(pool)
    cur = None
    broken = False
    try:
        # Cursor creation belongs inside the cleanup boundary too.
        cur = conn.cursor()
        yield conn, cur
        if commit:
            conn.commit()
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                broken = True
        try:
            conn.rollback()  # Clear implicit read transactions before reuse.
        except Exception:
            broken = True
        try:
            pool.putconn(conn, close=broken or bool(conn.closed))
        except PoolError:
            # A pool closed during shutdown no longer owns the connection.
            conn.close()
