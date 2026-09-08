"""Connection-scoped execution ownership: released by PostgreSQL on disconnect."""
from contextlib import contextmanager


def lock_name(job_id):return 'amendment-worker:'+job_id


@contextmanager
def worker_session(get_db,job_id):
    with get_db() as (connection,cur):
        try:
            cur.execute('SELECT pg_try_advisory_lock(hashtext(%s)) AS locked',(lock_name(job_id),))
            acquired=bool(cur.fetchone()['locked'])
            connection.commit()
        except Exception:
            connection.close()
            raise
        try:yield acquired
        finally:
            if acquired:
                try:
                    cur.execute('SELECT pg_advisory_unlock(hashtext(%s))',(lock_name(job_id),))
                    connection.commit()
                except Exception:
                    # Never return a possibly locked session to the connection pool.
                    connection.close()
                    raise
