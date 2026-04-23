"""APScheduler setup for media trackers.

build_scheduler() returns a configured BackgroundScheduler without starting it,
useful for tests. start() is called from app_v3.py at boot unless
APSCHEDULER_DISABLED env var is set.
"""
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore

import app_v3
from media_trackers import poller, extractor, transcribe


def _kill_switch_on() -> bool:
    """Return True if the scheduler should execute job bodies."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key = 'media_scheduler_enabled'")
            row = cur.fetchone()
        if not row:
            return True  # default on
        return str(row['value']).lower() not in ('false', '0', 'off', 'no')
    except Exception:
        return False  # fail closed


def run_feed_poller():
    if not _kill_switch_on():
        return
    poller.poll_all_feeds()


def run_transcribe_worker():
    if not _kill_switch_on():
        return
    transcribe.process_transcribe_batch()


def run_extract_worker():
    if not _kill_switch_on():
        return
    extractor.process_extract_batch()


def build_scheduler(use_memory_jobstore: bool = False) -> BackgroundScheduler:
    if use_memory_jobstore or not os.environ.get('DATABASE_URL'):
        jobstore = MemoryJobStore()
    else:
        jobstore = SQLAlchemyJobStore(url=os.environ['DATABASE_URL'], tablename='apscheduler_jobs')
    sched = BackgroundScheduler(
        jobstores={'default': jobstore},
        job_defaults={'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 300},
    )
    sched.add_job(run_feed_poller,        'interval', minutes=30, id='feed_poller',        replace_existing=True)
    sched.add_job(run_transcribe_worker,  'interval', minutes=2,  id='transcribe_worker',  replace_existing=True)
    sched.add_job(run_extract_worker,     'interval', minutes=2,  id='extract_worker',     replace_existing=True)
    return sched


_scheduler = None


def start():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = build_scheduler()
    _scheduler.start()
    return _scheduler
