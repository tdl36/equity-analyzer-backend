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
from media_trackers import poller, extractor, transcribe, clustering, cost_watch, notifications
import earnings
import finnhub_sync
import briefings
import theme_tracker


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


def run_cluster_weekly():
    if not _kill_switch_on():
        return
    clustering.run_weekly_clustering()


def run_email_digest_daily():
    if not _kill_switch_on():
        return
    notifications.send_daily_email_digest()


def run_cost_watch_daily():
    if not _kill_switch_on():
        return
    cost_watch.check_cost_warning()


def run_theme_scan():
    """Hourly: scan last 48h of podcast bullets; route per-analyst theme
    alerts (AI capex, etc.) when the same theme spans 2+ covered tickers
    with 3+ bullets."""
    if not _kill_switch_on():
        return
    try:
        theme_tracker.run_theme_scan()
    except Exception as e:
        print(f'run_theme_scan error: {e}')


def run_briefing_bmo():
    """6:30 AM ET (10:30 UTC) — before market open."""
    if not _kill_switch_on():
        return
    try:
        briefings.send_briefings_for_context('bmo')
    except Exception as e:
        print(f'run_briefing_bmo error: {e}')


def run_briefing_midday():
    """12:30 PM ET (16:30 UTC) — midday, after BMO reports have been digested."""
    if not _kill_switch_on():
        return
    try:
        briefings.send_briefings_for_context('midday')
    except Exception as e:
        print(f'run_briefing_midday error: {e}')


def run_briefing_amc():
    """5:30 PM ET (21:30 UTC) — after market close, before evening review."""
    if not _kill_switch_on():
        return
    try:
        briefings.send_briefings_for_context('amc')
    except Exception as e:
        print(f'run_briefing_amc error: {e}')


def run_finnhub_earnings_sync():
    """Nightly: pull the next 60 days of earnings dates from Finnhub and
    upsert into earnings_calendar for covered tickers."""
    if not _kill_switch_on():
        return
    try:
        finnhub_sync.sync(days_ahead=60)
    except Exception as e:
        # Missing API key is expected until user sets it — don't spam logs
        msg = str(e)
        if 'API key' not in msg:
            print(f'run_finnhub_earnings_sync error: {msg}')


def run_earnings_fetch_sweep():
    """Phase 3d: check earnings calendar twice per US trading day and dispatch
    fetch jobs for any ticker whose confirmed earnings date == today."""
    if not _kill_switch_on():
        return
    try:
        earnings.check_due_earnings()
    except Exception as e:
        print(f'run_earnings_fetch_sweep error: {e}')


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
    # Transcribe is the throughput bottleneck. Run every minute and allow 2
    # instances overlapping so a slow Gemini batch doesn't starve the next
    # tick. With MAX_EPISODES_PER_BATCH=4 and MAX_PARALLEL_TRANSCRIBE=4, peak
    # in-flight is 8 audio buffers (~1GB worst case — fits Render Standard,
    # tight on Starter; lower max_instances if memory pressure shows).
    sched.add_job(run_transcribe_worker,  'interval', minutes=1,  id='transcribe_worker',  replace_existing=True, max_instances=2, coalesce=False)
    sched.add_job(run_extract_worker,     'interval', minutes=1,  id='extract_worker',     replace_existing=True)
    sched.add_job(run_cluster_weekly,     'cron', day_of_week='sun', hour=5, minute=0,
                  id='cluster_weekly',     replace_existing=True)
    sched.add_job(run_email_digest_daily, 'cron', hour=7,  minute=0,
                  id='email_digest_daily', replace_existing=True)
    sched.add_job(run_cost_watch_daily,   'cron', hour=23, minute=0,
                  id='cost_watch_daily',   replace_existing=True)
    # Phase 3d: earnings auto-fetch sweeps. 11:30 UTC (~7:30am ET) for
    # pre-market reports, 21:30 UTC (~5:30pm ET) for post-close reports.
    sched.add_job(run_earnings_fetch_sweep, 'cron', hour='11,21', minute=30,
                  id='earnings_fetch_sweep', replace_existing=True)
    # Phase 3d+: Finnhub earnings calendar sync — daily at 08:00 UTC (~4am ET).
    sched.add_job(run_finnhub_earnings_sync, 'cron', hour=8, minute=0,
                  id='finnhub_earnings_sync', replace_existing=True)
    # Analyst briefings — 3x/day, weekdays only (Mon=0 .. Fri=4)
    sched.add_job(run_briefing_bmo,    'cron', hour=10, minute=30, day_of_week='mon-fri',
                  id='briefing_bmo',    replace_existing=True)
    sched.add_job(run_briefing_midday, 'cron', hour=16, minute=30, day_of_week='mon-fri',
                  id='briefing_midday', replace_existing=True)
    sched.add_job(run_briefing_amc,    'cron', hour=21, minute=30, day_of_week='mon-fri',
                  id='briefing_amc',    replace_existing=True)
    # Cross-stock theme tracker — hourly during US weekdays (10 UTC = 6 AM ET
    # through 23 UTC = 7 PM ET). 14 scans/day catches fresh cross-coverage
    # themes as they emerge in podcasts.
    sched.add_job(run_theme_scan, 'cron', hour='10-23', minute=5, day_of_week='mon-fri',
                  id='theme_scan', replace_existing=True)
    return sched


_scheduler = None


def start():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = build_scheduler()
    _scheduler.start()
    return _scheduler
