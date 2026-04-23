"""Tests for APScheduler setup and kill switch."""
import app_v3
import scheduler


def test_scheduler_registers_expected_jobs(clean_db):
    sched = scheduler.build_scheduler(use_memory_jobstore=True)
    job_ids = {j.id for j in sched.get_jobs()}
    assert {'feed_poller', 'transcribe_worker', 'extract_worker'} <= job_ids


def test_kill_switch_skips_job_body(clean_db, monkeypatch):
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO app_settings (key, value) VALUES ('media_scheduler_enabled', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        ''', ('false',))
    called = []
    monkeypatch.setattr('media_trackers.poller.poll_all_feeds', lambda: called.append(1))
    scheduler.run_feed_poller()
    assert called == []


def test_kill_switch_default_on_runs_job(clean_db, monkeypatch):
    # No app_settings row → default on
    called = []
    monkeypatch.setattr('media_trackers.poller.poll_all_feeds', lambda: called.append(1))
    scheduler.run_feed_poller()
    assert called == [1]
