"""Phase 3d: earnings calendar + auto-fetch helpers.

Dispatches `earnings_fetch` pipeline jobs when a ticker's confirmed earnings
date is today. The local agent runs the actual HTTP download into the
iCloud CATALYSTS/{ticker}/{quarter_label}/ folder. Once files land, the
existing catalyst auto-synth flow picks them up and Phase 3b routes the
result to covering analysts as an `earnings_recap` activity.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta

import app_v3


def _today() -> date:
    return datetime.utcnow().date()


def upcoming(window_days: int = 14) -> list[dict]:
    cutoff = _today() + timedelta(days=window_days)
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT ec.*, tec.ir_url, tec.pr_url_pattern, tec.transcript_source
              FROM earnings_calendar ec
              LEFT JOIN ticker_earnings_config tec ON tec.ticker = ec.ticker
             WHERE COALESCE(ec.confirmed_date, ec.expected_date) BETWEEN %s AND %s
             ORDER BY COALESCE(ec.confirmed_date, ec.expected_date) ASC
        ''', (_today(), cutoff))
        return [dict(r) for r in cur.fetchall()]


def due_today() -> list[dict]:
    """Return earnings_calendar rows where confirmed_date == today AND
    status allows fetching (upcoming OR retry)."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('''
            SELECT ec.*, tec.ir_url, tec.pr_url_pattern, tec.transcript_source,
                   tec.transcript_url_pattern
              FROM earnings_calendar ec
              LEFT JOIN ticker_earnings_config tec ON tec.ticker = ec.ticker
             WHERE ec.confirmed_date = %s
               AND ec.status IN ('upcoming', 'fetching', 'retry')
        ''', (_today(),))
        return [dict(r) for r in cur.fetchall()]


def _dispatch_earnings_fetch_job(row: dict) -> str:
    """Create a research_pipeline_jobs row of type 'earnings_fetch' so the
    local agent can process it. Returns the job id."""
    job_id = str(uuid.uuid4())
    detail = {
        'quarter_label': row.get('quarter_label'),
        'pr_url': row.get('pr_url'),
        'transcript_url': row.get('transcript_url'),
        'ir_url': row.get('ir_url'),
        'pr_url_pattern': row.get('pr_url_pattern'),
        'transcript_source': row.get('transcript_source'),
        'transcript_url_pattern': row.get('transcript_url_pattern'),
        'calendar_id': row.get('id'),
    }
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            INSERT INTO research_pipeline_jobs
              (id, batch_id, ticker, job_type, status, progress, current_step, total_steps, steps_detail)
            VALUES (%s, %s, %s, 'earnings_fetch', 'queued', 0, 'Waiting for local agent', 3, %s)
        ''', (job_id, str(uuid.uuid4()), row['ticker'], json.dumps(detail)))
        cur.execute('''
            UPDATE earnings_calendar
               SET status='fetching', last_fetch_attempt_at=NOW(), updated_at=NOW()
             WHERE id=%s
        ''', (row['id'],))
    return job_id


def check_due_earnings() -> list[dict]:
    """Called by the scheduler. Finds due-today entries and dispatches
    earnings_fetch jobs."""
    dispatched = []
    try:
        for row in due_today():
            try:
                # Skip if we already dispatched a job in the last 6h
                with app_v3.get_db() as (_c, cur):
                    cur.execute('''
                        SELECT id FROM research_pipeline_jobs
                         WHERE ticker=%s AND job_type='earnings_fetch'
                           AND created_at > NOW() - INTERVAL '6 hours'
                    ''', (row['ticker'],))
                    if cur.fetchone():
                        continue
                job_id = _dispatch_earnings_fetch_job(row)
                dispatched.append({'ticker': row['ticker'], 'jobId': job_id, 'quarter': row.get('quarter_label')})
            except Exception as e:
                print(f'earnings.check_due_earnings: dispatch failed for {row.get("ticker")}: {e}')
    except Exception as e:
        print(f'earnings.check_due_earnings error: {e}')
    return dispatched


def record_fetch_result(calendar_id: str, status: str, notes: str | None = None) -> None:
    """Called after local agent finishes an earnings_fetch job. status should
    be 'fetched', 'partial', or 'retry'."""
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute('''
            UPDATE earnings_calendar
               SET status=%s, fetch_notes=%s, updated_at=NOW()
             WHERE id=%s
        ''', (status, notes, calendar_id))
