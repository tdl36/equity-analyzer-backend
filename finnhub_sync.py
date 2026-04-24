"""Finnhub earnings calendar sync.

Nightly cron calls fetch_and_upsert() which:
  1. Pulls the list of covered tickers from analysts.coverage_tickers
  2. Queries Finnhub's /calendar/earnings endpoint for the next 60 days
  3. Upserts into earnings_calendar keyed on (ticker, quarter_label)

Free tier: 60 calls/min. We make one call covering all tickers in a window,
then filter client-side, so the entire run is a single API call.

API key lives in app_settings.finnhub_api_key — set via Settings UI.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime, timedelta

import app_v3


FINNHUB_BASE = 'https://finnhub.io/api/v1'


def _api_key() -> str | None:
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("SELECT value FROM app_settings WHERE key='finnhub_api_key'")
            row = cur.fetchone()
        if row and row['value']:
            return str(row['value']).strip()
    except Exception as e:
        print(f'finnhub_sync: api key lookup failed: {e}')
    return None


def _covered_tickers() -> list[str]:
    """Return sorted, deduplicated tickers across all analyst coverage."""
    with app_v3.get_db() as (_c, cur):
        cur.execute('SELECT coverage_tickers FROM analysts')
        rows = cur.fetchall() or []
    seen: set[str] = set()
    for r in rows:
        for t in (r.get('coverage_tickers') or []):
            if t:
                seen.add(str(t).upper())
    return sorted(seen)


def _quarter_label(dt: date) -> str:
    """Map a report date to a fiscal-quarter label (calendar quarter). The
    actual fiscal quarter may differ for some companies, but for UI labeling
    this is good-enough until we wire per-ticker fiscal-calendar metadata."""
    q = (dt.month - 1) // 3 + 1
    return f"{q}Q{dt.year % 100:02d}"


def fetch_upcoming(days_ahead: int = 120, days_back: int = 7, api_key: str | None = None) -> list[dict]:
    """Bulk calendar fetch — returns whatever Finnhub has for the window.
    Free tier occasionally under-returns names here, so sync() prefers
    per-symbol fetches (see fetch_for_symbol). Kept for backwards compat."""
    key = api_key or _api_key()
    if not key:
        raise RuntimeError('Finnhub API key not set. Save to Settings -> API Keys -> Finnhub.')
    today = datetime.utcnow().date()
    frm = (today - timedelta(days=max(0, days_back))).isoformat()
    to = (today + timedelta(days=days_ahead)).isoformat()
    qs = urllib.parse.urlencode({'from': frm, 'to': to, 'token': key})
    url = f'{FINNHUB_BASE}/calendar/earnings?{qs}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Charlie/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f'Finnhub fetch failed: {e}')
    return data.get('earningsCalendar') or []


def fetch_for_symbol(symbol: str, days_ahead: int = 120, days_back: int = 7,
                      api_key: str | None = None) -> list[dict]:
    """Per-symbol calendar fetch — more reliable than the bulk endpoint on
    the free tier. Returns Finnhub's earningsCalendar entries for this
    symbol within the window."""
    key = api_key or _api_key()
    if not key:
        raise RuntimeError('Finnhub API key not set.')
    today = datetime.utcnow().date()
    frm = (today - timedelta(days=max(0, days_back))).isoformat()
    to = (today + timedelta(days=days_ahead)).isoformat()
    qs = urllib.parse.urlencode({'from': frm, 'to': to, 'symbol': symbol, 'token': key})
    url = f'{FINNHUB_BASE}/calendar/earnings?{qs}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Charlie/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('earningsCalendar') or []
    except Exception as e:
        print(f'finnhub_sync: per-symbol fetch failed for {symbol}: {e}')
        return []


def upsert_entries(entries: list[dict], only_tickers: set[str] | None = None) -> dict:
    """Upsert earnings_calendar rows. only_tickers limits to covered names."""
    stats = {'considered': 0, 'skipped': 0, 'upserted': 0, 'errors': 0}
    with app_v3.get_db(commit=True) as (_c, cur):
        for e in entries:
            stats['considered'] += 1
            try:
                ticker = (e.get('symbol') or '').upper().strip()
                report_date = e.get('date')  # YYYY-MM-DD
                if not ticker or not report_date:
                    stats['skipped'] += 1
                    continue
                if only_tickers is not None and ticker not in only_tickers:
                    stats['skipped'] += 1
                    continue
                try:
                    rd = datetime.strptime(report_date, '%Y-%m-%d').date()
                except ValueError:
                    stats['skipped'] += 1
                    continue
                # hour: 'bmo' | 'amc' | 'dmh' (during market hours) | ''
                hour = (e.get('hour') or '').lower()
                timing = 'BMO' if hour == 'bmo' else ('AMC' if hour == 'amc' else None)
                quarter_label = _quarter_label(rd)

                cur.execute('''
                    INSERT INTO earnings_calendar
                        (id, ticker, quarter_label, expected_date, confirmed_date, timing, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'upcoming')
                    ON CONFLICT (ticker, quarter_label) DO UPDATE SET
                        expected_date = EXCLUDED.expected_date,
                        confirmed_date = COALESCE(EXCLUDED.confirmed_date, earnings_calendar.confirmed_date),
                        timing = COALESCE(EXCLUDED.timing, earnings_calendar.timing),
                        updated_at = NOW()
                      WHERE earnings_calendar.status IN ('upcoming', 'fetching', 'retry')
                ''', (str(uuid.uuid4()), ticker, quarter_label, rd, rd, timing))
                stats['upserted'] += 1
            except Exception as exc:
                stats['errors'] += 1
                print(f'finnhub_sync upsert error for {e}: {exc}')
    return stats


def sync(days_ahead: int = 120, days_back: int = 7) -> dict:
    """Full sync: iterate covered tickers and fetch each individually from
    Finnhub (more reliable than the bulk endpoint on free tier). Paces
    requests at ~1 rps to stay under 60 calls/min.

    Default 127-day window catches the full next-quarter cycle plus
    recently-reported names for backfill."""
    import time as _time
    tickers = sorted(set(_covered_tickers()))
    if not tickers:
        return {'ok': False, 'reason': 'no covered tickers'}
    key = _api_key()
    if not key:
        raise RuntimeError('Finnhub API key not set. Save to Settings -> API Keys -> Finnhub.')

    all_entries: list[dict] = []
    matched: list[str] = []
    unmatched: list[str] = []
    for i, t in enumerate(tickers):
        rows = fetch_for_symbol(t, days_ahead=days_ahead, days_back=days_back, api_key=key)
        if rows:
            all_entries.extend(rows)
            matched.append(t)
        else:
            unmatched.append(t)
        # Pace: stay under 60/min. 1.1s between calls.
        if i < len(tickers) - 1:
            _time.sleep(1.1)

    stats = upsert_entries(all_entries, only_tickers=set(tickers))
    stats['coverage_tickers'] = len(tickers)
    stats['window_days'] = days_ahead + days_back
    stats['days_ahead'] = days_ahead
    stats['days_back'] = days_back
    stats['covered_matched'] = matched
    stats['covered_unmatched'] = unmatched
    stats['ran_at'] = datetime.utcnow().isoformat()
    try:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute('''
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ('finnhub_last_sync', %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
            ''', (json.dumps(stats),))
    except Exception as e:
        print(f'finnhub_sync: last-sync recording failed: {e}')
    return stats
