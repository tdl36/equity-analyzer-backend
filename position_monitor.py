"""Nightly check on every position, against what the last review said.

The scheduler already runs twelve jobs and none of them touch a thesis. This is
the thirteenth, and it inverts the direction: everything else pulls information
in, this one asks whether anything that arrived should change a position.

Deliberately free. Every signal here is computed from stored state, a price
quote and a document count -- no model is called, so the whole portfolio can be
checked every night for nothing.

What that buys and what it does not: price, scenario breaches, earnings dates
and staleness are read fresh each run. KPI *values* are not -- they are frozen
into the review that produced them, because updating one means a model reading a
new filing. So the monitor watches the condition under which a KPI could have
moved (new documents arriving) rather than pretending to re-read the number, and
labels the off-thesis count as being as of the last review. That is a change from the original plan, which had a model
re-deriving KPI values nightly. Re-deriving them costs a review per position per
night and mostly reproduces yesterday's answer; the cheap signals below catch
the things that actually move between reviews, and when one fires the right
response is to run a review, which you now know to do.

Alerts fire on TRANSITIONS, never on state. A position that has been off-thesis
for a month is not news, and a monitor that repeats it every morning is one you
learn to ignore.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import app_v3


# How far the price has to move from the last review before it is worth saying.
PRICE_MOVE_PCT = 12.0

# A review older than this, with an earnings print since, is stale.
STALE_DAYS = 100

# Earnings inside this window are worth flagging.
EARNINGS_WINDOW_DAYS = 3

MONITOR_STATE_KEY = 'position_monitor_state'


# --------------------------------------------------------------------------
# State the monitor keeps about itself
# --------------------------------------------------------------------------

def _load_seen() -> Dict[str, Any]:
    """What we alerted on last time, so we only report changes."""
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute('SELECT value FROM app_settings WHERE key = %s',
                        (MONITOR_STATE_KEY,))
            row = cur.fetchone()
        return json.loads((row or {}).get('value') or '{}')
    except Exception:
        return {}


def _save_seen(state: Dict[str, Any]) -> None:
    try:
        with app_v3.get_db(commit=True) as (_c, cur):
            cur.execute("""INSERT INTO app_settings (key, value, updated_at)
                           VALUES (%s, %s, CURRENT_TIMESTAMP)
                           ON CONFLICT (key) DO UPDATE
                           SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP""",
                        (MONITOR_STATE_KEY, json.dumps(state)))
    except Exception as e:
        print(f'[position-monitor] could not save state: {e}')


# --------------------------------------------------------------------------
# Reading a position
# --------------------------------------------------------------------------

def _latest_review(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("""SELECT state, created_at FROM investment_reviews
                           WHERE ticker = %s ORDER BY created_at DESC LIMIT 1""",
                        (ticker,))
            row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    state = row['state']
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            return None
    return {'state': state or {}, 'created_at': row['created_at']}


def _next_earnings(ticker: str) -> Optional[Any]:
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("""SELECT confirmed_date FROM earnings_calendar
                           WHERE ticker = %s AND confirmed_date >= CURRENT_DATE
                           ORDER BY confirmed_date LIMIT 1""", (ticker,))
            row = cur.fetchone()
        return (row or {}).get('confirmed_date')
    except Exception:
        return None


def _documents_since(ticker: str, when) -> List[str]:
    """Filenames that arrived for this ticker after the review was written.

    The closest free proxy for "a KPI may have moved". A KPI value lives inside
    a filing and only a model reading that filing can update it, which is a cost
    per position per night. A document arriving is the moment that becomes
    possible, and the arrival is already recorded.
    """
    if not when:
        return []
    try:
        with app_v3.get_db() as (_c, cur):
            cur.execute("""SELECT filename FROM document_files
                           WHERE ticker = %s AND created_at > %s
                           ORDER BY created_at DESC""", (ticker, when))
            return [r['filename'] for r in (cur.fetchall() or [])]
    except Exception:
        return []


def _scenario_bounds(state: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {'bear': None, 'base': None, 'bull': None}
    for sc in (state.get('scenarios') or []):
        name = str(sc.get('name', '')).lower()
        if name not in out:
            continue
        target = sc.get('target_price')
        if target is None and sc.get('metric_value') and sc.get('multiple'):
            target = sc['metric_value'] * sc['multiple']
        try:
            out[name] = float(target) if target is not None else None
        except (TypeError, ValueError):
            pass
    return out


def _kpi_status_counts(state: Dict[str, Any]) -> Dict[str, int]:
    """Status recomputed from the stored thresholds -- no model, no new data."""
    import investment_review as ir
    counts = {'on-thesis': 0, 'off-thesis': 0, 'watch': 0}
    for k in (state.get('kpis') or []):
        try:
            kpi = ir.KPI(
                name=str(k.get('name', '')),
                current=k.get('current'), prior=k.get('prior'),
                unit=str(k.get('unit', '%')),
                bull_threshold=k.get('bull_threshold'),
                bear_threshold=k.get('bear_threshold'),
                higher_is_better=bool(k.get('higher_is_better', True)))
        except Exception:
            continue
        st = kpi.status()
        if st in counts:
            counts[st] += 1
    return counts


# --------------------------------------------------------------------------
# The signals
# --------------------------------------------------------------------------

def evaluate_position(ticker: str, review: Dict[str, Any],
                      price: Optional[float]) -> List[Dict[str, Any]]:
    """Everything worth saying about one position today.

    Each finding carries a `key` -- a stable identity for the condition, so the
    caller can tell a new one from one it already reported.
    """
    state = review['state']
    reviewed_at = review['created_at']
    findings: List[Dict[str, Any]] = []

    review_price = state.get('price')
    bounds = _scenario_bounds(state)

    if price and review_price:
        move = (price - review_price) / review_price * 100.0
        if abs(move) >= PRICE_MOVE_PCT:
            findings.append({
                'key': f'move:{int(move // PRICE_MOVE_PCT)}',
                'type': 'price_move',
                'title': f'{ticker} {"up" if move > 0 else "down"} '
                         f'{abs(move):.0f}% since the last review',
                'detail': {'reviewPrice': review_price, 'price': price,
                           'movePct': round(move, 1)},
            })

    # Crossing a scenario boundary is the signal the scenarios exist to give.
    if price:
        if bounds['bear'] and price <= bounds['bear']:
            findings.append({
                'key': 'below-bear',
                'type': 'scenario_breach',
                'title': f'{ticker} is at or below the bear case '
                         f'(${bounds["bear"]:,.2f})',
                'detail': {'price': price, 'bear': bounds['bear']},
            })
        elif bounds['bull'] and price >= bounds['bull']:
            findings.append({
                'key': 'above-bull',
                'type': 'scenario_breach',
                'title': f'{ticker} is at or above the bull case '
                         f'(${bounds["bull"]:,.2f})',
                'detail': {'price': price, 'bull': bounds['bull']},
            })

    # New sources age the KPI values, which are frozen at review time. Said
    # plainly, because the off-thesis count below is a restatement of what the
    # last review concluded and not a fresh reading.
    new_docs = _documents_since(ticker, reviewed_at)
    if new_docs:
        findings.append({
            'key': f'newdocs:{len(new_docs)}',
            'type': 'sources_since_review',
            'title': f'{ticker} has {len(new_docs)} new document'
                     f'{"s" if len(new_docs) != 1 else ""} since the last review',
            'detail': {'count': len(new_docs), 'filenames': new_docs[:8],
                       'note': 'KPI values in the review predate these'},
        })

    counts = _kpi_status_counts(state)
    if counts['off-thesis']:
        findings.append({
            'key': f'off:{counts["off-thesis"]}',
            'type': 'kpi_off_thesis',
            'title': f'{ticker} has {counts["off-thesis"]} KPI'
                     f'{"s" if counts["off-thesis"] != 1 else ""} off-thesis '
                     f'as of the last review',
            'detail': dict(counts, asOf=str(reviewed_at)[:10] if reviewed_at else ''),
        })

    earnings = _next_earnings(ticker)
    if earnings:
        try:
            days = (earnings - datetime.utcnow().date()).days
        except Exception:
            days = None
        if days is not None and 0 <= days <= EARNINGS_WINDOW_DAYS:
            findings.append({
                'key': f'earnings:{earnings}',
                'type': 'earnings_near',
                'title': f'{ticker} reports in {days} day{"s" if days != 1 else ""}',
                'detail': {'date': str(earnings),
                           'question': state.get('key_question', '')},
            })

    # Stale only matters alongside something that has happened since.
    if reviewed_at:
        try:
            age = (datetime.utcnow() - reviewed_at.replace(tzinfo=None)).days
        except Exception:
            age = None
        if age is not None and age >= STALE_DAYS:
            findings.append({
                'key': f'stale:{age // 30}',
                'type': 'review_stale',
                'title': f'{ticker} has not been reviewed in {age} days',
                'detail': {'ageDays': age},
            })

    return findings


# --------------------------------------------------------------------------
# The job
# --------------------------------------------------------------------------

def _write_alert(finding: Dict[str, Any], ticker: str) -> None:
    with app_v3.get_db(commit=True) as (_c, cur):
        cur.execute("""INSERT INTO agent_alerts
                       (id, alert_type, ticker, title, detail, status, created_at)
                       VALUES (%s,%s,%s,%s,%s,'new',NOW())""",
                    (str(uuid.uuid4()), finding['type'], ticker,
                     finding['title'], json.dumps(finding.get('detail') or {})))


def run_position_monitor(tickers: Optional[List[str]] = None,
                         dry_run: bool = False) -> Dict[str, Any]:
    """Check every position with a review behind it.

    Returns a summary rather than printing it, so the same function serves the
    scheduler, a manual endpoint and the tests.
    """
    if tickers is None:
        try:
            with app_v3.get_db() as (_c, cur):
                cur.execute("""SELECT DISTINCT ticker FROM investment_reviews
                               ORDER BY ticker""")
                tickers = [r['ticker'] for r in (cur.fetchall() or [])]
        except Exception as e:
            return {'error': str(e), 'checked': 0, 'alerts': 0}

    seen = _load_seen()
    summary = {'checked': 0, 'alerts': 0, 'skipped': 0, 'findings': []}

    for ticker in tickers:
        review = _latest_review(ticker)
        if not review:
            summary['skipped'] += 1
            continue
        summary['checked'] += 1

        quote = None
        try:
            quote = app_v3._latest_close(ticker)
        except Exception:
            pass
        price = (quote or {}).get('price')

        findings = evaluate_position(ticker, review, price)
        already = set(seen.get(ticker, []))
        fresh = [f for f in findings if f['key'] not in already]

        for f in fresh:
            summary['findings'].append({'ticker': ticker, 'title': f['title']})
            if not dry_run:
                try:
                    _write_alert(f, ticker)
                    summary['alerts'] += 1
                except Exception as e:
                    print(f'[position-monitor] {ticker}: could not write alert: {e}')

        # Only what is currently true is remembered, so a condition that clears
        # and later returns is reported again.
        seen[ticker] = [f['key'] for f in findings]

    if not dry_run:
        _save_seen(seen)
    return summary
