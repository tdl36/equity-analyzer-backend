"""Dated investor holdings and transparent, cached market returns. No trading actions."""
import json
import math
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from flask import Blueprint, jsonify, request

PERIODS = {'1d', '1w', '1m', '3m', '6m', 'ytd', '1y'}
_CACHE = {}
_HISTORY = {}
_LOCK = threading.Lock()


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('A holdings snapshot is required.')
    name = str(data.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Enter a portfolio name (up to 100 characters).')
    try:
        as_of = date.fromisoformat(data.get('asOf', ''))
    except (TypeError, ValueError):
        raise ValueError('Enter the holdings date.')
    if as_of > date.today():
        raise ValueError('Holdings date cannot be in the future.')
    rows = data.get('holdings')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError('Enter between 1 and 100 holdings.')
    clean, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid holding.')
        ticker = str(row.get('ticker', '')).strip().upper()
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.^=-]{0,19}', ticker) or ticker in seen:
            raise ValueError('Use valid, unique market-data tickers; combine duplicate positions.')
        seen.add(ticker)
        try:
            weight = float(row.get('weight'))
        except (TypeError, ValueError):
            raise ValueError(f'{ticker}: enter weight as a percentage, e.g. 5 for 5%.')
        if not math.isfinite(weight) or weight == 0 or abs(weight) > 1000:
            raise ValueError(f'{ticker}: weight must be nonzero and between -1000% and 1000%.')
        item = {'ticker': ticker, 'weight': weight}
        for key, default in [('sector', 'Unclassified'), ('company', ticker)]:
            value = str(row.get(key, '')).strip() or default
            if len(value) > 120:
                raise ValueError(f'{ticker}: {key} is too long.')
            item[key] = value
        clean.append(item)
    return {'name': name, 'asOf': as_of.isoformat(), 'holdings': clean}


def period_return(points, period):
    """Use last observation on/before calendar boundary; never invent an IPO baseline."""
    points = sorted((d, float(v)) for d, v in points if v is not None and math.isfinite(float(v)) and float(v) > 0)
    if len(points) < 2:
        return {'changePct': None, 'issue': 'Insufficient price history'}
    end, value = points[-1]
    if period == '1d':
        start, base = points[-2]
    else:
        days = {'1w': 7, '1m': 30, '3m': 91, '6m': 182, '1y': 365}
        target = date(end.year - 1, 12, 31) if period == 'ytd' else end - timedelta(days=days[period])
        prior = [(d, v) for d, v in points if d <= target]
        if not prior or (target - prior[-1][0]).days > 7:
            return {'changePct': None, 'asOf': end.isoformat(), 'issue': 'No comparable period baseline'}
        start, base = prior[-1]
    return {'changePct': (value / base - 1) * 100, 'asOf': end.isoformat(), 'baselineDate': start.isoformat(),
            'stale': (date.today() - end).days > 4}


def market_returns(tickers, period):
    key = (tuple(sorted(tickers)), period)
    now = time.time()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < 900:
            return cached[1]
    import yfinance as yf
    history_key = tuple(sorted(tickers))
    with _LOCK:
        history = _HISTORY.get(history_key)
    if history and now - history[0] < 900:
        data = history[1]
        fetched_at = history[0]
    else:
        fetched_at = now
        data = yf.download(tickers, period='2y', interval='1d', auto_adjust=True,
                           group_by='ticker', threads=4, progress=False, timeout=12)
        if data is not None and not data.empty:
            with _LOCK:
                if len(_HISTORY) >= 16:
                    _HISTORY.pop(next(iter(_HISTORY)))
                _HISTORY[history_key] = (now, data)
    quotes = {}
    for ticker in tickers:
        try:
            frame = data[ticker] if getattr(data.columns, 'nlevels', 1) > 1 else data
            series = frame['Close'].dropna()
            quotes[ticker] = period_return([(idx.date(), v) for idx, v in series.items()], period)
        except (KeyError, TypeError, AttributeError, ValueError):
            quotes[ticker] = {'changePct': None, 'issue': 'Market data unavailable'}
    result = {'quotes': quotes, 'period': period, 'fetchedAt': datetime.fromtimestamp(fetched_at, timezone.utc).isoformat(),
              'provider': 'Yahoo Finance via yfinance', 'basis': 'Dividend- and split-adjusted daily prices; may be delayed. No extended-hours feed.'}
    if any(q.get('changePct') is not None for q in quotes.values()):
        with _LOCK:
            if len(_CACHE) >= 64:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = (now, result)
    return result


def create_blueprint(get_db):
    bp = Blueprint('portfolio_heatmap', __name__)
    ready = False
    schema_lock = threading.Lock()

    def ensure():
        nonlocal ready
        with schema_lock:
            if ready:
                return
            with get_db(commit=True) as (_, cur):
                cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('portfolio-heatmap-schema',))
                cur.execute('''CREATE TABLE IF NOT EXISTS portfolio_heatmap_snapshot (
                    id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, body JSONB NOT NULL)''')
                cur.execute("INSERT INTO portfolio_heatmap_snapshot VALUES (1, 0, '{}'::jsonb) ON CONFLICT DO NOTHING")
            ready = True

    @bp.route('/api/portfolio/heatmap/holdings', methods=['GET', 'PUT'])
    def holdings():
        ensure()
        if request.method == 'GET':
            with get_db() as (_, cur):
                cur.execute('SELECT revision, body FROM portfolio_heatmap_snapshot WHERE id=1')
                row = cur.fetchone()
            return jsonify(dict(row))
        data = request.get_json(silent=True)
        try:
            body = validate(data)
            revision = data.get('revision')
            if type(revision) is not int or revision < 0:
                raise ValueError('Reload the saved snapshot before saving.')
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        with get_db(commit=True) as (_, cur):
            cur.execute('UPDATE portfolio_heatmap_snapshot SET body=%s::jsonb, revision=revision+1 WHERE id=1 AND revision=%s RETURNING revision',
                        (json.dumps(body), revision))
            row = cur.fetchone()
            if not row:
                return jsonify(error='Holdings changed in another session. Reload before saving.'), 409
        return jsonify(body=body, revision=row['revision'])

    @bp.route('/api/portfolio/heatmap/returns', methods=['POST'])
    def returns():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify(error='Invalid request.'), 400
        tickers, period = data.get('tickers', []), data.get('period', '1d')
        if (not isinstance(tickers, list) or not 1 <= len(tickers) <= 100 or
                any(not isinstance(t, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.^=-]{0,19}', t) for t in tickers) or
                not isinstance(period, str) or period not in PERIODS):
            return jsonify(error='Choose up to 100 valid tickers and a supported period.'), 400
        try:
            return jsonify(market_returns(sorted(set(tickers)), period))
        except Exception:
            return jsonify(error='Market data is temporarily unavailable. Your holdings are saved; try again shortly.'), 503
    return bp
