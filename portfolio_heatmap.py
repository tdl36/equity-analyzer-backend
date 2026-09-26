"""Dated investor holdings and transparent, cached market returns. No trading actions."""
import json
import math
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from flask import Blueprint, jsonify, request

PERIODS = {'1d', '1w', '1m', '3m', '6m', 'ytd', '1y'}
_HISTORY = {}
_LOCK = threading.Lock()


# Public issuer holdings are transparent proxies, not official index constituent feeds.
UNIVERSES = {
    'spx': ('SPX — S&P 500', 'IVV', '239726/ishares-core-sp-500-etf'),
    'nasdaq': ('Nasdaq — Nasdaq-100', 'IQQ', '351653/ishares-nasdaq-100-etf'),
    'rlv': ('RLV — Russell 1000 Value', 'IWD', '239708/ishares-russell-1000-value-etf'),
    'rlg': ('RLG — Russell 1000 Growth', 'IWF', '239706/ishares-russell-1000-growth-etf'),
}
_UNIVERSE_CACHE = {}
_UNIVERSE_LOCK = threading.Lock()


def parse_index_holdings(text):
    """Read dated issuer CSV; keep small equity positions even if weight rounds to zero."""
    import csv
    import io
    records = list(csv.reader(io.StringIO(text.lstrip('\ufeff'))))
    as_of = None
    header_at = None
    for i, row in enumerate(records):
        if row and row[0] == 'Fund Holdings as of' and len(row) > 1:
            as_of = datetime.strptime(row[1], '%b %d, %Y').date()
        if {'Ticker', 'Name', 'Sector', 'Asset Class', 'Weight (%)', 'Market Value'}.issubset(row):
            header_at = i
            break
    if not as_of or as_of > date.today() or header_at is None:
        raise ValueError('Issuer did not provide a valid dated holdings file.')
    header = records[header_at]
    parsed, total_value, excluded = [], 0., 0
    for values in records[header_at + 1:]:
        if len(values) != len(header):
            continue
        row = dict(zip(header, values))
        try:
            value = float(row['Market Value'].replace(',', ''))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        total_value += value
        if row['Asset Class'] != 'Equity':
            continue
        ticker = re.sub(r'[ .]', '-', row['Ticker'].strip().upper())
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9-]{0,19}', ticker) or value <= 0:
            excluded += 1
            continue
        parsed.append({'ticker': ticker, 'company': row['Name'], 'sector': row['Sector'] or 'Unclassified', 'value': value})
    if total_value <= 0 or len(parsed) < 50 or len(parsed) > 1500:
        raise ValueError('Issuer holdings file is incomplete or has an unexpected format.')
    merged = {}
    for row in parsed:
        value = row.pop('value')
        if row['ticker'] in merged:
            merged[row['ticker']]['weight'] += value / total_value * 100
        else:
            merged[row['ticker']] = dict(row, weight=value / total_value * 100)
    return {'asOf': as_of.isoformat(), 'holdings': sorted(merged.values(), key=lambda r: -r['weight']),
            'excludedEquities': excluded}


def index_holdings(code):
    import requests
    if code not in UNIVERSES:
        raise ValueError('Choose SPX, Nasdaq, RLV or RLG.')
    # Serialize cache fills to avoid repeated downloads across simultaneous viewers.
    with _UNIVERSE_LOCK:
        cached = _UNIVERSE_CACHE.get(code)
        if cached and time.time() - cached[0] < 21600:
            return cached[1]
        name, proxy, path = UNIVERSES[code]
        source = 'https://www.ishares.com/us/products/' + path
        response = requests.get(source + '/latest-holdings.csv', timeout=(5, 20))
        response.raise_for_status()
        if len(response.content) > 3_000_000:
            raise ValueError('Issuer holdings file exceeds the supported size.')
        body = dict(parse_index_holdings(response.text), name=name, proxy=proxy, sourceUrl=source,
                    universe=code, basis='ETF equity holdings proxy; cash and derivatives excluded. Weights may differ from the official index.')
        body['stale'] = (date.today() - date.fromisoformat(body['asOf'])).days > 7
        _UNIVERSE_CACHE[code] = (time.time(), body)
        return body


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


# Bound provider concurrency across all heat-map requests; reuse overlapping stocks.
from concurrent.futures import ThreadPoolExecutor
_PRICE_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='heatmap-price')
_TICKER_LOCKS = {}


def _ticker_history(ticker, long_history):
    import yfinance as yf
    with _LOCK:
        lock = _TICKER_LOCKS.setdefault(ticker, threading.Lock())
    with lock:
        now = time.time()
        with _LOCK:
            cached = _HISTORY.get(ticker)
        if cached and now - cached['fetched'] < 900 and (cached['long'] or not long_history):
            return cached
        # Daily tiles need two observations, not two years of downloaded history.
        frame = yf.Ticker(ticker).history(period='2y' if long_history else '5d',
                                         interval='1d', auto_adjust=True, timeout=12)
        series = frame['Close'].dropna()
        points = [(idx.date(), float(v)) for idx, v in series.items()]
        result = {'points': points, 'fetched': time.time(), 'long': long_history}
        if points:
            with _LOCK:
                if len(_HISTORY) >= 3000:
                    _HISTORY.pop(next(iter(_HISTORY)))
                _HISTORY[ticker] = result
        return result


def market_returns(tickers, period):
    def quote(ticker):
        try:
            history = _ticker_history(ticker, period != '1d')
            result = period_return(history['points'], period)
            result['fetchedAt'] = datetime.fromtimestamp(history['fetched'], timezone.utc).isoformat()
            return result
        except Exception:
            return {'changePct': None, 'issue': 'Market data unavailable'}
    futures = {t: _PRICE_POOL.submit(quote, t) for t in sorted(set(tickers))}
    quotes = {t: f.result() for t, f in futures.items()}
    dates = [q['fetchedAt'] for q in quotes.values() if q.get('fetchedAt')]
    return {'quotes': quotes, 'period': period,
            'fetchedAt': min(dates) if dates else datetime.now(timezone.utc).isoformat(),
            'provider': 'Yahoo Finance via yfinance',
            'basis': 'Dividend- and split-adjusted daily prices; may be delayed. No extended-hours feed.'}


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

    @bp.route('/api/portfolio/heatmap/universe/<code>', methods=['GET'])
    def universe(code):
        if code not in UNIVERSES:
            return jsonify(error='Choose SPX, Nasdaq, RLV or RLG.'), 400
        try:
            return jsonify(body=index_holdings(code))
        except Exception:
            return jsonify(error='Index holdings are temporarily unavailable from the issuer. Retry loading this market.'), 503

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
