"""Bounded SEC original ingestion for an existing historical review target.

No model calls. Prices require a separately configured provider ingestion path.
No caller-supplied URLs, replacement of originals, or investor approval.
"""
import hashlib
import json
from datetime import datetime, timezone
import requests
from historical_valuation_context import load_context
from historical_valuation_source_store import save_original


def ingest_sec(get_db, aid, eid, *, user_agent, session=None, retrieval_id=None):
    if not isinstance(user_agent, str) or not user_agent.strip() or '\n' in user_agent or '\r' in user_agent:
        raise ValueError('A configured SEC identification header is required.')
    with get_db() as (_, cur):
        before, identity = load_context(cur, aid, eid)
        if retrieval_id:
            from historical_valuation_retrieval import preflight_receipt
            preflight_receipt(cur, retrieval_id, before, {'kind':'sec_companyfacts'})
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity['cik']:010d}.json"
    client = session or requests
    # Release database locks during network access, then recheck the frozen target.
    try:
        with client.get(url, headers={'User-Agent': user_agent, 'Accept':'application/json'},
                        timeout=(10,30), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError('SEC retrieval failed; no original was recorded.')
            parts, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > 15000000:
                    raise ValueError('SEC original exceeds the historical evidence bound.')
                parts.append(chunk)
            raw = b''.join(parts)
    except requests.RequestException:
        # Provider exception strings can contain headers/URLs; do not propagate.
        raise ValueError('SEC retrieval unavailable; inspect connectivity and the retrieval outcome without automatically retrying.') from None
    try:
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out: raise ValueError('Duplicate SEC JSON key.')
                out[key] = value
            return out
        data = json.loads(raw, object_pairs_hook=unique)
        if (not isinstance(data, dict) or type(data.get('cik')) is not int
                or data['cik'] != identity['cik'] or not isinstance(data.get('facts'), dict)
                or not data['facts']):
            raise ValueError('SEC issuer or facts missing.')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError('SEC response did not verify the saved issuer and facts.') from None
    sha = hashlib.sha256(raw).hexdigest()
    # Include trusted retrieval date so a later vintage cannot reuse an old date.
    day = datetime.now(timezone.utc).date().isoformat()
    record = dict(assignmentId=aid, id=f'sec-companyfacts:{sha}:{day}', sha256=sha,
                  kind='sec_companyfacts', retrievedDate=day, content=raw)
    with get_db(commit=True) as (_, cur):
        after, current_identity = load_context(cur, aid, eid)
        if before != after or identity != current_identity:
            raise ValueError('The target changed during retrieval; no original was recorded.')
        receipt = save_original(cur, record)
        result = {**receipt, 'kind':'sec_companyfacts', 'retrievedDate':day, 'sourceUrl':url,
                  'caveat':'Current SEC response retained. Historical filing completeness and share basis remain unverified.'}
        if retrieval_id:
            from historical_valuation_retrieval import retain_receipt
            retain_receipt(cur,retrieval_id,before,result)
    return result


def ingest_prices(get_db, aid, eid, price_date, *, api_key, session=None, retrieval_id=None):
    """Retrieve compact raw daily prices; no premium escalation or fallback.

    Missing older dates are actionable coverage gaps. A later provider selection
    may add full history only under an explicit entitlement/cost policy.
    """
    import re
    from historical_valuation import _date
    from historical_valuation_evidence import daily_close_from_original
    day = datetime.now(timezone.utc).date().isoformat()
    if _date(price_date) >= _date(day):
        raise ValueError('Choose a prior completed trading date.')
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError('Configure the price-provider credential before retrieval.')
    with get_db() as (_, cur):
        before, identity = load_context(cur, aid, eid)
        if retrieval_id:
            from historical_valuation_retrieval import preflight_receipt
            preflight_receipt(cur, retrieval_id, before, {'kind':'daily_prices', 'priceDate':price_date})
    symbol = identity['symbol']
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        raise ValueError('Saved provider symbol is invalid.')
    client = session or requests
    try:
        with client.get('https://www.alphavantage.co/query',
                        params={'function':'TIME_SERIES_DAILY','symbol':symbol,
                                'outputsize':'compact','datatype':'json','apikey':api_key},
                        timeout=(10,30), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError('Price retrieval failed; no original recorded.')
            parts, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > 15000000:
                    raise ValueError('Price original exceeds the historical evidence bound.')
                parts.append(chunk)
            raw = b''.join(parts)
    except requests.RequestException:
        raise ValueError('Price retrieval unavailable; inspect provider access without automatically retrying.') from None
    sha = hashlib.sha256(raw).hexdigest()
    receipt = {'id':f'daily-prices:{sha}:{day}', 'sha256':sha}
    extracted = daily_close_from_original(raw, receipt, symbol, price_date, day)
    if extracted['status'] != 'extracted':
        raise ValueError('Requested raw daily price could not be verified. Check date coverage and provider access; no premium or adjusted-price fallback was used.')
    record = dict(receipt, assignmentId=aid, kind='daily_prices', retrievedDate=day, content=raw)
    with get_db(commit=True) as (_, cur):
        after, current_identity = load_context(cur, aid, eid)
        if before != after or identity != current_identity:
            raise ValueError('The target changed during retrieval; no original was recorded.')
        saved = save_original(cur, record)
        result = {**saved, 'kind':'daily_prices', 'retrievedDate':day,
                  'sourceUrl':'https://www.alphavantage.co/documentation/#daily',
                  'coverage':'compact', 'priceDate':price_date,
                  'caveat':'Exact raw daily response retained. Share class, currency and corporate-action compatibility remain unverified.'}
        if retrieval_id:
            from historical_valuation_retrieval import retain_receipt
            retain_receipt(cur,retrieval_id,before,result)
    return result
