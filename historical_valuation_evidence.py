"""Read-only extraction of annual EPS from an exact-byte SEC companyfacts receipt.

Selection is limited to facts present in this response. It does not certify that
SEC coverage is complete, or that prices and earnings have compatible split bases.
"""
from decimal import Decimal
import hashlib
import json
import re
from historical_valuation import _date, _number, _source


def annual_eps_before(content, receipt, cik, price_date):
    """Select latest annual period and latest filing strictly before price date.

    Later comparative restatements are excluded. Conflicting values, start dates
    or accessions at the selected latest filing date require explicit review.
    Invalid relevant facts cannot silently cause fallback to an older EPS value.
    """
    def gap(message):
        return {'status': 'unverified', 'message': message}
    if (not isinstance(content, bytes) or len(content) > 15000000 or not _source(receipt)
            or hashlib.sha256(content).hexdigest() != receipt['sha256']):
        return gap('SEC original bytes are missing, oversized or differ from the receipt.')
    if type(cik) is not int or not 0 < cik < 10**10:
        return gap('A verified issuer CIK is required.')
    try:
        observed = _date(price_date)
        def reject_constant(value):
            raise ValueError('Nonfinite JSON')
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        raw = json.loads(content, parse_float=Decimal, parse_constant=reject_constant,
                         object_pairs_hook=unique_object)
        if not isinstance(raw, dict) or type(raw.get('cik')) is not int or raw['cik'] != cik:
            return gap('SEC issuer identity does not match the requested security.')
        facts = raw['facts']['us-gaap']['EarningsPerShareDiluted']['units']['USD/shares']
        if not isinstance(facts, list) or len(facts) > 20000:
            return gap('SEC annual EPS inventory is unavailable or exceeds the bound.')
        eligible = []
        for fact in facts:
            if not isinstance(fact, dict): return gap('Malformed SEC earnings inventory.')
            if fact.get('form') not in ('10-K', '10-K/A'): continue
            # Date failures in annual records are ambiguous, not permission to
            # choose an older value that happens to parse successfully.
            filed = _date(fact.get('filed'))
            if filed >= observed: continue
            start, end = _date(fact.get('start')), _date(fact.get('end'))
            if not 350 <= (end - start).days <= 380: continue
            if end > filed: return gap('An annual observation predates its reporting period.')
            eligible.append((end, filed, start, fact))
        if not eligible: return gap('No annual diluted EPS filing predates the price date in this original.')
        latest_end = max(x[0] for x in eligible)
        if (observed - latest_end).days > 550:
            return gap('Latest available annual period is too old for this comparison.')
        period = [x for x in eligible if x[0] == latest_end]
        latest_filed = max(x[1] for x in period)
        selected = [x for x in period if x[1] == latest_filed]
        candidates = set()
        for end, filed, start, fact in selected:
            value = fact.get('val')
            if type(value) not in (int, Decimal): return gap('Selected EPS value is not a finite numeric observation.')
            if isinstance(value, Decimal) and (not value.is_finite() or value.adjusted() > 11 or value.as_tuple().exponent < -6):
                return gap('Selected EPS precision or magnitude exceeds the bound.')
            text = format(value, 'f') if isinstance(value, Decimal) else str(value)
            number = _number(text)
            accession = fact.get('accn')
            if not isinstance(accession, str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
                return gap('Selected annual EPS has no valid filing accession.')
            candidates.add((start.isoformat(), end.isoformat(), filed.isoformat(), accession, number))
        if len(candidates) != 1:
            return gap('Latest annual filing observations conflict; no older EPS fallback was selected.')
        start, end, filed, accession, eps = candidates.pop()
    except (ValueError, TypeError, KeyError, UnicodeError, ArithmeticError, RecursionError):
        return gap('SEC original is malformed or required earnings fields are unavailable.')
    return {'status': 'extracted', 'earnings': {
        'basis': 'GAAP diluted EPS', 'periodType': 'annual', 'currency': 'USD',
        'periodStart': start, 'periodEnd': end, 'publishedDate': filed,
        'accession': accession, 'value': format(eps, 'f'), 'source': dict(receipt)},
        'message': 'Selected from this SEC response only. Security share class, split basis and filing completeness still require source review.'}


def check_sec_supplement(supplement, originals, cik):
    """Bind proposed EPS values to their SEC bytes before arithmetic diagnostics.

    This is still a pending diagnostic: price extraction, corporate actions and
    full filing-inventory coverage need their own verified adapters.
    """
    from historical_valuation import check_supplement
    result = check_supplement(supplement, originals)
    for index, row in enumerate(supplement['observations']):
        diagnosis = result['observations'][index]
        if diagnosis['status'] not in ('calculated', 'not_meaningful'):
            continue
        earnings = row['earnings']; receipt = earnings['source']
        extracted = annual_eps_before(originals.get(receipt['id']), receipt, cik, row['price']['date'])
        message = extracted.get('message')
        if extracted['status'] == 'extracted':
            expected = extracted['earnings']
            # Decimal equality tolerates presentation precision, never a changed
            # financial amount. Other metadata must match the selected fact.
            equal = all(earnings.get(key) == value for key, value in expected.items() if key != 'value')
            equal = equal and _number(earnings['value']) == _number(expected['value'])
            if equal:
                diagnosis['earningsExtraction'] = 'matched_sec_annual_fact'
                diagnosis['message'] += ' EPS matches the selected SEC fact; price extraction and corporate actions remain unverified.'
                continue
            message = 'Proposed EPS or filing metadata differs from the selected historical SEC observation.'
        result['observations'][index] = {'index': index, 'status': 'unverified', 'message': message}
    return result


def daily_close_from_original(content, receipt, symbol, price_date, retrieved_date):
    """Extract a completed daily raw close from Alpha Vantage DAILY JSON bytes.

    No network access or account purchase. Only the unadjusted DAILY response
    shape is supported. Currency, issuer/share class and corporate-action
    compatibility must be bound separately; a matching ticker does not prove them.
    Provider contract: https://www.alphavantage.co/documentation/#daily
    """
    def gap(message):
        return {'status': 'unverified', 'message': message}
    if (not isinstance(content, bytes) or len(content) > 15000000 or not _source(receipt)
            or hashlib.sha256(content).hexdigest() != receipt['sha256']):
        return gap('Price original bytes are missing, oversized or differ from the receipt.')
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        return gap('An explicit provider symbol is required.')
    try:
        observed, retrieved = _date(price_date), _date(retrieved_date)
        if observed >= retrieved:
            return gap('A prior completed trading date is required; same-day finality is unresolved.')
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        def reject_constant(value):
            raise ValueError('Nonfinite JSON')
        raw = json.loads(content, object_pairs_hook=unique_object,
                         parse_float=Decimal, parse_constant=reject_constant)
        if not isinstance(raw, dict) or set(raw) != {'Meta Data', 'Time Series (Daily)'}:
            return gap('A successful raw daily price response is required.')
        meta, series = raw['Meta Data'], raw['Time Series (Daily)']
        if (not isinstance(meta, dict) or meta.get('2. Symbol') != symbol
                or meta.get('1. Information') != 'Daily Prices (open, high, low, close) and Volumes'):
            return gap('Price symbol or raw daily series identity does not match.')
        refreshed = _date(meta.get('3. Last Refreshed'))
        if not observed <= refreshed <= retrieved:
            return gap('Requested price date is outside the response vintage.')
        if not isinstance(series, dict) or not 1 <= len(series) <= 15000:
            return gap('Daily price inventory is missing or exceeds the bound.')
        # Validate the inventory dates rather than substituting a nearby close.
        for day in series:
            if _date(day) > refreshed:
                return gap('Price inventory contains a date beyond its response vintage.')
        row = series.get(price_date)
        fields = {'1. open', '2. high', '3. low', '4. close', '5. volume'}
        if not isinstance(row, dict) or set(row) != fields:
            return gap('Exact date raw OHLCV is missing; no adjusted or adjacent-day fallback.')
        opening, high, low, close = (_number(row[key]) for key in
                                    ('1. open', '2. high', '3. low', '4. close'))
        if not 0 < low <= min(opening, close) <= max(opening, close) <= high:
            return gap('Daily price values are nonpositive or internally inconsistent.')
        if not isinstance(row['5. volume'], str) or not re.fullmatch(r'[1-9][0-9]{0,15}', row['5. volume']):
            return gap('A positive bounded trading volume is required.')
    except (ValueError, TypeError, KeyError, UnicodeError, ArithmeticError, RecursionError):
        return gap('Price original is malformed or required daily fields are unavailable.')
    return {'status': 'extracted', 'price': {'symbol': symbol, 'date': price_date,
            'basis': 'unadjusted_close', 'value': format(close, 'f'), 'source': dict(receipt)},
            'message': 'Raw close extracted from this original only. Currency, issuer/share class, source provenance and split compatibility remain unverified.'}


def check_market_supplement(supplement, originals, cik, symbol, price_retrieval_dates):
    """Reconcile both financial inputs to originals, without certifying valuation.

    Retrieval dates are supplied by a trusted receipt inventory, keyed by source
    ID, not by the proposed observation. Currency/security mapping and corporate
    actions still require separate evidence. No network or research mutation.
    """
    if type(cik) is not int or not 0 < cik < 10**10:
        raise ValueError('A bounded issuer CIK is required for the evidence context.')
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        raise ValueError('A bounded provider symbol is required for the evidence context.')
    if not isinstance(price_retrieval_dates, dict) or len(price_retrieval_dates) > 60:
        raise ValueError('A bounded trusted price-receipt retrieval-date inventory is required.')
    for source_id, retrieved in price_retrieval_dates.items():
        if not isinstance(source_id, str) or not 0 < len(source_id) <= 200:
            raise ValueError('Invalid source identity in retrieval inventory.')
        _date(retrieved)
    result = check_sec_supplement(supplement, originals, cik)
    result['valuationReady'] = False
    result['unresolvedChecks'] = ['security_currency_mapping', 'corporate_actions',
                                  'filing_inventory_completeness', 'investor_review']
    for index, row in enumerate(supplement['observations']):
        diagnosis = result['observations'][index]
        if diagnosis.get('earningsExtraction') != 'matched_sec_annual_fact':
            continue
        price = row['price']; receipt = price['source']
        extracted = daily_close_from_original(originals.get(receipt['id']), receipt,
                                              symbol, price['date'],
                                              price_retrieval_dates.get(receipt['id']))
        if extracted['status'] != 'extracted':
            result['observations'][index] = {'index': index, 'status': 'unverified',
                                             'message': extracted['message']}
            continue
        expected = extracted['price']
        if (price.get('basis') != expected['basis'] or price.get('date') != expected['date']
                or _number(price['value']) != _number(expected['value'])):
            result['observations'][index] = {'index': index, 'status': 'unverified',
                'message': 'Proposed price differs from the exact-date raw close in the original.'}
            continue
        diagnosis['priceExtraction'] = 'matched_raw_daily_close'
        diagnosis['message'] = ('Both amounts match their source observations. Currency/security '
                                'mapping, corporate actions and complete filing coverage remain '
                                'unverified; this is not a validated historical valuation.')
    # Freeze diagnostics with the actual bytes and trusted retrieval context.
    # The proposal digest alone cannot identify a replay under different evidence.
    referenced = set()
    for row in supplement['observations']:
        if not isinstance(row, dict): continue
        for part in ('price', 'earnings'):
            value = row.get(part)
            receipt = value.get('source') if isinstance(value, dict) else None
            if _source(receipt): referenced.add(receipt['id'])
    evidence = {'adapterVersion': 1, 'proposalDigest': result['digest'],
                'cik': cik, 'symbol': symbol,
                'priceRetrievalDates': dict(price_retrieval_dates),
                'originalHashes': {key: hashlib.sha256(originals[key]).hexdigest()
                                   if isinstance(originals.get(key), bytes) else None
                                   for key in sorted(referenced)}}
    result['evidenceContext'] = evidence
    result['evidenceDigest'] = hashlib.sha256(json.dumps(
        evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result
