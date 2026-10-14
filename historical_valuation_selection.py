"""Deterministic retained-source planning for one coordinator-requested date.

No provider discovery, adjacent-date substitution, saves or approval. Inventory
must be loaded from trusted storage under the assignment/edition transaction.
Multiple eligible receipts are deliberately ambiguous, even if amounts agree.
"""
import hashlib
import json
import re
from historical_valuation import _date
from historical_valuation_review import _baseline
from historical_valuation_originals import load_originals
from historical_valuation_evidence import annual_eps_before, daily_close_from_original
from historical_valuation_candidates import assemble_candidates


def plan_selection(records, baseline, *, cik, symbol, price_date):
    frozen = _baseline(baseline)
    _date(price_date)
    if type(cik) is not int or not 0 < cik < 10**10:
        raise ValueError('A verified issuer CIK is required.')
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}', symbol):
        raise ValueError('A verified provider symbol is required.')
    if not isinstance(records, list) or len(records) > 60:
        raise ValueError('A complete bounded retained inventory is required.')
    refs = [{k: r.get(k) for k in ('id','sha256')} for r in records if isinstance(r,dict)]
    if len(refs) != len(records):
        raise ValueError('Malformed retained inventory.')
    if records:
        load_originals(refs, records, assignment_id=frozen['assignmentId'])
    eligible = {'price': [], 'earnings': []}
    checks, inventory = [], []
    for record in sorted(records, key=lambda r:r['id']):
        receipt = {k:record[k] for k in ('id','sha256')}
        part = 'price' if record['kind']=='daily_prices' else 'earnings'
        extracted = (daily_close_from_original(record['content'],receipt,symbol,price_date,record['retrievedDate'])
                     if part=='price' else annual_eps_before(record['content'],receipt,cik,price_date))
        valid = extracted['status']=='extracted'
        if valid: eligible[part].append(receipt)
        checks.append({'source':receipt,'input':part,'eligible':valid,'message':extracted['message']})
        inventory.append({**receipt,'kind':record['kind'],'retrievedDate':record['retrievedDate']})
    # Every receipt has a diagnostic; never silently pick the first/latest source.
    gaps = [{'input':part,'code':'missing_coverage' if not rows else 'ambiguous_receipts',
             'eligibleReceipts':rows} for part,rows in eligible.items() if len(rows)!=1]
    payload = {'version':1,'baseline':frozen,'cik':cik,'symbol':symbol,'date':price_date,
               'inventory':inventory,'checks':checks,'gaps':gaps,
               'status':'needs_evidence' if any(not rows for rows in eligible.values()) else
                        ('ambiguous' if gaps else 'selected'),
               'valuationReady':False,'investorReviewRequired':True}
    if not gaps:
        selection = {'date':price_date,'priceSource':eligible['price'][0],'earningsSource':eligible['earnings'][0]}
        selected_ids = {selection['priceSource']['id'], selection['earningsSource']['id']}
        selected_records = [record for record in records if record['id'] in selected_ids]
        payload['candidate'] = assemble_candidates([selection],selected_records,frozen,cik=cik,symbol=symbol)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded)>500000:
        raise ValueError('Retained selection plan exceeds the bound.')
    return {'planDigest':hashlib.sha256(encoded).hexdigest(),'payload':json.loads(encoded)}
