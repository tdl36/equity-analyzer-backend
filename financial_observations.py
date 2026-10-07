"""Bounded, analyst-confirmed revenue and positive EBITDA observations from frozen research evidence.

Passage/number matching is mechanical. Metric meaning, period, currency and
accounting basis remain an explicit analyst attestation, never an AI fact badge.
"""
import hashlib
import re
import uuid
from decimal import Decimal

SCALES = {'units': Decimal('0.000001'), 'thousands': Decimal('0.001'),
          'millions': Decimal(1), 'billions': Decimal(1000)}
TOKEN = re.compile(r'(?<![\w.,+\-(])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.,)]|\s*%)')


def numbers(excerpt):
    # Preserve the exact printed token. No arithmetic, inferred ranges or locale conversion.
    return list(dict.fromkeys(m.group() for m in TOKEN.finditer(excerpt)))[:100]


def selection(data, metric="revenue"):
    from operating_model import number, text
    if metric not in ('revenue', 'ebitda'):
        raise ValueError('Unsupported financial observation.')
    label = 'consolidated annual revenue' if metric == 'revenue' else 'consolidated annual positive EBITDA'
    if not isinstance(data, dict):
        raise ValueError('Choose a revenue observation.')
    if data.get('confirmed') is not True:
        raise ValueError(f'Confirm issuer, {label}, period, currency and accounting basis.')
    year = number(data.get('fiscalYear'), 'observation fiscal year', 1900, 2200)
    if year != year.to_integral_value():
        raise ValueError('Observation fiscal year must be a whole number.')
    currency = text(data.get('currency'), 'observation currency', 3).upper()
    if not re.fullmatch('[A-Z]{3}', currency):
        raise ValueError('Use a three-letter observation currency.')
    unit = data.get('unit')
    if unit not in SCALES:
        raise ValueError('Choose the source unit: units, thousands, millions or billions.')
    token = text(data.get('token'), 'exact source number', 32)
    if not TOKEN.fullmatch(token):
        raise ValueError('Select a positive printed number, without a currency symbol or percent.')
    value = number(token.replace(',', ''), 'source '+metric, '.000001', '1000000000000000')
    normalized = value * SCALES[unit]
    number(str(normalized), metric+' in millions', '.000001', '1000000000')
    return {'researchRunId': str(uuid.UUID(data.get('researchRunId', ''))),
            'claimId': text(data.get('claimId'), 'claim identifier', 100),
            'sourceId': text(data.get('sourceId'), 'source identifier', 100),
            'excerptHash': text(data.get('excerptHash'), 'passage identity', 64),
            'token': token, 'unit': unit, 'fiscalYear': int(year), 'currency': currency,
            'basis': text(data.get('basis'), 'revenue accounting basis / definition', 1800),
            'locator': text(data.get('locator'), 'page or section locator', 300),
            'confirmed': True, 'valueMillions': str(normalized)}


def candidates(run, ticker):
    if not run or run['ticker'] != ticker or run['status'] != 'complete':
        raise ValueError('Choose a completed research run for this company.')
    sources = {s['id']: s for s in run['sources']}
    result = []
    seen = set()
    for section in run['state'].get('sections', []):
        for claim in section.get('claims', []):
            if claim.get('basis') != 'reported_fact' or claim.get('review') != 'supported':
                continue
            for ref in claim.get('evidence', []):
                source = sources.get(ref.get('sourceId'))
                excerpt = ref.get('excerpt', '')
                if not source or not ref.get('matched') or len(' '.join(excerpt.split())) < 30:
                    continue
                if ' '.join(excerpt.split()) not in ' '.join(source['text'].split()):
                    continue
                tokens = numbers(excerpt)
                signature = hashlib.sha256(excerpt.encode()).hexdigest()
                if not tokens or (source['id'], signature) in seen:
                    continue
                seen.add((source['id'], signature))
                result.append({'researchRunId': run['id'], 'claimId': claim['id'],
                    'sourceId': source['id'], 'excerptHash': signature,
                    'statement': claim['statement'], 'excerpt': excerpt, 'numbers': tokens,
                    'filename': source['filename'], 'sourceUrl': source.get('sourceUrl', ''),
                    'originalHash': source['originalHash'], 'extractionHash': source['extractionHash']})
    return result


def resolve(data, ticker, cur, metric="revenue"):
    from company_research import eligible
    from command_thesis_bridge import file_hash
    chosen = selection(data, metric)
    cur.execute("SELECT to_regclass('company_research_runs') AS name")
    if not cur.fetchone()['name']:
        raise ValueError('Complete source-backed research before linking a revenue observation.')
    cur.execute('SELECT * FROM company_research_runs WHERE id=%s', (chosen['researchRunId'],))
    run = cur.fetchone()
    options = candidates(run, ticker)
    match = next((c for c in options if all(c[k] == chosen[k] for k in ('claimId','sourceId','excerptHash'))), None)
    if not match or chosen['token'] not in match['numbers']:
        raise ValueError('The number or passage is not eligible reviewed evidence. Reload the source choices.')
    cur.execute('SELECT filename,file_data,metadata FROM document_files WHERE ticker=%s AND filename=%s FOR SHARE',
                (ticker, match['filename']))
    document = cur.fetchone()
    if not document or not eligible(document.get('metadata') or {}):
        raise ValueError('This original is missing or restricted. Restore eligible evidence or unlink the observation.')
    if file_hash(document) != match['originalHash']:
        raise ValueError('The stored original changed. Start fresh research or unlink this historical observation.')
    return {**chosen, **{k: match[k] for k in ('filename','sourceUrl','originalHash','extractionHash','excerpt')},
            'ticker': ticker, 'metric': 'consolidated_annual_'+metric,
            'checks': ['passage_matched', 'printed_number_matched', 'unit_conversion_checked'],
            'interpretation': 'Issuer, metric meaning, fiscal year, currency and accounting basis are analyst-confirmed, not independently verified.'}


def validate_model_link(model, metric="revenue"):
    """Reject stale links when the analyst edits the model's base inputs."""
    key = 'baseRevenue' if metric == 'revenue' else 'baseEbitda'
    observation = selection(model[key+'Observation'], metric)
    if (Decimal(model[key]) != Decimal(observation['valueMillions'])
            or model['baseYear'] != observation['fiscalYear'] or model['currency'] != observation['currency']):
        raise ValueError('Base financial value, fiscal year or currency differs from its source observation. Relink or unlink it before calculating or saving.')
    if metric == 'ebitda' and model['ebitdaBasis'] != observation['basis']:
        raise ValueError('EBITDA definition differs from its observation. Relink or unlink before calculating or saving.')
    return observation
