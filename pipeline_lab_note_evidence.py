"""Bounded generation evidence retained with Lab draft revisions.

These are model-produced claims, not verified source facts. No rendering or calls.
"""
import math


def normalize(value):
    if not isinstance(value, dict) or set(value) != {'sourcesMarkdown', 'chartSpecs', 'stopReason'}:
        raise ValueError('Sources appendix, chart specifications and stop reason are required.')
    sources = value['sourcesMarkdown']
    if not isinstance(sources, str) or len(sources.encode('utf-8')) > 500_000:
        raise ValueError('Sources appendix must be bounded text.')
    reason = value['stopReason']
    if not isinstance(reason, str) or len(reason) > 100:
        raise ValueError('A bounded generation stop reason is required.')
    charts = value['chartSpecs']
    if not isinstance(charts, list) or len(charts) > 4:
        raise ValueError('At most four segment chart specifications are supported.')
    result = []
    identities = set()
    for chart in charts:
        if not isinstance(chart, dict) or set(chart) != {'kind', 'period', 'data'}:
            raise ValueError('Invalid segment chart specification.')
        kind, period, rows = chart['kind'], chart['period'], chart['data']
        if kind not in ('revenue', 'profit') or not isinstance(period, str) or not 1 <= len(period) <= 80:
            raise ValueError('Chart kind and fiscal period are required.')
        if (kind, period) in identities:
            raise ValueError('Duplicate segment chart.')
        identities.add((kind, period))
        if not isinstance(rows, list) or len(rows) > 50:
            raise ValueError('Segment chart rows exceed their bound.')
        names = set()
        clean = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'segment', 'value'}:
                raise ValueError('Invalid segment row.')
            name, number = row['segment'], row['value']
            if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120
                    or name.strip() in names or type(number) not in (int, float)
                    or abs(number) > 1e15 or not math.isfinite(number)):
                raise ValueError('Invalid segment name or finite numeric value.')
            names.add(name.strip())
            clean.append({'segment': name.strip(), 'value': number})
        result.append({'kind': kind, 'period': period, 'data': clean})
    return {'sourcesMarkdown': sources, 'chartSpecs': result, 'stopReason': reason}


def issues(value):
    """Structural warnings only; never certify factual quality from shape checks."""
    result = ['Generated note, sources appendix and chart claims require investor verification.']
    if not value['sourcesMarkdown'].strip():
        result.append('The generated sources appendix is missing.')
    if value['stopReason'] in ('max_tokens', 'length', 'MAX_TOKENS'):
        result.append('Generation reached its output limit; the draft may be incomplete.')
    if not value['chartSpecs']:
        result.append('No segment chart specifications were supplied.')
    return result
