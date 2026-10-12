"""Dated public financial context; never modify frozen observations or research."""
from datetime import date
from decimal import Decimal, InvalidOperation

ABT_2024_SOURCE = 'https://abbott.mediaroom.com/2025-01-22-Abbott-Reports-Fourth-Quarter-and-Full-Year-2024-Results-Issues-2025-Financial-Outlook'
ABT_2024_HASH = '1b9f0f60323ba6095222a988a2cb1a99d56e433e96995766f2b495459378923a'


def context(snapshot):
    """Apply only to matching issuer, annual period, units and saved GAAP values.

    These are reported issuer adjustments, not analyst-normalized earnings. No
    assumption is made about whether excluded expenses recur or are economic.
    """
    if snapshot.get('ticker') != 'ABT' or str(snapshot.get('cik')) != '1800':
        return None
    try:
        if date.fromisoformat(snapshot.get('asOf', '')) < date(2025, 1, 22):
            return None
    except (TypeError, ValueError):
        return None
    rows = [r for r in snapshot.get('annuals', []) if r.get('start') == '2024-01-01' and r.get('end') == '2024-12-31']
    if len(rows) != 1:
        return None
    row = rows[0]
    for key, expected, unit in [('eps', '7.64', 'USD/shares'), ('netIncome', '13402000000', 'USD')]:
        item = row.get(key) or {}
        try:
            if item.get('unit') != unit or isinstance(item.get('value'), bool) or Decimal(str(item.get('value'))) != Decimal(expected):
                return None
        except (InvalidOperation, ValueError):
            return None
    return {
        'sourceUrl': ABT_2024_SOURCE, 'sourceHash': ABT_2024_HASH,
        'published': '2025-01-22', 'verified': '2026-10-12',
        'locator': 'Non-GAAP Reconciliation of Financial Information — Twelve Months Ended December 31, 2024; annual earnings footnotes 1–2',
        'summary': '2024 diluted EPS: GAAP $7.64 − $2.97 net specified benefits = issuer-adjusted $4.67. The year included a $7.497bn non-cash tax valuation-allowance benefit; the $2.97/share bridge also includes other specified items.',
        'detail': '2024 net earnings: GAAP $13.402bn − $5.202bn net specified benefits = issuer-adjusted $8.200bn. Abbott’s annual bridge adds $2.633bn to pretax earnings and $7.835bn to tax expense, a net −$5.202bn adjustment. Specified items also include amortization, impairments, restructuring, acquisitions/divestiture and other expenses.',
        'caveat': 'Issuer-defined non-GAAP measure, not normalized earnings. GAAP chart unchanged; remaining years are not reconciled here.',
    }
