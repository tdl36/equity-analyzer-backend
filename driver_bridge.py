"""Deterministic, single operating-margin sensitivity; not a full financial model."""
from decimal import Decimal,InvalidOperation
from datetime import date

def calculate(d):
    if not isinstance(d,dict):raise ValueError('Model inputs required.')
    def number(k,lo,hi):
        v=d.get(k)
        if isinstance(v,bool) or not isinstance(v,(str,int,float)) or len(str(v))>40:raise ValueError('Enter '+k)
        try:n=Decimal(str(v))
        except InvalidOperation:raise ValueError('Invalid '+k)
        if not n.is_finite() or not lo<=n<=hi:raise ValueError('Out of range: '+k)
        return n
    revenue=number('revenueMillions',Decimal('.000001'),Decimal('1000000000'))
    shares=number('sharesMillions',Decimal('.000001'),Decimal('1000000000'))
    before=number('beforeMarginPct',-100,100);after=number('afterMarginPct',-100,100)
    tax=number('taxPct',0,100);eps=number('baselineEPS',-1000000,1000000)
    pe=number('multiple',Decimal('.000001'),1000);price=number('referencePrice',Decimal('.000001'),10000000)
    for k in ('currency','period','basis'):
        if not isinstance(d.get(k),str) or not d[k].strip() or len(d[k])>100:raise ValueError('Enter '+k)
    if date.fromisoformat(d.get('asOf',''))>date.today():raise ValueError('Reference date cannot be in the future.')
    op=revenue*(after-before)/100;net=op*(1-tax/100);delta=net/shares;new=eps+delta
    def fmt(n):return str(n.quantize(Decimal('.0001')))
    return {'operatingProfitDeltaMillions':fmt(op),'netIncomeDeltaMillions':fmt(net),'epsDelta':fmt(delta),
        'beforeEPS':fmt(eps),'afterEPS':fmt(new),'beforeValue':fmt(eps*pe) if eps>0 else None,
        'afterValue':fmt(new*pe) if new>0 else None,
        'afterPriceReturnPct':fmt((new*pe/price-1)*100) if new>0 else None,
        'scope':'Single operating-margin change only. Revenue and diluted shares both in millions. All other inputs held constant; no dividends, discounting, debt or spreadsheet propagation. P/E valuation unavailable for nonpositive EPS.'}
