"""Conservative source-linked arithmetic. Never converts an estimate into fact."""
from decimal import Decimal, InvalidOperation
import re

UNITS={
    'usd':('usd',Decimal(1),'USD'), 'usd_m':('usd',Decimal(1000000),'USD'),
    'usd_bn':('usd',Decimal(1000000000),'USD'), 'usd_per_share':('usd_per_share',Decimal(1),'USD/share'),
    'percent':('percent',Decimal(1),'percentage points'), 'bps':('percent',Decimal('.01'),'percentage points'),
    'count':('count',Decimal(1),'units'), 'millions':('count',Decimal(1000000),'units'),
    'multiple':('multiple',Decimal(1),'x')}
UNIT_WORDS={'usd':r'\$|USD|dollars','usd_m':r'\$|USD|dollars','usd_bn':r'\$|USD|dollars',
    'usd_per_share':r'per.share|EPS|earnings per share','percent':r'%|percent','bps':r'bps|basis points',
    'count':r'.','millions':r'million|\bmn\b','multiple':r'\bx\b|multiple'}
MAGNITUDE={'usd_m':r'million|\bmn\b|\d\s*m\b','usd_bn':r'billion|\bbn\b|\d\s*b\b'}


def number(v):
    if isinstance(v,bool) or not isinstance(v,(str,int,float)):raise ValueError('Missing or invalid numeric value')
    try:x=Decimal(str(v).replace(',',''))
    except InvalidOperation:raise ValueError('Invalid numeric value')
    if not x.is_finite() or abs(x)>Decimal('1e18'):raise ValueError('Numeric value is not finite or exceeds supported range')
    return x


def normal(v):return ' '.join(str(v or '').lower().split())

def quoted_number(value,quote):
    # Exact numeric tokens, not substring matches (54 must not match 540).
    tokens=re.findall(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?=$|[^\w.]|[mMbBxX](?:n)?\b|bps\b)',quote)
    return any(number(t)==value for t in tokens)


def reconcile(raw,sources):
    from recap_validation import validate_claims
    comparisons=raw if isinstance(raw,list) else []
    result=[]
    for row in comparisons[:8]:
        if not isinstance(row,dict):continue
        issues=[];points={};metric=str(row.get('metric',''))[:200]
        for name in ('actual','benchmark'):
            point=row.get(name)
            if not isinstance(point,dict):issues.append(f'{name}: value and source required');continue
            claim=validate_claims({'claims':[{'statement':metric,**point}]},sources)
            checked=claim[0] if claim else {}
            quote=str(point.get('quote',''))[:6000];unit=point.get('unit') if isinstance(point.get('unit'),str) else None;period=point.get('period') if isinstance(point.get('period'),str) else None;basis=point.get('basis') if isinstance(point.get('basis'),str) else None
            out={'value':str(point.get('value'))[:100] if isinstance(point.get('value'),(str,int,float)) else None,'unit':unit,'period':period,'basis':basis,'sourceId':checked.get('sourceId'),'page':checked.get('page'),'quote':quote,'filename':checked.get('filename')};points[name]=out
            if not checked.get('passageMatched'):issues.append(f'{name}: source passage did not match')
            try:
                value=number(point.get('value'))
                if not quoted_number(value,quote):issues.append(f'{name}: numeric token absent from quotation')
                if not isinstance(unit,str) or unit not in UNITS:raise ValueError('unsupported unit')
                if not re.search(UNIT_WORDS[unit],quote,re.I) or (unit in MAGNITUDE and not re.search(MAGNITUDE[unit],quote,re.I)):issues.append(f'{name}: unit not established by quotation')
                if not isinstance(period,str) or not period.strip() or normal(period) not in normal(quote):issues.append(f'{name}: fiscal period not explicit in quotation')
                if not isinstance(basis,str) or basis not in ('reported','organic','adjusted','gaap','non_gaap'):issues.append(f'{name}: measurement basis required')
                basis_patterns={'organic':r'\borganic\b','adjusted':r'\badjusted\b','gaap':r'(?<!non-)(?<!non )\bgaap\b','non_gaap':r'\bnon[- ]?gaap\b'}
                if basis in basis_patterns and not re.search(basis_patterns[basis],quote,re.I):issues.append(f'{name}: measurement basis not explicit in quotation')
                if unit=='usd_per_share' and not re.search(r'\$|USD|dollars',quote,re.I):issues.append(f'{name}: currency is not established')
                points[name]['normalized']=value*UNITS[unit][1]
            except ValueError as e:issues.append(f'{name}: {e}')
        kind=row.get('benchmarkType') if isinstance(row.get('benchmarkType'),str) else None
        if kind not in ('prior_period','guidance','broker_estimate','consensus'):issues.append('Benchmark must identify prior period, guidance, broker estimate or consensus')
        if kind=='consensus':
            quote=normal(points.get('benchmark',{}).get('quote'))
            if 'consensus' not in quote or re.search(r'not.{0,30}consensus|single.{0,30}estimate|individual.{0,30}estimate|our estimate',quote):issues.append('Consensus aggregation is not established by the quotation')
        a=points.get('actual',{});b=points.get('benchmark',{})
        if a and b:
            if a.get('basis')!=b.get('basis'):issues.append('Measurement bases differ')
            if kind!='prior_period' and normal(a.get('period'))!=normal(b.get('period')):issues.append('Fiscal periods differ for an expectations comparison')
            if a.get('unit') in UNITS and b.get('unit') in UNITS and UNITS[a['unit']][0]!=UNITS[b['unit']][0]:issues.append('Units measure different quantities')
        delta=None;relative=None;bps=None;unit_label=None
        if not issues and 'normalized' in a and 'normalized' in b:
            delta=a['normalized']-b['normalized'];unit_label=UNITS[a['unit']][2]
            if b['normalized']!=0:relative=delta/abs(b['normalized'])*100
            if UNITS[a['unit']][0]=='percent':bps=delta*100
        for p in points.values():p.pop('normalized',None)
        result.append({'metric':metric,'benchmarkType':kind,'actual':a,'benchmark':b,
            'status':'needs_review' if issues else 'arithmetic_checked','issues':list(dict.fromkeys(issues)),
            'delta':str(delta) if delta is not None else None,'deltaUnit':unit_label,
            'relativePercent':str(round(relative,4)) if relative is not None else None,
            'basisPointDelta':str(bps) if bps is not None else None,
            'limitation':'Checks token presence, unit/period annotations and arithmetic. Source meaning and estimate classification still require analyst review.'})
    return result
