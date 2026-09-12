"""Validated analyst annotations for transparent case diagnostics, not trade instructions."""
from datetime import date
from decimal import Decimal, InvalidOperation
import uuid


def decay_rules(raw):
    if not isinstance(raw,dict):raise ValueError('Enter evidence decay rules.')
    result={}
    for key in ('event','cyclical'):
        value=raw.get(key,30 if key=='event' else 180)
        if type(value) is not int or not 1<=value<=3650:raise ValueError('Half-life must be 1–3650 days.')
        result[key]=value
    result['structural']=None
    return result


def validate(raw, assumption_ids):
    if raw is None:return {}
    if not isinstance(raw,dict):raise ValueError('Invalid case diagnostics.')
    def text(v,label,limit=3000):
        if not isinstance(v,str) or len(v)>limit:raise ValueError('Invalid '+label)
        return v.strip()
    def enum(v,choices,label):
        if v not in choices:raise ValueError('Invalid '+label)
        return v
    def number(v,label):
        if v in ('',None):return ''
        if isinstance(v,bool):raise ValueError('Invalid '+label)
        try:n=Decimal(str(v))
        except InvalidOperation:raise ValueError('Invalid '+label)
        if not n.is_finite() or abs(n)>10**12:raise ValueError('Invalid '+label)
        return str(n)
    def day(v):
        if v in ('',None):return ''
        try:d=date.fromisoformat(v)
        except (ValueError,TypeError):raise ValueError('Enter a valid observation date.')
        if d>date.today():raise ValueError('Observation dates cannot be in the future.')
        return d.isoformat()
    def rows(key,limit):
        values=raw.get(key,[])
        if not isinstance(values,list) or len(values)>limit:raise ValueError('Too many '+key)
        seen=set();out=[]
        for v in values:
            if not isinstance(v,dict):raise ValueError('Invalid '+key)
            try:ident=str(uuid.UUID(v.get('id','')))
            except (ValueError,TypeError,AttributeError):raise ValueError('Invalid record ID.')
            if ident in seen:raise ValueError('Duplicate record ID.')
            seen.add(ident)
            aid=v.get('assumptionId')
            if aid not in assumption_ids:raise ValueError('A diagnostics record refers to a removed assumption. Remove or reassign it first.')
            out.append((v,{'id':ident,'assumptionId':aid}))
        return out
    result={'observations':[],'tests':[],'variants':[]}
    for v,r in rows('observations',200):
        for k in ('statement','source','targetLabel'):r[k]=text(v.get(k,''),k)
        r['targetType']=enum(v.get('targetType'),('pillar','signpost','estimate','risk'),'target type')
        r['direction']=enum(v.get('direction'),('support','challenge','neutral'),'evidence direction')
        r['kind']=enum(v.get('kind'),('event','cyclical','structural'),'evidence lifetime')
        r['status']=enum(v.get('status','active'),('active','retired'),'evidence status')
        r['asOf']=day(v.get('asOf'))
        if not r['statement'] or not r['source'] or not r['asOf']:raise ValueError('Each evidence assessment needs a statement, source reference and observation date.')
        result['observations'].append(r)
    for v,r in rows('tests',60):
        for k in ('question','metric','unit','period','source','needed'):r[k]=text(v.get(k,''),k)
        r['operator']=enum(v.get('operator','lte'),('lte','gte'),'threshold direction')
        r['current']=number(v.get('current'),'current observation');r['threshold']=number(v.get('threshold'),'threshold');r['asOf']=day(v.get('asOf'))
        if not r['question']:raise ValueError('State what would disprove the assumption.')
        if r['current']!='' and not all(r[k] for k in ('metric','unit','period','source','asOf')):raise ValueError('Numeric observations need metric, unit, period, source and date.')
        result['tests'].append(r)
    for v,r in rows('variants',60):
        for k in ('metric','unit','period','source','rationale'):r[k]=text(v.get(k,''),k)
        r['view']=number(v.get('view'),'my estimate');r['market']=number(v.get('market'),'market estimate');r['asOf']=day(v.get('asOf'))
        r['baselineType']=enum(v.get('baselineType','unknown'),('unknown','consensus','broker','implied'),'market baseline type')
        r['pricing']=enum(v.get('pricing','unknown'),('unknown','reflected','not_reflected'),'pricing assessment')
        if r['market']!='' and (not all(r[k] for k in ('metric','unit','period','source','asOf')) or r['baselineType']=='unknown'):raise ValueError('Market comparisons need a named dated baseline, type, metric, unit and period.')
        if r['pricing']!='unknown' and not r['rationale']:raise ValueError('Explain your assessment of what is priced in.')
        result['variants'].append(r)
    p=raw.get('position',{})
    if not isinstance(p,dict):raise ValueError('Invalid position snapshot.')
    result['position']={k:text(p.get(k,''),k) for k in ('portfolio','benchmark','rationale')}
    result['position'].update(asOf=day(p.get('asOf')),conviction=enum(p.get('conviction','unknown'),('unknown','low','medium','high'),'case conviction'))
    for k in ('weight','benchmarkWeight'):
        value=number(p.get(k),k)
        if value!='' and not (-100<=Decimal(value)<=100):raise ValueError('Weights must be between -100 and 100 percent.')
        if k=='benchmarkWeight' and value!='' and Decimal(value)<0:raise ValueError('Benchmark weight must be nonnegative.')
        result['position'][k]=value
    if result['position']['weight']!='' and not all(result['position'][k] for k in ('portfolio','asOf')):raise ValueError('Position weight needs portfolio name and as-of date.')
    if result['position']['benchmarkWeight']!='' and not result['position']['benchmark']:raise ValueError('Name the benchmark before comparing active weight.')
    return result
