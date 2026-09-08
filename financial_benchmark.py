"""Deterministic scoring of explicitly annotated financial assertions.

This scores structured candidate annotations, not arbitrary prose entailment.
"""
from decimal import Decimal, InvalidOperation

SCALES={'USD':Decimal(1),'USD_million':Decimal(1000000),'USD_billion':Decimal(1000000000)}


def number(value):
    if isinstance(value,bool):raise ValueError('Boolean is not a financial value')
    try:result=Decimal(str(value))
    except (InvalidOperation,ValueError):raise ValueError('Invalid financial value')
    if not result.is_finite():raise ValueError('Financial values must be finite')
    if result.copy_abs()>Decimal('1e30') or (result and result.copy_abs()<Decimal('1e-30')):raise ValueError('Financial value exceeds benchmark numeric bounds')
    return result


def score(candidate,pack):
    specs=pack.get('financialExpectations',[])
    if not specs:return {'checked':0,'errors':[]}
    facts=pack['financialFacts'];assertions=candidate.get('financialAssertions')
    if not isinstance(assertions,list):return {'checked':len(specs),'errors':['Structured financial assertions are missing']}
    errors=[];known={s['id'] for s in specs}
    if any(not isinstance(a,dict) or a.get('id') not in known for a in assertions):errors.append('Unknown financial assertion')
    def fact(ident):return number(facts[ident]['value'])
    for spec in specs:
        label=spec['id'];rows=[a for a in assertions if isinstance(a,dict) and a.get('id')==label]
        if len(rows)!=1:
            errors.append(label+': missing or duplicate assertion');continue
        row=rows[0]
        try:
            operation=spec['operation']
            if operation=='reported':expected=fact(spec['source'])
            elif operation=='growth_percent':expected=(fact(spec['current'])/fact(spec['prior'])-1)*100
            elif operation=='margin_percent':expected=fact(spec['numerator'])/fact(spec['denominator'])*100
            elif operation=='margin_change_pp':
                expected=(fact(spec['currentNumerator'])/fact(spec['currentDenominator'])-fact(spec['priorNumerator'])/fact(spec['priorDenominator']))*100
            else:raise ValueError('Unknown financial operation')
            value=number(row.get('value'))
            if spec['unit']=='USD':
                if row.get('unit') not in SCALES:raise ValueError('Expected a dollar-denominated unit')
                value*=SCALES[row['unit']]
            elif row.get('unit')!=spec['unit']:raise ValueError('Percent and percentage points are different units')
            if row.get('period')!=spec['period']:raise ValueError('Fiscal comparison period does not match')
            if row.get('kind')!=spec['kind']:raise ValueError('Reported facts, calculations and guidance must remain distinct')
            if abs(value-expected)>number(spec.get('tolerance','0.02')):raise ValueError('Value fails the annotated arithmetic check')
        except (ValueError,KeyError,InvalidOperation,ZeroDivisionError) as exc:errors.append(label+': '+str(exc))
    return {'checked':len(specs),'errors':errors}
