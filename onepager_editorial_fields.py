"""Bounded scenario additions, isolated from the original saved report."""
import re
from stock_evidence import report_fields

SCENARIO_FIELDS=('name','business_conditions','revenue_growth','operating_margin',
                 'eps_or_fcf','valuation_multiple','implied_value','upside_downside','major_risks')
_PATTERN=re.compile(r'^/scenarios/([0-2])/('+'|'.join(SCENARIO_FIELDS)+r')$')


def editable_fields(report):
    fields={p:v for p,v in report_fields(report).items() if isinstance(v,str) and len(v)<=12000}
    from onepager_core_fields import missing_fields
    fields.update(missing_fields(report))
    from onepager_monitor_fields import missing_fields as missing_monitor_fields
    fields.update(missing_monitor_fields(report))
    cases=report.get('scenarios')
    if cases is None:cases=[]
    if isinstance(cases,list):
        if len(cases)>3 or any(not isinstance(row,dict) for row in cases):return fields
        rows={str(i):row for i,row in enumerate(cases)}
    elif isinstance(cases,dict) and all(k in ('0','1','2') and isinstance(v,dict) for k,v in cases.items()):rows=cases
    else:return fields
    for i in range(3):
        row=rows.get(str(i),{})
        for key in SCENARIO_FIELDS:
            if key not in row or row[key] is None:fields[f'/scenarios/{i}/{key}']=None
    return fields


def add_scenario_field(report,path,statement):
    match=_PATTERN.fullmatch(path)
    if not match or editable_fields(report).get(path,'INVALID') is not None:
        raise ValueError('This missing field is not an allowed scenario addition.')
    index,key=match.groups();cases=report.get('scenarios')
    if cases is None:cases=[];report['scenarios']=cases
    if isinstance(cases,list):
        while len(cases)<=int(index):cases.append({})
        cases[int(index)][key]=statement
    else:cases.setdefault(index,{})[key]=statement


def add_editorial_field(report,path,statement):
    if path.startswith('/scenarios/'):
        return add_scenario_field(report,path,statement)
    if path.startswith('/monitor/kpis/'):
        from onepager_monitor_fields import add_field
        return add_field(report,path,statement)
    from onepager_core_fields import add_field
    return add_field(report,path,statement)
