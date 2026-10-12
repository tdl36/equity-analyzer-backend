"""Read-only arithmetic diagnostics for a saved editorial selection.

No original-byte revalidation or investor decision is performed here. Only saved
supported claims and this edition's explicit proposals supply calculation inputs.
"""
import re
from stock_evidence import valid_citations
from scenario_arithmetic import check

_KEYS=('eps_or_fcf','valuation_multiple','implied_value','upside_downside')
_PATH=re.compile(r'^/scenarios/([0-2])/(name|eps_or_fcf|valuation_multiple|implied_value|upside_downside)$')


def scenario_review(run,record):
    claims=dict(valid_citations(run))
    for entry in record.get('claims',[]):
        proposal=entry.get('proposal',{})
        path=proposal.get('path')
        if isinstance(path,str):claims[path]=proposal
    grouped={}
    for path,claim in claims.items():
        match=_PATH.fullmatch(path)
        if match:grouped.setdefault(int(match[1]),{})[match[2]]=claim
    result=[]
    for index,row in sorted(grouped.items()):
        if not any(key in row for key in _KEYS):continue
        diagnosis=check(row)
        result.append({'index':index,'label':str(row.get('name',{}).get('statement') or f'Scenario {index+1}'),
                       'inputs':{key:row.get(key,{}).get('statement') for key in _KEYS},**diagnosis})
    return result
