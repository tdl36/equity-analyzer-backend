"""Allow only named missing investment sections; preserve existing containers."""
SUMMARY=('one_liner','core_question','earnings_direction')
DEBATE=('topic','market_view','evidence_for_market','alternative_view','evidence_for_alternative','resolution_signal')


def missing_fields(report):
    paths={}
    summary=report.get('summary')
    if summary is None:summary={}
    if isinstance(summary,dict):
        for key in SUMMARY:
            if summary.get(key) is None:paths['/summary/'+key]=None
        thesis=summary.get('investment_thesis')
        if thesis is None:thesis=[]
        if isinstance(thesis,list) and all(v is None or isinstance(v,str) for v in thesis):
            for i in range(3):
                if i>=len(thesis) or thesis[i] is None:paths[f'/summary/investment_thesis/{i}']=None
    debates=report.get('debates')
    if debates is None:debates=[]
    if isinstance(debates,list) and len(debates)<=20 and all(isinstance(row,dict) for row in debates):
        for i in range(2):
            row=debates[i] if i<len(debates) else {}
            for key in DEBATE:
                if row.get(key) is None:paths[f'/debates/{i}/{key}']=None
    return paths


def add_field(report,path,statement):
    if path not in missing_fields(report):
        raise ValueError('This missing field is not an allowed investment-section addition.')
    parts=path.split('/')[1:]
    if parts[0]=='summary':
        if report.get('summary') is None:report['summary']={}
        summary=report['summary']
        if len(parts)==2:summary[parts[1]]=statement
        else:
            if summary.get('investment_thesis') is None:summary['investment_thesis']=[]
            rows=summary['investment_thesis'];index=int(parts[2])
            while len(rows)<=index:rows.append(None)
            rows[index]=statement
    else:
        if report.get('debates') is None:report['debates']=[]
        rows=report['debates'];index=int(parts[1])
        while len(rows)<=index:rows.append({})
        rows[index][parts[2]]=statement
