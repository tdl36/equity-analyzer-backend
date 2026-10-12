"""Bounded missing monitoring inputs; never infer deadlines or thresholds."""
FIELDS=('metric','current','bull_threshold','bear_threshold','timing','why_it_matters')


def missing_fields(report):
    monitor=report.get('monitor')
    if monitor is None:monitor={}
    if not isinstance(monitor,dict):return {}
    rows=monitor.get('kpis')
    if rows is None:rows=[]
    if not isinstance(rows,list) or len(rows)>20 or any(not isinstance(row,dict) for row in rows):return {}
    return {f'/monitor/kpis/{i}/{key}':None for i in range(3) for key in FIELDS
            if (rows[i] if i<len(rows) else {}).get(key) is None}


def add_field(report,path,statement):
    if path not in missing_fields(report):
        raise ValueError('This missing field is not an allowed monitoring addition.')
    if report.get('monitor') is None:report['monitor']={}
    if report['monitor'].get('kpis') is None:report['monitor']['kpis']=[]
    rows=report['monitor']['kpis'];_,_,_,index,key=path.split('/')
    while len(rows)<=int(index):rows.append({})
    rows[int(index)][key]=statement
