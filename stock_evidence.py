"""Shared read-time evidence identity checks for stock research derivatives.

These are consistency checks, not independent factual verification. Inputs stay
unchanged and historical exports are never rewritten by this module.
"""
import math
import re


def scalar_text(value):
    if value is None or isinstance(value,(dict,list)):return None
    if isinstance(value,bool):return 'true' if value else 'false'
    if isinstance(value,float):
        if not math.isfinite(value):return None
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def report_fields(value, path='', depth=0):
    if depth>12:return {}
    if isinstance(value,dict):
        out={}
        for key,item in value.items():
            key=str(key).replace('~','~0').replace('/','~1')
            out.update(report_fields(item,path+'/'+key,depth+1))
        return out
    if isinstance(value,list):
        out={}
        for index,item in enumerate(value):out.update(report_fields(item,path+'/'+str(index),depth+1))
        return out
    return {path:scalar_text(value)}


def missing_value(value):
    text=scalar_text(value)
    return text is None or not text.strip() or bool(re.match(r'^(unavailable|not available|not provided|not disclosed|not recorded|n/?a)(?:\b|[.:])',text,re.I))


def valid_citations(run):
    fields=report_fields(run.get('state',{}).get('report',{}))
    ids={source.get('id') for source in run.get('sources',[]) if source.get('id')}
    valid={}
    for path,c in run.get('state',{}).get('citations',{}).items():
        value=fields.get(path)
        if missing_value(value):continue
        if not isinstance(c,dict) or c.get('statement')!=value or c.get('review')!='supported' or c.get('passageMatched') is not True:continue
        if any(e.get('matched') is True and e.get('sourceId') in ids for e in c.get('evidence',[])):
            valid[path]=c
    return valid
