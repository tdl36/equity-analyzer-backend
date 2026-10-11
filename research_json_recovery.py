"""One lossless envelope repair; never invent, truncate or rewrite research values."""
import json


def _pairs(items):
    out={}
    for key,value in items:
        if key in out:raise ValueError('Duplicate JSON key')
        out[key]=value
    return out


def recover_report_envelope(text):
    if not isinstance(text,str):raise ValueError('No saved response text')
    text=text.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text=text[8:-3].strip()
    elif text.startswith('```\n') and text.endswith('```'):
        text=text[4:-3].strip()
    decoder=json.JSONDecoder(object_pairs_hook=_pairs)
    first,end=decoder.raw_decode(text)
    if not isinstance(first,dict) or set(first)!={'report'} or not isinstance(first['report'],dict):
        raise ValueError('Not a report envelope')
    if text[end-1:end]!='}' or not text[end:].lstrip().startswith(',"citations"'):
        raise ValueError('Not a premature report closure')
    repaired=decoder.decode(text[:end-1]+text[end:])
    if set(repaired)!={'report','citations'} or repaired['report']!=first['report'] or not isinstance(repaired['citations'],list):
        raise ValueError('Unsupported response envelope')
    return repaired
