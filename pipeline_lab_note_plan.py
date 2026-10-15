"""Immutable full-note run plans; pending production integration.

Store bounded context and representation identities, never credentials or a second
copy of source bytes. Re-preparation must reproduce the recorded representation.
"""
import hashlib
import json
import re
from pipeline_lab_notes import _note_identity, normalized, obj
from pipeline_lab_note_calls import request_value


def schema(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS pipeline_lab_note_plans (
        note_id UUID PRIMARY KEY, plan JSONB NOT NULL, plan_hash CHAR(64) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')


def validate(value):
    keys={'noteId','ticker','title','context','promptTemplate','modelKey','maxTokens',
          'sourcePack','representationSha256','batchCount','warnings','allowPdfText'}
    if not isinstance(value,dict) or not keys <= set(value) or set(value)-keys-{'researchProvenance'}:
        raise ValueError('A complete frozen note plan is required.')
    out=dict(value);out['noteId']=_note_identity(value['noteId'])
    if not isinstance(value['ticker'],str) or not re.fullmatch('[A-Z0-9][A-Z0-9.-]{0,19}',value['ticker']):
        raise ValueError('Confirmed ticker required.')
    for field,bound in (('title',240),('context',500_000),('promptTemplate',100_000)):
        if not isinstance(value[field],str) or len(value[field].encode('utf-8'))>bound:
            raise ValueError('Frozen note context exceeds its bound.')
    if not value['title'].strip() or not value['promptTemplate'].strip():
        raise ValueError('A title and full-note prompt are required.')
    if type(value['allowPdfText']) is not bool:
        raise ValueError('Explicit PDF representation policy required.')
    out['sourcePack']=normalized({'title':value['title'],'markdown':'Manifest',
                                  'sourcePack':value['sourcePack'],'qualityIssues':[]})['sourcePack']
    if 'researchProvenance' in value:
        from pipeline_lab_note_provenance import validate as provenance
        out['researchProvenance']=provenance(value['researchProvenance'],out['sourcePack'])
    # Reuse the journal's model, hash, batch and output-cap validation.
    request_value({'noteId':out['noteId'],'batch':value['batchCount'],
                   'sourcePack':out['sourcePack'],'contextSha256':value['representationSha256'],
                   'promptSha256':'0'*64,'modelKey':value['modelKey'],'maxTokens':value['maxTokens']})
    warnings=value['warnings'];refs={s['reference'] for s in out['sourcePack']}
    if not isinstance(warnings,list) or len(warnings)>1000:
        raise ValueError('Bounded representation warnings required.')
    seen=set()
    for warning in warnings:
        if (not isinstance(warning,dict) or set(warning)!={'reference','issue'}
                or not isinstance(warning['reference'],str) or warning['reference'] not in refs
                or warning['reference'] in seen or not isinstance(warning['issue'],str)
                or not 1<=len(warning['issue'])<=1000):
            raise ValueError('Invalid source representation warning.')
        seen.add(warning['reference'])
    return out


def freeze(cur, value):
    plan=validate(value)
    encoded=json.dumps(plan,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('lab-note-plan:'+plan['noteId'],))
    cur.execute('SELECT plan_hash FROM pipeline_lab_note_plans WHERE note_id=%s',(plan['noteId'],))
    old=cur.fetchone()
    if old:
        if old['plan_hash']!=digest:
            raise ValueError('This note already has a different frozen plan.')
        return {'noteId':plan['noteId'],'planHash':digest,'replayed':True}
    cur.execute('INSERT INTO pipeline_lab_note_plans(note_id,plan,plan_hash) VALUES(%s,%s::jsonb,%s)',(plan['noteId'],encoded,digest))
    return {'noteId':plan['noteId'],'planHash':digest,'replayed':False}


def load(cur, note_id):
    cur.execute('SELECT plan FROM pipeline_lab_note_plans WHERE note_id=%s',(_note_identity(note_id),))
    row=cur.fetchone()
    return obj(row['plan']) if row else None


def verify_preparation(plan, prepared):
    """Fail before dispatch if extraction, batching or warnings changed."""
    if (plan['sourcePack']!=prepared['sourcePack']
            or plan['representationSha256']!=prepared['representationSha256']
            or plan['batchCount']!=len(prepared['batches'])
            or plan['warnings']!=prepared['warnings']):
        raise ValueError('Source representation changed since the plan was frozen. Review a new plan.')
