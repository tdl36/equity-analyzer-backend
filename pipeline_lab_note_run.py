"""Pending full-note coordinator using injected storage and provider adapters.

No production route/worker invokes this yet. Never call the legacy generator.
"""
import hashlib
from pipeline_lab_note_plan import load, verify_preparation
from pipeline_lab_note_sources import prepare
from pipeline_lab_note_calls import run_once
from pipeline_lab_note_response import store_draft
from pipeline_lab_notes import read


def execute(get_db, note_id, load_documents, invoke, planner=None):
    """Resume from immutable replies; unknown paid outcomes stop automatically.

    load_documents(plan) returns the exact-ID original mapping. invoke accepts
    messages/model_key/max_tokens and returns the journal's normalized reply.
    The production adapter must enforce account/model policy and input budget.
    """
    with get_db() as (_,cur):
        plan=load(cur,note_id)
        if plan is None:raise ValueError('A frozen note plan is required.')
        existing=read(cur,note_id)
        if existing:
            return {'status':'draft','noteId':note_id,'revision':existing['revision']}
        cur.execute('SELECT status FROM pipeline_lab_note_calls WHERE note_id=%s AND batch=%s',(note_id,plan['batchCount']))
        final=cur.fetchone()
    def materialize():
        try:
            with get_db(commit=True) as (_,cur):
                draft=store_draft(cur,note_id,plan['batchCount'],plan['ticker'],plan['title'],plan['warnings'])
            return {'status':'draft','noteId':draft['noteId'],'revision':draft['revision']}
        except ValueError:
            return {'status':'attention','reason':'saved_reply_needs_review'}
    if final and final['status']=='complete':
        return materialize()
    previous=''
    for index in range(plan['batchCount']):
        # Recheck availability/eligibility and exact ordered representation before
        # each new dispatch. No filename-only fallback or live thesis reload.
        prepared=prepare(plan['ticker'],plan['sourcePack'],load_documents(plan),planner,plan['allowPdfText'])
        verify_preparation(plan,prepared)
        prompt=plan['promptTemplate']+'\n\nFROZEN CONTEXT:\n'+plan['context']
        if previous:
            prompt+='\n\nPREVIOUS COMPLETE NOTE RESPONSE — extend with this batch, preserving supported claims and attribution:\n'+previous
        if len(prompt.encode('utf-8'))>600_000:
            return {'status':'attention','reason':'merge_context_exceeds_bound'}
        messages=[{'role':'user','content':prepared['batches'][index]+[{'type':'text','text':prompt}]}]
        identity={'noteId':note_id,'batch':index+1,'sourcePack':plan['sourcePack'],
                  'contextSha256':prepared['representationSha256'],
                  'promptSha256':hashlib.sha256(prompt.encode()).hexdigest(),
                  'modelKey':plan['modelKey'],'maxTokens':plan['maxTokens']}
        with get_db() as (_,cur):
            cur.execute('SELECT status FROM pipeline_lab_note_calls WHERE note_id=%s AND batch=%s',(note_id,index+1))
            journal=cur.fetchone()
        if journal is None and hasattr(invoke,'preflight'):
            try:invoke.preflight(messages=messages,model_key=plan['modelKey'],max_tokens=plan['maxTokens'])
            except Exception:return {'status':'attention','reason':'preflight_failed'}
        result=run_once(get_db,identity,lambda:invoke(messages=messages,model_key=plan['modelKey'],max_tokens=plan['maxTokens']))
        if result['status']!='complete':return result
        previous=result['response']['text']
        # Do not compound an incomplete intermediate response with more paid work.
        if index+1<plan['batchCount'] and result['response']['stopReason'] in ('max_tokens','length','MAX_TOKENS'):
            return {'status':'attention','reason':'intermediate_reply_truncated'}
    return materialize()
