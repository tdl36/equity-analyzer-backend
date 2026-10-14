"""Exact-source preparation for the pending full Lab note adapter.

No database writes, provider calls or implicit substitution of selected files.
"""
import hashlib
import json
from pipeline_lab import check_originals
from pipeline_lab_notes import normalized


def prepare(ticker, source_pack, documents, planner=None, allow_pdf_text=False):
    pack = normalized({'title':'Source pack','markdown':'Sources','qualityIssues':[],
                       'sourcePack':source_pack})['sourcePack']
    if type(allow_pdf_text) is not bool:
        raise ValueError('Explicit PDF representation policy required.')
    references = [s['reference'] for s in pack]
    checked = check_originals(ticker,'note',0,references,documents)
    if checked['blockers']:
        raise ValueError('Frozen originals are unavailable or ineligible. Recheck the selected sources.')
    if sorted(checked['verified'], key=lambda s:s['reference']) != pack:
        raise ValueError('Original bytes or names differ from the frozen source pack.')
    selected = [dict(documents[s['reference']], reference=s['reference']) for s in pack]
    if planner is None:
        from notegen import plan_batches
        planner = plan_batches
    batches = planner(selected)
    if not isinstance(batches,list) or not 1 <= len(batches) <= 8:
        raise ValueError('The selected source pack could not be planned completely.')
    seen = []
    output = []
    warnings = []
    for batch in batches:
        if not batch:
            raise ValueError('The source plan contains an empty batch.')
        content = []
        for doc in batch:
            ref = doc.get('reference')
            if ref not in references or ref in seen:
                raise ValueError('The source plan substituted or repeated a selected original.')
            seen.append(ref)
            original = documents[ref]
            # The planner may add extracted text, but must not replace originals.
            if doc.get('filename') != original['filename'] or doc.get('file_data') != original['file_data']:
                raise ValueError('The source plan changed a selected original.')
            mode = doc.get('send_as')
            is_pdf = original['filename'].lower().endswith('.pdf') or 'pdf' in str(original.get('file_type','')).lower()
            if mode == 'pages' and is_pdf:
                content.append({'type':'document','source':{'type':'base64','media_type':'application/pdf',
                                                           'data':original['file_data']}})
                content.append({'type':'text','text':'[Original: '+ref+'; '+original['filename']+']'})
            elif mode in ('text','file'):
                if is_pdf and not allow_pdf_text:
                    raise ValueError('A PDF requires text-only conversion. Review that representation before generation.')
                text = doc.get('extracted_text')
                if not isinstance(text,str) or not text.strip() or len(text.encode('utf-8')) > 2_000_000:
                    raise ValueError('A selected original has no bounded readable text representation.')
                content.append({'type':'text','text':'[Original: '+ref+'; '+original['filename']+' — extracted text]\n'+text})
                warnings.append({'reference':ref,'issue':'Extracted text representation; embedded images and layout are not verified.'})
            else:
                raise ValueError('A selected original cannot be represented; no source may be silently skipped.')
        output.append(content)
    if set(seen) != set(references):
        raise ValueError('The source plan omitted a selected original; generation is blocked.')
    # Bind the actual ordered representations, including any extraction, rather
    # than only the original hashes. A changed extraction cannot reuse a call.
    digest = hashlib.sha256(json.dumps(output,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    return {'sourcePack':pack,'batches':output,'representationSha256':digest,'warnings':warnings}
