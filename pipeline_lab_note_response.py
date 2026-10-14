"""Parse a saved full-note reply into a reviewable Lab draft, without model calls.

Malformed replies stay in the call journal for recovery. Never relabel arbitrary
provider text or an assignment summary as a successfully generated full note.
"""
import json
from pipeline_lab_notes import normalized
from pipeline_lab_note_calls import response_value


def parse(response, title, source_pack):
    reply = response_value(response)
    text = reply['text']
    spans = []

    def block(name, start, end, required=False):
        starts, ends = text.count(start), text.count(end)
        if starts == ends == 0 and not required:
            return ''
        if starts != 1 or ends != 1:
            raise ValueError('Missing or ambiguous ' + name + ' boundaries in the saved reply.')
        a, b = text.index(start), text.index(end)
        if b <= a:
            raise ValueError('Invalid ' + name + ' boundary order in the saved reply.')
        spans.append((a, b + len(end)))
        return text[a + len(start):b].strip()

    note = block('note', '===NOTE_START===', '===NOTE_END===', required=True)
    sources = block('sources', '===SOURCES_START===', '===SOURCES_END===')
    raw_charts = block('chart', '===CHART_DATA===', '===CHART_DATA_END===')
    ordered = sorted(spans)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Overlapping reply sections cannot be interpreted safely.')
    specs = []
    if raw_charts:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('Duplicate chart JSON key.')
                value[key] = item
            return value
        try:
            charts = json.loads(raw_charts, object_pairs_hook=unique)
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError('The saved chart block is not unambiguous JSON.')
        if not isinstance(charts, dict) or set(charts) != {'priorYear', 'currentYear'}:
            raise ValueError('Both fiscal-year chart groups are required.')
        for key in ('priorYear', 'currentYear'):
            part = charts[key]
            if not isinstance(part, dict) or set(part) != {'label','revenue','profit'}:
                raise ValueError('Fiscal-year labels and segment series are required.')
            for kind in ('revenue','profit'):
                specs.append({'kind':kind,'period':part['label'],'data':part[kind]})
    payload = {'title':title, 'markdown':note, 'sourcePack':source_pack, 'qualityIssues':[],
               'generationEvidence':{'sourcesMarkdown':sources,'chartSpecs':specs,
                                     'stopReason':reply['stopReason']}}
    checked = normalized(payload)
    # Carry structural warnings into the immutable draft. Fact-checking remains
    # outstanding even when the response is well formed.
    return {key:checked[key] for key in payload}


def store_draft(cur, note_id, batch, ticker, title):
    """Materialize a completed final reply in the caller's transaction.

    The future coordinator must supply its frozen final batch/ticker/title. No
    provider call is made here, including when parsing or draft saving fails.
    """
    import uuid
    from pipeline_lab_notes import _note_identity, append, obj
    note_id = _note_identity(note_id)
    if type(batch) is not int or not 1 <= batch <= 8:
        raise ValueError('A valid completed batch is required.')
    cur.execute('SELECT status,request,response FROM pipeline_lab_note_calls WHERE note_id=%s AND batch=%s', (note_id,batch))
    row = cur.fetchone()
    if not row or row['status'] != 'complete':
        raise ValueError('A completed provider reply is required before saving a draft.')
    payload = parse(obj(row['response']),title,obj(row['request'])['sourcePack'])
    event = str(uuid.uuid5(uuid.UUID(note_id),'generated-draft'))
    return append(cur,note_id,ticker,event,0,payload)
