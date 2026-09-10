"""Freeze shared context once per new meeting job; recovery reuses that exact view."""
import json
import company_memory
from research_amendments import obj


def freeze(get_db, job_id, ticker, owner=None):
    with get_db(commit=True) as (_, cur):
        cur.execute('SELECT ticker,input,status FROM mp_jobs WHERE id=%s FOR UPDATE', (job_id,))
        row = cur.fetchone()
        if not row or row['ticker'] != ticker:
            raise ValueError('Meeting context belongs to a different or missing assignment.')
        inp = obj(row['input'])
        if not inp.get('companyMemoryRequested'):
            return None  # Old jobs retain their original generation contract.
        if row['status'] != 'running' or (owner and inp.get('workerToken') != owner):
            raise ValueError('Meeting worker no longer owns this assignment.')
        if inp.get('companyMemory') is not None:
            return inp['companyMemory']
        try:
            snapshot = company_memory.load(get_db, ticker)
        except Exception as exc:
            raise ValueError('Company memory could not be loaded. Retry the meeting pack; no source analysis was started.') from exc
        inp['companyMemory'] = snapshot
        cur.execute('UPDATE mp_jobs SET input=%s::jsonb,updated_at=NOW() WHERE id=%s', (json.dumps(inp), job_id))
        return snapshot


def instruction(snapshot):
    if snapshot is None:return ''
    return (company_memory.render(snapshot) +
        '\nMEETING USE: Use this saved view to identify what to test, unresolved assumptions, and contrary evidence. '
        'It is not an additional source document and must never satisfy a source citation or passage requirement. '
        'Phrase unsupported beliefs as open questions, not factual premises. Do not infer that a planned question '
        'was asked, or that a thesis edit records an actual management answer. Keep the selected meeting length '
        'and breadth; do not let thesis-specific questions crowd out comprehensive business coverage.\n')
