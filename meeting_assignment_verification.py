"""Read-only acceptance checks for a real managed meeting assignment.

This verifies delivery and saved source references, not financial correctness.
"""
from meeting_commands import validate_pack


def verify(command, collection, recap, meeting, job):
    checks=[]
    def check(name, passed):
        checks.append({'check':name,'passed':bool(passed)})
    check('Collection completed',collection.get('status')=='complete')
    result=collection.get('result') or {}
    check('Originals imported',result.get('importedOriginals',0)>0)
    check('Recap belongs to assignment',bool(command.get('topic')) and (recap.get('detail') or {}).get('topic')==command.get('topic') and recap.get('ticker')==command.get('ticker'))
    check('Recap completed',recap.get('status')=='complete' and bool((recap.get('result') or {}).get('markdown')))
    evidence=(recap.get('result') or {}).get('evidenceSnapshot') or {}
    sources=evidence.get('sources') or []
    names=[s.get('filename') for s in sources]
    check('Recap received every imported original',bool(names) and len(names)==len(set(names))==result.get('importedOriginals'))
    check('Meeting job belongs to assignment',bool(command.get('job_id')) and job.get('id')==command.get('job_id') and job.get('ticker')==command.get('ticker'))
    check('Meeting generation completed',job.get('status')=='done' and command.get('prep_status')=='done')
    saved=meeting.get('meeting') or {};docs=meeting.get('documents') or [];qs=meeting.get('questionSet') or {}
    check('Meeting belongs to assignment',str(saved.get('id'))==str(command.get('meeting_id')) and saved.get('ticker')==command.get('ticker'))
    check('Meeting preserves recap source set',bool(names) and sorted(d.get('filename','') for d in docs)==sorted(names))
    check('Saved question set matches job receipt',bool(qs.get('id')) and qs.get('id')==(job.get('result') or {}).get('questionSetId'))
    valid=False
    try:
        validate_pack(qs.get('topics'),docs);valid=True
    except (ValueError,TypeError,KeyError):pass
    check('Every question has a verified source, rationale and follow-up',valid)
    return {'status':'passed' if all(c['passed'] for c in checks) else 'incomplete', 'checks':checks,
            'sourceCount':len(docs),'questionCount':sum(len(t.get('questions',[])) for t in qs.get('topics',[]) if isinstance(t,dict)),
            'limitations':['Delivery and citation identity checks only; analyst review of factual support and investment judgment remains required.']}
