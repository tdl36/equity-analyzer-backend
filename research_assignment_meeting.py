"""Use the same completed report for meeting preparation without another paid run."""
from research_meeting_handoff import preview, prepare


def selection(run, options):
    pack=preview(run)
    indices=[q['index'] for q in pack['questions'] if q['sourceMatched']]
    if not indices:
        raise ValueError('No source-matched meeting questions were saved. Other outputs are retained; review the report and prepare questions explicitly. No additional research was charged.')
    body={'fingerprint':pack['fingerprint'],'indices':indices,
          'meetingDate':options['date'],'meetingType':options['type'],
          'expectedSources':[{k:s[k] for k in ('filename','originalHash')} for s in run['sources']]}
    prepare(run,body)
    return body
