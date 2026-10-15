"""Server-owned note scope and exact-source context, independent of generated prose."""
import html
import re
from lab_source_identity import validate as identities
from research_source_context import validate as source_context
from research_purpose import validate as purpose


def validate(value, pack):
    if not isinstance(value, dict) or set(value) != {'version', 'purpose', 'sourceEvidence'} or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('A versioned note provenance record is required.')
    expected=identities([{'id':s['reference'][len('charlie:thesis_'):], 'filename':s['filename'], 'sha256':s['sha256']} for s in pack], [s['filename'] for s in pack])
    return {'version':1, 'purpose':purpose(value['purpose'], ['summary_note']),
            'sourceEvidence':source_context(value['sourceEvidence'], expected)}


def from_assignment(assignment, pack):
    if not assignment.get('researchPurpose'):
        return {}  # Legacy or standalone notes have no inferred purpose.
    return {'researchProvenance':validate({'version':1, 'purpose':assignment['researchPurpose'],
            'sourceEvidence':assignment.get('processingEvidenceContext')}, pack)}


def export_lines(value):
    def safe(text):
        return re.sub(r'([\\`*_{}\[\]()#+.!|>-])', r'\\\1', html.escape(str(text)).replace('\n',' ').replace('\r',' '))
    if not value:
        return ['\n## Saved research scope', 'This note has no structured frozen research-purpose or source-date record. Its scope is not inferred from its title.']
    p=value['purpose'];event=p.get('event')
    lines=['\n## Saved research scope']
    if event:
        lines += ['Event note / debrief: '+safe(event['title'])+' · '+safe(event['date']),
                  'Event-focused analysis with historical context. This note does not update the thesis or establish thesis freshness. Thesis implications require a separate proposal.']
    else:
        lines += ['Longitudinal thesis review', 'Discovery dates limit new-source collection, not analytical history. This note does not apply thesis changes.']
    lines += ['\n## Frozen source context', 'Publication dates are not financial periods. Thesis-register references do not prove incorporation or current validity. Missing references do not establish that evidence is new.']
    evidence=value.get('sourceEvidence')
    if evidence is None:
        return lines+['Source dates and roles were not recorded for this note.']
    for s in evidence['sources']:
        reference={True:'Referenced by saved thesis register',False:'No saved thesis-register reference',None:'Thesis-register reference unknown'}[s['referencedBySavedThesis']]
        lines.append('- '+safe(s['filename'])+' · '+('Collected evidence' if s['role']=='collection' else 'Supplemental history')+' · Publication: '+safe(s['publishedDate'] or 'Unknown')+' · Publisher: '+safe(s['publisher'] or 'Unknown')+' · Type: '+safe(s['sourceKind'] or 'Unknown')+' · '+reference)
    return lines
