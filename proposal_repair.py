"""Repair unsupported drafts without changing other proposals or saved research."""
import copy
import json


def repair_prompt(changes):
    return ('\nREPAIR TASK — this supersedes the broad comparison above. Rewrite ONLY the following rejected paths. '
            'Do not generate condition_assessments. Address the review findings using the original documents, not the rejected quotations. '
            'Replace verbose appended history with concise current wording: normally 2–4 short sentences, target 60–100 words per field. '
            'Preserve material qualifications; do not silently remove a still-relevant issue just to meet a word target. '
            'Do not copy unsupported legacy claims into the replacement. Every factual assertion in the complete replacement must be supported by the cited contiguous passage. '
            'Use an EXACT quotation from the original, not a paraphrase of a broker report. Attribute broker descriptions of management comments to the broker. '
            'Clearly label interpretation. A single supported update is better than a comprehensive unsupported narrative. '
            'Return changes:[] if no material, supportable replacement exists; never manufacture support. '
            'Keep reason to 1–2 sentences on the material delta and investment implication.\nREJECTED DRAFTS:\n'+json.dumps(changes))


def merge_repair(previous, revised, targets):
    """Keep stable IDs and untouched siblings; an empty repair remains unacceptably flagged."""
    allowed={c['path']:c for c in targets}
    if any(c['path'] not in allowed for c in revised):
        raise ValueError('Repair attempted to change a field outside the rejected drafts.')
    replacements={c['path']:c for c in revised}
    merged=[]
    for original in previous:
        c=copy.deepcopy(original)
        if c['path'] in allowed:
            if c['path'] in replacements:
                c=copy.deepcopy(replacements[c['path']]);c['id']=original['id']
                c['repairOutcome']='revised' if c.get('passageMatched') and c.get('reviewPassed') else 'still_unsupported'
            else:
                c['repairOutcome']='no_supported_change'
        merged.append(c)
    return merged
