"""Deterministic source excerpts within the existing research context ceiling."""
import copy
import hashlib

LIMIT = 160000
MARKER = '\n\n[Source excerpt gap: intervening text was not supplied for research.]\n\n'


def fit_sources(sources, limit=LIMIT):
    """Share a fixed budget fairly; retain exact front/back passages, never summaries.

    Short originals stay complete. Longer originals share the remaining space.
    Source identities/original hashes remain unchanged; passage matching uses only
    supplied text. Character ranges refer to the complete extraction, not PDF bytes.
    """
    if not sources or any(not s['text'].strip() for s in sources):
        raise ValueError('A selected original has no readable text.')
    result=copy.deepcopy(sources)
    if sum(len(s['text']) for s in result)<=limit:return result
    low,high=0,limit
    while low<high:
        mid=(low+high+1)//2
        if sum(min(len(s['text']),mid) for s in result)<=limit:low=mid
        else:high=mid-1
    if low<len(MARKER)+1000:raise ValueError('Insufficient research context for this source pack.')
    for s in result:
        text=s['text']
        if len(text)<=low:continue
        available=low-len(MARKER);head=available*3//4;tail=available-head
        s['coverage']={'mode':'bounded_excerpts','extractedCharacters':len(text),
            'includedRanges':[[0,head],[len(text)-tail,len(text)]],
            'omittedCharacters':len(text)-available,
            'fullExtractionHash':s['extractionHash'],
            'limitation':'Only exact opening and closing excerpts were supplied. Intervening text was omitted to keep the shared research context within 160,000 characters; this is not full-document review.'}
        s['text']=text[:head]+MARKER+text[-tail:]
        s['extractionHash']=hashlib.sha256(s['text'].encode()).hexdigest()
    return result


FULL_TEXT_BYTE_LIMIT = 480000


def complete_sources(sources):
    """New-policy packs preserve every returned character or stop before a paid call.

    This is a UTF-8 input budget, not a claim of PDF/visual completeness. Legacy
    frozen packs keep their exact representations; this never repairs them in place.
    """
    from research_extraction import has_omissions
    if not sources or any(not isinstance(s.get('text'), str) or not s['text'].strip() for s in sources):
        raise ValueError('A selected original has no readable text.')
    if has_omissions(sources):
        raise ValueError('This pack contains omitted source text. Prepare a new selection from the originals; no paid research has started.')
    if sum(len(s['text'].encode('utf-8')) for s in sources) > FULL_TEXT_BYTE_LIMIT:
        raise ValueError('Complete extracted source text exceeds the 480,000-byte research budget. Choose a smaller task-specific pack; originals are retained and no paid research has started. Sources will not be silently clipped.')
    return copy.deepcopy(sources)



def rewrite_sources(parent, documents, extract_text):
    """Re-extract the same verified originals for a newly priced rewrite only."""
    from research_originals import verified_original
    from research_extraction import frozen_source
    result=[]
    for old in parent['sources']:
        matches=[d for d in documents if d['filename']==old['filename']]
        if len(matches)!=1:
            raise ValueError('A frozen original is missing or ambiguous. No rewrite started.')
        doc=copy.deepcopy(matches[0])
        verified_original(old,doc,parent['input'].get('hashes',{}).get(old['filename']))
        fresh=frozen_source(doc,old['originalHash'],extract_text(doc),old['id'])
        if old.get('selectionContext') is not None:
            fresh['selectionContext']=copy.deepcopy(old['selectionContext'])
        result.append(fresh)
    return complete_sources(result)
