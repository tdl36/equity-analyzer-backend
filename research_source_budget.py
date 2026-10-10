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
