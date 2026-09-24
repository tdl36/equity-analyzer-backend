"""Conservative headline triage. Similarity is a review cue, never verified identity."""
import re
from datetime import datetime, timezone


STOPWORDS = {
    'a','after','an','and','announces','announcement','as','at','by','company',
    'corporation','from','for','in','inc','its','new','of','on','reports','says',
    'the','to','update','with',
}


def clinical_context(title):
    text = title.lower()
    medical = bool(re.search(r'\b(clinical|patients?|placebo|endpoint|efficacy|therapy|treatment|migraine|oncology|cancer|vaccine)\b', text))
    medical = medical or bool(re.search(r'\b[a-z][a-z-]{2,}(?:mab|nib|gepant|glutide|cept|ciclib|meran|tide|vir|parib|sertib)\b',text))
    mining = bool(re.search(r'\b(drill(?:ing)?|uranium|boreholes?|mineral(?:ization|isation)?|ore|copper mountain)\b', text))
    return medical and not mining


def clinical_signature(signal):
    """Only named trial/asset + phase + outcome can propose a duplicate hold."""
    if signal.get('category') != 'clinical':
        return None
    title = signal.get('title', '')
    phase = re.search(r'\bphase\s+((?:3|iii|2|ii|1|i)(?:[ab])?(?:\s*[/\-]\s*(?:3|iii|2|ii|1|i)(?:[ab])?)?)\b', title, re.I)
    if not phase:
        return None
    raw_phase=re.sub(r'\s+','',phase[1].lower())
    phase='/'.join({'i':'1','ii':'2','iii':'3'}.get(p,p) for p in re.split(r'[/\-]',raw_phase))
    trials = re.findall(r'\b([A-Z][A-Z0-9-]{2,})\s+(?:study|trial)\b', title)
    assets = re.findall(r'\b[a-z][a-z-]{2,}(?:mab|nib|gepant|glutide|cept|ciclib|meran|tide|vir|parib|sertib)\b', title.lower())
    # Trial names and recognizable INN suffixes are deliberately bounded. Unknown
    # assets stay separate; do not fuzzy-merge generic positive-results headlines.
    anchors = set(assets)
    for trial in trials:
        if trial.lower() not in {'phase','clinical','pivotal'}:
            anchors.add('trial:' + trial.lower())
    if not anchors:
        return None
    negative = bool(re.search(r'\b(failed|fails|missed|negative)\b', title, re.I))
    positive = bool(re.search(r'\b(positive|met|meets|meeting)\b', title, re.I))
    outcome = 'negative' if negative else 'positive' if positive else None
    if not outcome:
        return None
    day = datetime.fromtimestamp(signal['publishedAt'], timezone.utc).date().isoformat()
    return signal['ticker'], day, phase, outcome, anchors


def _tokens(title):
    return {word for word in re.findall(r'[a-z0-9]+', title.lower())
            if len(word) > 2 and word not in STOPWORDS}


def event_signature(signal):
    """A cautious non-clinical signature for differently worded syndication.

    Similarity only pauses a second automatic run for review; it never merges or
    accepts research. The publication-day boundary avoids collapsing follow-up
    developments that use similar language later.
    """
    category=signal.get('category')
    if not category or category=='clinical':
        return None
    title=signal.get('title','')
    tokens=_tokens(title)
    if len(tokens)<3:
        return None
    day=datetime.fromtimestamp(signal['publishedAt'],timezone.utc).date().isoformat()
    numbers=set(re.findall(r'\b\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?(?:bn|b|mn|m)?\b',title.lower()))
    return signal.get('ticker'),day,category,tokens,numbers


def possible_duplicate(signal, prior):
    current = clinical_signature(signal)
    if current:
        for row in prior:
            earlier = row.get('input') or {}
            other = clinical_signature(earlier)
            if other and current[:4] == other[:4] and current[4] & other[4]:
                return row
        return None
    current=event_signature(signal)
    if current:
        for row in prior:
            other=event_signature(row.get('input') or {})
            if not other or current[:3]!=other[:3]:
                continue
            shared=current[3] & other[3]
            union=current[3] | other[3]
            # Require substantial wording overlap, or matching numeric anchors
            # plus several shared event words. This is deliberately a review
            # cue rather than a destructive deduplication decision.
            if (len(shared)>=4 and len(shared)/len(union)>=0.45) or (current[4] and current[4] & other[4] and len(shared)>=3):
                return row
    return None
