"""Conservative headline triage. Similarity is a review cue, never verified identity."""
import re
from datetime import datetime, timezone


def clinical_context(title):
    text = title.lower()
    medical = bool(re.search(r'\b(clinical|patients?|placebo|endpoint|efficacy|therapy|treatment|migraine|oncology|cancer|vaccine)\b', text))
    mining = bool(re.search(r'\b(drill(?:ing)?|uranium|boreholes?|mineral(?:ization|isation)?|ore|copper mountain)\b', text))
    return medical and not mining


def clinical_signature(signal):
    """Only named trial/asset + phase + outcome can propose a duplicate hold."""
    if signal.get('category') != 'clinical':
        return None
    title = signal.get('title', '')
    phase = re.search(r'\bphase\s+(3|iii|2|ii|1|i)\b', title, re.I)
    if not phase:
        return None
    phase = {'i': '1', 'ii': '2', 'iii': '3'}.get(phase[1].lower(), phase[1])
    trial = re.search(r'\b(?:phase\s+(?:[123]|I{1,3})\s+)([A-Z][A-Z0-9-]{2,})\s+(?:study|trial)\b', title)
    assets = re.findall(r'\b[a-z][a-z-]{2,}(?:mab|nib|gepant|glutide|cept|ciclib)\b', title.lower())
    # Trial names and recognizable INN suffixes are deliberately bounded. Unknown
    # assets stay separate; do not fuzzy-merge generic positive-results headlines.
    anchors = set(assets)
    if trial:
        anchors.add('trial:' + trial[1].lower())
    if not anchors:
        return None
    negative = bool(re.search(r'\b(failed|fails|missed|negative)\b', title, re.I))
    positive = bool(re.search(r'\b(positive|met|meets|meeting)\b', title, re.I))
    outcome = 'negative' if negative else 'positive' if positive else None
    if not outcome:
        return None
    day = datetime.fromtimestamp(signal['publishedAt'], timezone.utc).date().isoformat()
    return signal['ticker'], day, phase, outcome, anchors


def possible_duplicate(signal, prior):
    current = clinical_signature(signal)
    if not current:
        return None
    for row in prior:
        earlier = row.get('input') or {}
        other = clinical_signature(earlier)
        if other and current[:4] == other[:4] and current[4] & other[4]:
            return row
    return None
