"""Mechanical pack diagnostics, explicitly separate from investment judgment."""
import re
from collections import Counter


def inspect(topics,filenames):
    questions=[q for t in topics for q in t.get('questions',[])]
    norm=lambda s:re.sub(r'\W+',' ',str(s).lower()).strip()
    repeated=[q for q,n in Counter(norm(q.get('question')) for q in questions).items() if n>1]
    available=set(filenames);cited={f for q in questions for f in q.get('source_filenames',[])}
    missing=[i+1 for i,q in enumerate(questions) if not all(q.get(k) for k in ('question','context','source','follow_up_angle','priority'))]
    support=sum(bool(q.get('supporting_quotes')) and bool(q.get('source_support')) for q in questions)
    return {'questions':len(questions),'topics':len(topics),'highPriority':sum(q.get('priority')=='high' for q in questions),
        'withVerifiedPassages':support,'availableSources':len(available),'citedSources':len(cited & available),
        'uncitedSources':sorted(available-cited),'unknownCitations':sorted(cited-available),'duplicateQuestions':len(repeated),
        'incompleteQuestions':missing,'numericPremisesToReview':[{'question':i+1,'values':premise_flags(q)} for i,q in enumerate(questions) if premise_flags(q)],'scope':'Mechanical checks only. Review factual premises, source interpretation, topic coverage and decision relevance against originals. Uncited sources are not automatically omissions.'}


def premise_flags(question):
    """Flag unmatched numeric premises for human review, never infer falsity."""
    text=' '.join(q.get('quote','') for q in question.get('supporting_quotes',[]) if isinstance(q,dict))
    normalize=lambda s:re.sub(r'[\s,]','',s).lower()
    numbers=re.findall(r'(?<![\w])\$?\d+(?:,\d{3})*(?:\.\d+)?(?:%|bps)?',question.get('question',''))
    return sorted({n for n in numbers if normalize(n) not in normalize(text)}) if text else []
