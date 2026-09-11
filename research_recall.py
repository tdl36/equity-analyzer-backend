"""Bounded literal history retrieval, not semantic evidence adjudication."""
import re

STOP = set('the and that this with from have what when which will would could should into your their about been were then than they them does also case thesis growth company market'.split())


def terms_for(case, focus=''):
    body=(case or {}).get('body',{})
    from research_amendments import obj
    body=obj(body)
    texts=[focus,body.get('thesis',''),body.get('variantView',''),body.get('changeConditions','')]
    texts += [a.get('claim','') for a in body.get('assumptions',[]) if isinstance(a,dict)]
    words=re.findall(r'[a-z][a-z0-9-]{3,}', ' '.join(str(t) for t in texts).lower())
    unique=list(dict.fromkeys(w for w in words if w not in STOP))
    return unique[:24],len(unique)>24


def retrieve(cur,ticker,case,recent,as_of,focus=''):
    terms,limited=terms_for(case,focus)
    excluded=[str(r['id']) for r in recent]
    # Score whole tokens. SQL scans all ticker history; only returned records are
    # bounded. No model calls, hidden source retrieval, or automatic dispositions.
    cur.execute('''SELECT d.id,d.revision,d.body,d.created_at,
        EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=d.ticker
            AND n.body->>'supersedes'=d.id) AS superseded,
        (SELECT count(*) FROM unnest(%s::text[]) term WHERE term = ANY(
            regexp_split_to_array(lower(concat_ws(' ',d.body->>'decision',
            d.body->>'rationale',d.body->>'revisitWhen',d.body->>'issue')), '[^a-z0-9-]+'))) AS relevance
        FROM research_decisions d WHERE d.ticker=%s AND NOT(d.id=ANY(%s::text[]))
        AND d.body->>'decisionDate'<=%s
        ORDER BY relevance DESC,d.revision DESC LIMIT 13''',(terms,ticker,excluded,as_of))
    rows=cur.fetchall()
    selected=[dict(r) for r in rows[:12] if r['relevance']>0]
    return selected,{'method':'literal_question_and_case_terms' if focus else 'literal_case_terms','terms':terms,'termsLimited':limited,
        'selectedIds':[str(r['id']) for r in selected],
        'additionalMatchesOmitted':len(rows)>12 and rows[12]['relevance']>0,
        'scope':'Up to 12 older matching decisions in addition to the latest 20. Literal matching can miss relevant history. No original documents or meeting answers retrieved.'}
