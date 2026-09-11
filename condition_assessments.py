"""Validate AI suggestions against frozen underweight condition identities and excerpts."""
import research_evidence


def validate(raw,baseline,sources):
    rows=raw.get('condition_assessments',[]) if isinstance(raw,dict) else []
    if not isinstance(rows,list) or len(rows)>20:raise ValueError('Invalid underweight condition assessment list.')
    lookup={(w['id'],c['id']):(w,c) for w in baseline.get('_investmentCase',{}).get('underweightReviews',[])
            for c in w['body'].get('reviewConditions',[])}
    seen=set();out=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid condition assessment.')
        pair=(row.get('work_id'),row.get('condition_id'))
        if pair not in lookup or pair in seen:raise ValueError('Unknown or repeated underweight condition.')
        seen.add(pair);work,condition=lookup[pair]
        state=row.get('assessment');reason=row.get('reason')
        if state not in ('met','partly_met','not_met','unresolved') or not isinstance(reason,str) or not 1<=len(reason.strip())<=6000:
            raise ValueError('Condition assessment requires a recognized state and explanation.')
        snapshot=research_evidence.build_snapshot({'facts':[{'statement':reason,'source_id':row.get('source_id'),'source_excerpt':row.get('source_excerpt')}]},sources)
        claim=snapshot['claims'][0]
        out.append(dict(id='condition-'+str(len(out)),workId=work['id'],workRevision=work['revision'],
                        workTitle=work['body']['title'],conditionId=condition['id'],trigger=condition['trigger'],
                        assessment=state,after=reason.strip(),reason=reason.strip(),evidence=claim['evidence'],
                        passageMatched=claim['status']=='passage_matched',caseRevision=baseline['_investmentCase']['revision']))
    return out
