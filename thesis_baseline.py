"""Copy existing thesis text into a reviewable case draft; no AI inference."""
import hashlib,json,uuid
from investment_case import validate


def draft(ticker,analysis):
    if not isinstance(analysis,dict) or not isinstance(analysis.get('thesis'),dict):
        raise ValueError('No structured saved thesis is available to copy.')
    thesis=analysis['thesis'];digest=hashlib.sha256(json.dumps(analysis,sort_keys=True).encode()).hexdigest()
    assumptions=[]
    for i,p in enumerate(thesis.get('pillars') or []):
        if not isinstance(p,dict) or not isinstance(p.get('description'),str) or not p['description'].strip():
            raise ValueError('A saved thesis pillar has no readable description. Review it before copying.')
        title=p.get('title') or p.get('name') or p['description']
        assumptions.append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL,f'charlie:baseline:{ticker}:{digest}:{i}')),
            claim=title,support=p['description'],contrary='',nextTest='',evidenceType='interpretation',
            sourceReference=f'Copied from saved Charlie thesis, pillar {i+1}; research snapshot SHA-256 {digest}. Not independently verified original evidence.'))
    if not assumptions:raise ValueError('The saved thesis has no textual pillars. Add assumptions manually.')
    conditions=[]
    for section in ('signposts','threats'):
        for item in analysis.get(section) or []:
            if not isinstance(item,dict):raise ValueError('A saved risk or signpost needs manual review.')
            fields=[f'{k}: {item[k]}' for k in ('title','name','description','target','triggerPoints') if isinstance(item.get(k),str) and item[k]]
            if fields:conditions.append(section.upper()+'\n'+'\n'.join(fields))
    body=validate(dict(thesis=thesis.get('summary') or '',assumptions=assumptions,changeConditions='\n\n'.join(conditions)))
    return dict(body=body,sourceHash=digest,scope='Copies thesis summary, pillar text, and textual risks/signposts. Valuation, variant view and market baseline require separate review. Saved thesis is unchanged; this is an unsaved draft.')
