#!/usr/bin/env python3
"""Offline mechanical evaluation of a saved recap against an annotated source pack.
No model requests; passing does not certify investment judgment or entailment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recap_validation import catalog,validate_claims
from financial_benchmark import score as financial_score


def evaluate(candidate,pack):
    text=candidate.get('markdown','');audit=candidate.get('claimReview') or {}
    sources,issues=catalog(pack['sourceParts'])
    provenance=pack.get('provenance') or {}
    if provenance.get('excerptSha256'):
        payload='\n'.join(p.get('content','') for p in pack['sourceParts'])
        if hashlib.sha256(payload.encode()).hexdigest()!=provenance['excerptSha256']:
            issues.append('Annotated public-source excerpt hash changed')
    expected_hashes=pack.get('sourceHashes')
    if expected_hashes is not None:
        actual={p['name']:hashlib.sha256(p.get('content','').encode()).hexdigest() for p in pack['sourceParts']}
        if actual!=expected_hashes or len(actual)!=len(pack['sourceParts']):issues.append('Frozen source pack content or identity changed')
    financial=financial_score(candidate,pack)
    claims=validate_claims(audit,sources)
    expected=pack.get('requiredPatterns',[])
    missing=[p for p in expected if not re.search(p,text,re.I)]
    prohibited=[p for p in pack.get('forbiddenPatterns',[]) if re.search(p,text,re.I)]
    unmatched=[c['id'] for c in claims if not c['passageMatched']]
    review_count=sum(c.get('reviewPassed') is True for c in audit.get('claims',[]))
    passed=bool(claims) and not (issues or missing or prohibited or unmatched or financial['errors']) and review_count==len(claims)
    return {'passed':passed,'financialChecks':financial,'requiredPatternCoverage':(len(expected)-len(missing))/len(expected) if expected else None,
        'selectedClaims':len(claims),'matchedQuotes':len(claims)-len(unmatched),'missingPatterns':missing,
        'prohibitedPatterns':prohibited,'unmatchedClaims':unmatched,'extractionIssues':issues,
        'scope':'Mechanical regression only. Pattern presence and source matches do not certify numerical reasoning, entailment or investment judgment.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',default=str(Path(__file__).resolve().parents[1]/'evals/research-quality/earnings-synthetic.json'))
    parser.add_argument('--candidate',help='Saved JSON with markdown and claimReview')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--suite',action='store_true',help='Run positive/negative controls for every checked-in source pack')
    args=parser.parse_args()
    if args.suite:
        checks=[]
        for path in sorted((Path(__file__).resolve().parents[1]/'evals/research-quality').glob('*.json')):
            pack=json.loads(path.read_text());positive=evaluate(pack['good'],pack);negative=evaluate(pack['bad'],pack)
            adversarial=[evaluate(c,pack) for c in pack.get('badCases',[])]
            checks.append({'pack':path.name,'passed':positive['passed'] and not negative['passed'] and all(not c['passed'] for c in adversarial),'negativeControls':1+len(adversarial),'sourceType':pack.get('sourceType','synthetic'),'expertReviewed':(pack.get('annotation') or {}).get('expertReviewed',False)})
        result={'passed':bool(checks) and all(c['passed'] for c in checks),'packs':checks,'scope':'Control-fixture regression, not a benchmark of a live model or investment judgment.'}
        print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
    pack=json.loads(Path(args.pack).read_text())
    if args.self_test:
        good=evaluate(pack['good'],pack);bad=evaluate(pack['bad'],pack)
        result={'passed':good['passed'] and not bad['passed'],'positive':good,'negative':bad}
    else:
        if not args.candidate:parser.error('Provide --candidate or --self-test')
        result=evaluate(json.loads(Path(args.candidate).read_text()),pack)
    print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
