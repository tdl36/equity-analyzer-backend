import unittest
from recap_validation import catalog, validate_claims, audit

class RecapValidationTests(unittest.TestCase):
    def setUp(self):
        self.parts=[{'name':'release.txt','type':'text','content':'Revenue grew ten percent during the reported first fiscal quarter.'}]
        self.sources,_=catalog(self.parts)
        self.claim={'statement':'Revenue grew ten percent.','sourceId':'s1','page':None,'quote':self.parts[0]['content'],'kind':'reported_fact'}
    def test_fabricated_quote_never_passes(self):
        c=dict(self.claim,quote='Revenue doubled and margins increased significantly this quarter.')
        self.assertFalse(validate_claims({'claims':[c]},self.sources)[0]['passageMatched'])
    def test_wrong_page_or_source_does_not_match(self):
        for changes in ({'sourceId':'missing'},{'page':2}):
            self.assertFalse(validate_claims({'claims':[dict(self.claim,**changes)]},self.sources)[0]['passageMatched'])
    def test_quote_match_is_not_independent_approval(self):
        c=validate_claims({'claims':[self.claim]},self.sources)[0]
        self.assertTrue(c['passageMatched']);self.assertFalse(c['reviewPassed'])
    def test_adverse_review_and_missing_baseline_remain_visible(self):
        responses=iter([{'claims':[self.claim],'changes':[{'area':'earnings','change':'Increase','baselineAvailable':True,'claimIds':['1']}]},
            {'checks':[{'id':'1','verdict':'revise','issue':'Claim period is missing'}]}])
        result=audit(self.parts,'Draft','',lambda *args:next(responses))
        self.assertFalse(result['claims'][0]['reviewPassed'])
        self.assertFalse(result['changes'][0]['baselineAvailable'])
        self.assertEqual(result['status'],'needs_review')
    def test_matching_claim_with_unique_review_retains_manual_review(self):
        responses=iter([{'claims':[self.claim]}, {'checks':[{'id':'1','verdict':'pass','issue':''}]}])
        result=audit(self.parts,'Draft','',lambda *args:next(responses))
        self.assertTrue(result['claims'][0]['reviewPassed']);self.assertEqual(result['status'],'needs_review')
