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

    def test_malformed_audit_json_retries_once_then_validates_normally(self):
        from recap_validation import structured_call
        from unittest.mock import Mock
        call=Mock(side_effect=['{"claims":', '{"claims":[]}'])
        self.assertEqual(structured_call(call,'Audit',100),{'claims':[]})
        self.assertEqual(call.call_count,2)
        self.assertIn('FORMAT RETRY',call.call_args.args[0])
        call=Mock(return_value='broken')
        with self.assertRaises(ValueError):structured_call(call,'Audit',100)
        self.assertEqual(call.call_count,2)

    def test_long_first_document_does_not_starve_later_audit_sources(self):
        from recap_validation import audit_excerpts
        sources=[{'id':str(i),'filename':str(i)+'.pdf','pages':[{'page':1,'text':'x'*5000}]} for i in range(3)]
        excerpts,issues=audit_excerpts(sources,budget=3000)
        self.assertEqual([len(s['pages'][0]['text']) for s in excerpts],[1000,1000,1000])
        self.assertEqual(len(issues),3)
