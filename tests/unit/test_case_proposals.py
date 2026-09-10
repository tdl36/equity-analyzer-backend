import unittest,uuid,copy
from research_amendments import editable_fields,validate_changes,baseline_current
from investment_case import source_change

class CaseProposalTests(unittest.TestCase):
    def test_stable_assumption_targets_and_context_fencing(self):
        aid=str(uuid.uuid4());body={'thesis':'Recovery view','assumptions':[{'id':aid,'claim':'Margins recover','evidenceType':'interpretation','support':'Old support'}]}
        baseline={'_investmentCase':{'revision':1,'body':body}}
        path=f'assumptions.{aid}.support'
        self.assertIn(path,editable_fields(baseline))
        self.assertNotIn(f'assumptions.{aid}.claim',editable_fields(baseline))
        excerpt='Management expects improvement subject to demand.'
        sources=[{'id':'s','filename':'Transcript','text':excerpt}]
        raw={'changes':[{'path':path,'after':excerpt,'reason':'New guidance','source_id':'s','source_excerpt':excerpt}]}
        change=validate_changes(raw,baseline,sources)[0];change['reviewPassed']=True
        job={'id':'j','status':'awaiting_approval','result':{'changes':[change],'sources':sources}}
        selection={'changeId':change['id'],'assumptionId':aid,'field':'support'}
        result=source_change(body,job,selection)
        self.assertEqual(result['assumptions'][0]['support'],excerpt)
        self.assertEqual(body['assumptions'][0]['support'],'Old support')
        for updated in ({**body,'thesis':'Changed view'}, {**body,'assumptions':[{**body['assumptions'][0],'support':'New user edit'}]}):
            with self.assertRaises(ValueError):source_change(updated,job,selection)
        with self.assertRaises(ValueError):source_change(body,job,{**selection,'field':'contrary'})
        bad=copy.deepcopy(raw);bad['changes'][0]['path']=f'assumptions.{aid}.claim'
        with self.assertRaises(ValueError):validate_changes(bad,baseline,sources)
    def test_recovery_checks_case_revision_not_legacy_thesis(self):
        body={'thesis':'Case','assumptions':[]}
        class Cursor:
            def execute(self,q,args):self.query=q
            def fetchone(self):return {'revision':2,'body':{**body,'evidenceLinks':[]}}
        cur=Cursor()
        self.assertTrue(baseline_current(cur,'ABBV',{'baseline':{'_investmentCase':{'revision':2,'body':body}}}))
        self.assertIn('investment_case_versions',cur.query)
        self.assertFalse(baseline_current(cur,'ABBV',{'baseline':{'_investmentCase':{'revision':1,'body':body}}}))
