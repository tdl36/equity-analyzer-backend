import copy
import unittest
from stock_evidence import valid_citations
from research_assignment_outputs import fields
from research_infographic import Editor
class EvidenceIdentityTests(unittest.TestCase):
    def fixture(self):
        return {'sources':[{'id':'s'}],'state':{'report':{'summary':{'investment_thesis':['Exact claim'],'one_liner':'Unavailable'},'financials':{'zero':0}},'citations':{
            '/summary/investment_thesis/0':{'statement':'Exact claim','review':'supported','passageMatched':True,'evidence':[{'matched':True,'sourceId':'s'}]}}}}
    def test_real_field_identity_is_shared_by_derivative_selection(self):
        r=self.fixture();before=copy.deepcopy(r)
        self.assertEqual(len(valid_citations(r)),1);self.assertEqual(len(fields(r,'/summary/')),1);self.assertEqual(len(Editor(r).cs),1)
        self.assertEqual(r,before)
    def test_stale_orphan_unknown_source_and_unavailable_are_omitted(self):
        for mutate in [lambda r:r['state']['report']['summary'].update(investment_thesis=['Changed']),lambda r:r.update(sources=[]),lambda r:r['state']['citations']['/summary/investment_thesis/0'].update(passageMatched='true')]:
            r=self.fixture();mutate(r);self.assertEqual(valid_citations(r),{})
        r=self.fixture();c=r['state']['citations']['/summary/investment_thesis/0']
        r['state']['citations']['/summary/one_liner']={**c,'statement':'Unavailable'}
        r['state']['citations']['/summary/orphan']={**c,'statement':'Orphan'}
        self.assertEqual(len(valid_citations(r)),1)
    def test_zero_and_escaped_pointer_are_preserved(self):
        r=self.fixture();c=r['state']['citations']['/summary/investment_thesis/0']
        r['state']['citations']['/financials/zero']={**c,'statement':'0'}
        r['state']['report']['summary']['a/b~c']='Escaped'
        r['state']['citations']['/summary/a~1b~0c']={**c,'statement':'Escaped'}
        self.assertEqual(len(valid_citations(r)),3)
