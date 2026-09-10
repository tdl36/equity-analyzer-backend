import unittest
import uuid
from datetime import date,timedelta
from research_decisions import validate
from company_memory import assemble,render
class DecisionTests(unittest.TestCase):
    def payload(self):return {'requestId':str(uuid.uuid4()),'revision':0,'decisionDate':'2026-01-01','decision':'Wait','rationale':'Weak evidence','revisitWhen':'New filing'}
    def test_requires_explicit_reason_and_revisit_condition(self):
        for k,v in [('rationale',''),('revisitWhen',''),('decision','x'*2001),('revision',True),('decisionDate',(date.today()+timedelta(days=1)).isoformat()),('requestId','bad'),('supersedes','bad')]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate({**self.payload(),k:v})
    def test_client_cannot_set_creation_or_supersession_status(self):
        _,_,body=validate({**self.payload(),'created_at':'fake','superseded':True})
        self.assertNotIn('created_at',body);self.assertNotIn('superseded',body)
    def test_memory_labels_historical_records_and_missing_older_history(self):
        records=[{'id':'id','revision':1,'created_at':'today','superseded':True,'body':{'decision':'Prior view'}}]
        snap=assemble('ABT',decisions=records,decisions_more=True)
        self.assertEqual(snap['entries'][0]['status'],'superseded')
        self.assertIn('older history is omitted',render(snap))
        self.assertIn('not executed trades',render(snap))
