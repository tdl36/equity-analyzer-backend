import unittest
from datetime import datetime,timezone
from intake_status import reason,annotate
class IntakeStatusTests(unittest.TestCase):
 def setUp(self):self.now=datetime(2026,10,11,1,tzinfo=timezone.utc);self.state={'enabled':True,'dailyLimit':2,'used':2,'day':'2026-10-11'}
 def test_daily_limit_uses_utc_day_and_does_not_claim_dispatch(self):
  code,text=reason(self.state,now=self.now);self.assertEqual(code,'daily_limit');self.assertIn('2026-10-12 00:00 UTC',text)
  self.assertEqual(reason({**self.state,'day':'2026-10-10'},now=self.now)[0],'worker')
 def test_pause_credentials_and_budget_take_precedence(self):
  self.assertEqual(reason({**self.state,'enabled':False},now=self.now,has_key=False)[0],'paused')
  self.assertEqual(reason(self.state,now=self.now,has_key=False)[0],'credentials')
  self.assertEqual(reason(self.state,now=self.now,budget='Budget reached')[0],'budget')
 def test_baseline_and_open_review_are_explained_without_claiming_approval(self):
  state={**self.state,'used':0}
  self.assertEqual(reason(state,now=self.now,has_thesis=False)[0],'baseline')
  self.assertEqual(reason(state,now=self.now,has_thesis=True,blocking='awaiting_approval')[0],'review')
  self.assertEqual(reason(state,now=self.now,blocking='processing')[0],'in_progress')
 def test_annotation_queries_once_per_waiting_company_and_never_writes(self):
  class Cursor:
   calls=0
   def execute(self,sql,args):
    assert sql.strip().startswith('SELECT');self.calls+=1
   def fetchone(self):return {'has_thesis':True,'blocking':'awaiting_approval'}
  cur=Cursor();events=[{'ticker':'UNH','status':'waiting'},{'ticker':'UNH','status':'waiting'},{'ticker':'SYK','status':'submitted'}]
  annotate(cur,events,{**self.state,'used':0},now=self.now)
  self.assertEqual(cur.calls,1);self.assertEqual(events[0]['waitCode'],'review');self.assertNotIn('waitReason',events[2])
