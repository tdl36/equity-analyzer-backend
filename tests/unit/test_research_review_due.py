import unittest
from datetime import datetime,timezone
from research_review_due import due_items

class Cursor:
    def __init__(self,rows):self.rows=rows;self.params=[]
    def execute(self,q,p=None):self.params.append(p)
    def fetchone(self):return {'name':'research_decisions'}
    def fetchall(self):return self.rows

class ReviewDueTests(unittest.TestCase):
    def test_new_york_date_future_invalid_and_old_issue_visibility(self):
        now=datetime(2026,10,11,1,tzinfo=timezone.utc)
        rows=[dict(id=str(i),ticker='TEST',revision=1,issue='Demand recovery',due=d,created_at='2026-01-01T00:00:00') for i,d in enumerate(['2026-10-10','2026-10-11','2026-02-31','2026-10-01'])]
        cur=Cursor(rows);items,more=due_items(cur,now)
        self.assertFalse(more);self.assertEqual(cur.params[-1],('2026-10-10',201))
        self.assertEqual(len(items),2);self.assertIn('today',items[0]['step']);self.assertIn('overdue',items[1]['step'])
        self.assertTrue(all(i['bucket']=='attention' and not i['stale'] for i in items))
        self.assertNotIn('rationale',items[0])
    def test_bounded_snapshot_reports_more(self):
        row=dict(id='id',ticker='TEST',revision=1,issue='Question',due='2026-10-01',created_at='2026-01-01')
        items,more=due_items(Cursor([row]*201),datetime(2026,10,11,tzinfo=timezone.utc))
        self.assertEqual(len(items),200);self.assertTrue(more)
