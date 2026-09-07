import copy
import json
import unittest
from contextlib import contextmanager
from flask import Flask
from research_automation import ResearchAutomation, identity, new_entries

class IntakeTests(unittest.TestCase):
    def test_deduplication_including_repeated_scans_and_reappearance(self):
        seen,entries=new_entries([], [('DE','new.pdf'),('DE','new.pdf')])
        self.assertEqual(entries,[('DE','new.pdf')])
        self.assertEqual(new_entries(seen,[('DE','new.pdf')])[1],[])
    def test_paths_hidden_types_and_invalid_tickers_do_not_queue(self):
        _,entries=new_entries([], [('DE','../a.pdf'),('BAD TICKER','a.pdf'),('DE','download.zip'),('DE','.hidden.pdf'),('AMT','report.PDF')])
        self.assertEqual(entries,[('AMT','report.PDF')])
    def test_same_filename_is_distinct_for_each_company(self):
        self.assertNotEqual(identity('DE','call.pdf'),identity('AMT','call.pdf'))

class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.state={'enabled':True,'dailyLimit':2,'seen':[],'manifestInitialized':False,'day':'','used':0}
        self.events=[];owner=self
        class Cursor:
            def execute(self,sql,args):
                if sql.startswith('INSERT INTO mp_jobs'):owner.events.append(args)
            def fetchone(self):return None
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        self.manager=ResearchAutomation(Flask(__name__),db,None,None,lambda:{},lambda:False,lambda:None)
        self.manager.load=lambda cur:copy.deepcopy(self.state)
        self.manager.save=lambda cur,state:setattr(self,'state',copy.deepcopy(state))
        self.manager.wake=lambda:None
    def test_first_manifest_baselines_existing_files_without_paid_work(self):
        self.manager.observe([('DE','old.pdf')],initial_manifest=True)
        self.assertEqual(self.events,[])
        self.manager.observe([('DE','old.pdf'),('DE','new.pdf')],initial_manifest=True)
        self.assertEqual(len(self.events),1)
        self.assertEqual(json.loads(self.events[0][-1])['filename'],'new.pdf')
    def test_pause_prevents_intake(self):
        self.state['enabled']=False
        self.manager.observe([('DE','new.pdf')])
        self.assertEqual(self.events,[])
    def test_uploaded_file_queues_without_waiting_for_first_manifest(self):
        self.manager.observe([('DE','new.pdf')])
        self.assertEqual(len(self.events),1)
        self.manager.observe([('DE','new.pdf')],initial_manifest=True)
        self.assertEqual(len(self.events),1)
    def test_daily_limit_prevents_dispatch_and_keeps_reservation_count(self):
        from datetime import datetime, timezone
        self.state.update(day=datetime.now(timezone.utc).date().isoformat(),used=2)
        self.manager.has_key=lambda:True
        self.manager.tick()
        self.assertIn('Daily comparison limit',self.state['lastIssue'])
        self.assertEqual(self.state['used'],2)
    def test_monthly_budget_blocks_dispatch(self):
        self.manager.has_key=lambda:True
        self.manager.budget_block=lambda:'Monthly budget reached'
        self.manager.tick()
        self.assertEqual(self.state['lastIssue'],'Monthly budget reached')
        self.assertEqual(self.state['used'],0)
    def test_missing_server_key_leaves_queue_untouched(self):
        self.manager.tick()
        self.assertIn('server-side',self.state['lastIssue'])
        self.assertEqual(self.state['used'],0)

if __name__=='__main__':unittest.main()
