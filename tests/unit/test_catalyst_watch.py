import unittest
from catalyst_watch import candidate

class CatalystWatchTests(unittest.TestCase):
    def setUp(self):
        self.a={'headline':'AbbVie reports positive Phase 3 clinical trial results','url':'https://example.com/result','datetime':1000,'related':'ABBV'}
    def test_clinical_result_is_research_signal_not_a_materiality_verdict(self):
        v=candidate('ABBV',self.a,900,1100)
        self.assertEqual(v['category'],'clinical');self.assertIn('confirm company',v['reason'])
    def test_old_future_unrelated_and_preview_news_are_not_triggers(self):
        for change in [{'datetime':899},{'datetime':2000},{'related':'PFE'},{'headline':'AbbVie will present positive Phase 3 trial results'},{'headline':'AbbVie clinical trial recruiting patients'},{'url':'javascript:alert(1)'}]:
            with self.subTest(change=change):self.assertIsNone(candidate('ABBV',{**self.a,**change},900,1100))
    def test_syndication_deduplicates_by_ticker_headline_and_day(self):
        a=candidate('ABBV',self.a,900,1100);b=candidate('ABBV',{**self.a,'url':'https://another.com/same','datetime':1010},900,1100)
        self.assertEqual(a['id'],b['id'])
    def test_guidance_regulatory_and_merger_signals(self):
        for headline,kind in [('AbbVie raises guidance','guidance'),('AbbVie receives FDA approval','regulatory'),('AbbVie signs definitive agreement','corporate')]:
            self.assertEqual(candidate('ABBV',{**self.a,'headline':headline},900,1100)['category'],kind)

class WatchDispatchTests(unittest.TestCase):
    def setUp(self):
        import json
        from contextlib import contextmanager
        from datetime import datetime,timezone
        from flask import Flask
        from catalyst_watch import CatalystWatch
        self.now=datetime.now(timezone.utc).timestamp();self.jobs={}
        self.state={'enabled':True,'automatic':True,'tickers':['ABBV'],'dailyLimit':1,'used':0,'day':'','cursor':0,'lastCheck':0,'enabledAt':self.now-100}
        owner=self
        class Cursor:
            def execute(self,sql,args=()):
                self.row=None
                self.rows=[]
                if sql.startswith('SELECT value FROM app_settings WHERE key=%s'):self.row={'value':json.dumps(owner.state)}
                elif "key='finnhub_api_key'" in sql:self.row={'value':'fake-test-key'}
                elif "key='collection_control_snapshot'" in sql:self.row={'value':{'policies':[{'ticker':'ABBV','enabled':True,'hours':168,'lookbackDays':7,'workflow':'thesis','kinds':['transcript'],'topic':'','instructions':''}]}}
                elif sql.startswith('INSERT INTO app_settings'):owner.state=json.loads(args[1])
                elif sql.startswith('SELECT id FROM analysts'):self.row={'id':'covering-analyst'}
                elif sql.startswith('SELECT id FROM mp_jobs'):self.row=owner.jobs.get(args[0])
                elif sql.startswith('SELECT id,input,result FROM mp_jobs'):self.rows=[j for j in owner.jobs.values() if j['stage']=='catalyst_signal']
                elif sql.startswith('INSERT INTO mp_jobs'):
                    if "'collection_control'" in sql:owner.jobs[args[0]]={'id':args[0],'stage':'collection_control','input':json.loads(args[2])}
                    else:owner.jobs[args[0]]={'id':args[0],'stage':args[1],'status':args[3],'input':json.loads(args[4]),'result':json.loads(args[5])}
                elif not sql.startswith('SELECT pg_advisory'):raise AssertionError(sql)
            def fetchone(self):return self.row
            def fetchall(self):return self.rows
        @contextmanager
        def db(commit=False):yield None,Cursor()
        self.watch=CatalystWatch(Flask(__name__),db,lambda:True)
        self.article={'headline':'AbbVie reports positive Phase 3 clinical trial results','url':'https://example.com/result','datetime':self.now-1,'related':'ABBV'}

    def tick(self,articles):
        from unittest.mock import patch,Mock
        response=Mock();response.json.return_value=articles
        with patch('catalyst_watch.requests.get',return_value=response):self.watch.tick()

    def test_automatic_signal_queues_one_event_command_and_never_replays(self):
        self.tick([self.article]);commands=[j for j in self.jobs.values() if j['stage']=='collection_control']
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0]['input']['payload']['policy']['workflow'],'recap')
        self.assertIn('press-release',commands[0]['input']['payload']['policy']['kinds'])
        count=len(self.jobs);self.state['lastCheck']=0;self.tick([self.article]);self.assertEqual(len(self.jobs),count)

    def test_daily_limit_retains_additional_signal_without_paid_dispatch(self):
        self.tick([self.article,{**self.article,'headline':'AbbVie cuts guidance after earnings results'}])
        self.assertEqual(len([j for j in self.jobs.values() if j['stage']=='collection_control']),1)
        self.assertEqual(len([j for j in self.jobs.values() if j.get('status')=='detected']),1)

    def test_detection_only_does_not_queue_collection(self):
        self.state['automatic']=False;self.tick([self.article])
        self.assertEqual(len(self.jobs),1);self.assertEqual(next(iter(self.jobs.values()))['status'],'detected')

    def test_different_luna_headlines_hold_second_dispatch_across_scans(self):
        self.state['dailyLimit']=10
        self.tick([{**self.article,'headline':'AbbVie Extends Migraine Leadership with Positive Phase 3 Atogepant Results in Menstrual Migraine'}])
        self.state['lastCheck']=0
        self.tick([{**self.article,'headline':'AbbVie Announces Topline Results From Its Phase 3 LUNA Study Of Atogepant For Preventive Treatment Of Menstrual Migraine In Adults; Meeting Primary And All Eight Ranked Secondary Endpoints Versus Placebo'}])
        self.assertEqual(len([j for j in self.jobs.values() if j['stage']=='collection_control']),1)
        held=[j for j in self.jobs.values() if j.get('result',{}).get('duplicateReview')]
        self.assertEqual(len(held),1)
        self.assertTrue(held[0]['result']['commandId'])
        self.assertEqual(self.state['used'],1)

    def test_mining_results_do_not_create_clinical_assignment(self):
        self.tick([{**self.article,'headline':'Myriad Uranium Announces Further Phase II Drill Results from Copper Mountain'}])
        self.assertEqual(self.jobs,{})
