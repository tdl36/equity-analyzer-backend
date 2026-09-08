import copy
import json
import tempfile
import unittest
from pathlib import Path
from research_commands import plan, favorites, revision, DEFAULTS
from research_task_sources import collect, filing_sources, verify_public_sources
from collection_refresh import RefreshManager
from charlie_collector import Collector
from catalyst_sources import inventory


class CommandTests(unittest.TestCase):
    def test_explicit_window_and_reusable_tokens(self):
        p=plan(dict(ticker='unh',instruction=DEFAULTS[0]['instruction'],kind='filing',date='2026-09-07',days=3))
        self.assertEqual(p['since'],'2026-09-05')
        self.assertIn("UNH's",p['instruction']);self.assertIn('2026-09-07',p['instruction'])

    def test_invalid_inputs_fail_before_queue(self):
        d=dict(ticker='UNH',instruction='Review',kind='filing',date='2026-09-07',days=1)
        for change in ({'ticker':'../UNH'},{'date':'2999-01-01'},{'days':True},{'days':0},{'kind':'shell'},{'instruction':' '}):
            with self.assertRaises(ValueError):plan({**d,**change})
        with self.assertRaises(ValueError):plan([])

    def test_favorite_validation_and_optimistic_revision(self):
        self.assertEqual(favorites(DEFAULTS),DEFAULTS)
        changed=copy.deepcopy(DEFAULTS);changed[0]['name']='My filing brief'
        self.assertNotEqual(revision(DEFAULTS),revision(changed))
        for invalid in ([],[DEFAULTS[0],DEFAULTS[0]],[{**DEFAULTS[0],'days':False}]):
            with self.assertRaises(ValueError):favorites(invalid)


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);(root/'STOCKS'/'UNH').mkdir(parents=True);(root/'CATALYSTS'/'UNH').mkdir(parents=True)
        self.c=Collector(root/'state',root/'STOCKS');self.addCleanup(self.c.db.close)
        self.m=RefreshManager(self.c,lambda:1788796800.)
        self.m.save(dict(ticker='UNH',hours=168,kinds=['transcript']))
        self.p=plan(dict(ticker='UNH',instruction='Review {ticker} on {date}',kind='filing',date='2026-09-07',days=1))
        self.command={'action':'research_task','payload':self.p}
        self.result=self.m.apply_cloud_command('command-001',self.command);self.claim=self.m.claim()
        self.root=self.c.catalysts/'UNH'/self.result['topic'];self.calls=[]

    def fetch(self,url):
        self.calls.append(url)
        if url.endswith('company_tickers.json'):return json.dumps({'0':{'ticker':'UNH','cik_str':123}}).encode()
        if 'submissions' in url:
            return json.dumps({'filings':{'recent':{'form':['8-K','10-Q'],'filingDate':['2026-09-07','2026-09-07'],'accessionNumber':['0000000123-26-000001','ignored'],'primaryDocument':['filing.htm','ignored']}}}).encode()
        if url.endswith('index.json'):return json.dumps({'directory':{'item':[{'name':'ex99-1.htm'},{'name':'logo.jpg'}]}}).encode()
        return b'<html><body>Verified test filing evidence</body></html>'

    def test_command_preserves_weekly_policy_and_replays_once(self):
        self.assertEqual(self.m.apply_cloud_command('command-001',self.command),self.result)
        self.assertEqual(self.m.status()['policies'][0]['hours'],168)
        self.assertEqual(self.claim['collection']['since'],'2026-09-07')
        self.assertEqual(self.claim['config']['instructions'],'Review UNH on 2026-09-07')
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM refresh_requests').fetchone()[0],1)
        self.assertTrue(inventory(self.root)[1])

    def test_sources_filter_forms_exhibits_and_preserve_provenance(self):
        result=collect(self.m,self.claim['id'],self.claim['owner'],self.fetch)
        self.assertEqual(len(result['documents']),2);self.assertTrue(result['complete'])
        self.assertTrue(all(d['url'].startswith('https://www.sec.gov/Archives/') for d in result['documents']))
        before=len(self.calls);self.assertEqual(collect(self.m,self.claim['id'],self.claim['owner'],self.fetch),result)
        self.assertEqual(len(self.calls),before)

    def test_interruption_resumes_only_remaining_sources(self):
        def fail(url):
            if url.endswith('ex99-1.htm'):raise ValueError('temporary retrieval failure')
            return self.fetch(url)
        with self.assertRaises(ValueError):collect(self.m,self.claim['id'],self.claim['owner'],fail)
        self.assertEqual(len(list(self.root.glob('SEC*'))),1)
        with self.assertRaises(ValueError):verify_public_sources(self.m,self.c.db.execute('SELECT * FROM refresh_requests').fetchone(),lambda tk:{'files':[]})
        self.calls=[];result=collect(self.m,self.claim['id'],self.claim['owner'],self.fetch)
        self.assertEqual(len(result['documents']),2)
        self.assertEqual(len(self.calls),1);self.assertTrue(self.calls[0].endswith('ex99-1.htm'))

    def test_revoked_lease_prevents_writing_sources(self):
        self.m.cancel(self.claim['id'])
        with self.assertRaises(ValueError):collect(self.m,self.claim['id'],self.claim['owner'],self.fetch)
        self.assertEqual(list(self.root.glob('SEC*')),[])

    def test_cancel_during_fetch_prevents_next_write(self):
        def cancel(url):
            data=self.fetch(url)
            if url.endswith('filing.htm'):self.m.cancel(self.claim['id'])
            return data
        with self.assertRaises(ValueError):collect(self.m,self.claim['id'],self.claim['owner'],cancel)
        self.assertEqual(list(self.root.glob('SEC*')),[])

    def test_manifest_requires_exact_folder_and_unchanged_hash(self):
        result=collect(self.m,self.claim['id'],self.claim['owner'],self.fetch)
        row=self.c.db.execute('SELECT * FROM refresh_requests').fetchone()
        with self.assertRaises(ValueError):verify_public_sources(self.m,row,lambda tk:{'files':[]})
        manifest={'files':[{'filename':d['filename'],'folder':'Catalysts/'+self.result['topic']} for d in result['documents']]}
        self.assertEqual(verify_public_sources(self.m,row,lambda tk:manifest),2)
        (self.root/result['documents'][0]['filename']).write_text('changed')
        with self.assertRaises(ValueError):verify_public_sources(self.m,row,lambda tk:manifest)

    def test_failed_lookup_is_not_zero_sources(self):
        def fail(url):raise ValueError('SEC blocked retrieval')
        with self.assertRaises(ValueError):filing_sources('UNH','2026-09-07','2026-09-07',fail)

    def test_unqueried_archive_is_not_empty_success(self):
        def historical(url):
            if 'submissions' in url:return json.dumps({'filings':{'files':[{'filingFrom':'2020-01-01','filingTo':'2021-01-01'}],'recent':{}}}).encode()
            return self.fetch(url)
        with self.assertRaises(ValueError):filing_sources('UNH','2020-02-01','2020-02-02',historical)

    def test_empty_search_completes_without_dispatch_and_removes_gate(self):
        def empty(url):
            if 'submissions' in url:return b'{"filings":{"recent":{}}}'
            return self.fetch(url)
        collect(self.m,self.claim['id'],self.claim['owner'],empty)
        for task in self.claim['collection']['tasks']:
            self.c.observe(self.claim['run'],'UNH',task['kind'],'https://research.alpha-sense.com/search',0,'Verified empty search')
            self.c.finish(self.claim['run'],'UNH',task['kind'],0)
        self.m.dispatch=lambda *args:self.fail('No-source task must not dispatch')
        result=self.m.complete(self.claim['id'],self.claim['owner'],lambda tk:{'files':[]})
        self.assertEqual(result['publicDocuments'],0);self.assertIn('no recap',result['research'])
        self.assertFalse((self.root/'.charlie-collection-pending').exists())
        self.assertIsNone(self.m.status()['policies'][0]['lastSuccess'])

if __name__=='__main__':unittest.main()
