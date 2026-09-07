import tempfile
import unittest
from pathlib import Path
from collection_refresh import RefreshManager
from charlie_collector import Collector


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);(root/'STOCKS'/'MDT').mkdir(parents=True)
        (root/'CATALYSTS'/'MDT').mkdir(parents=True)
        self.c=Collector(root/'state',root/'STOCKS');self.addCleanup(self.c.db.close)
        self.clock=1788796800.;self.m=RefreshManager(self.c,lambda:self.clock)
        self.cfg={'ticker':'MDT','hours':24,'lookbackDays':7,'kinds':['transcript']}

    def test_manual_trigger_coalesces_without_duplicate_browser_runs(self):
        self.m.save(self.cfg)
        a=self.m.trigger('MDT');b=self.m.trigger('MDT')
        self.assertEqual(a,b)
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],1)

    def test_schedule_waits_until_due_and_does_not_backfill_missed_slots(self):
        self.m.save(self.cfg);self.assertEqual(self.m.due(),[])
        self.clock+=3*86400
        self.assertEqual(len(self.m.due()),1)
        self.assertEqual(self.m.due(),[])

    def test_global_browser_lease_blocks_overlapping_workers(self):
        self.m.save(self.cfg);self.m.trigger('MDT');first=self.m.claim()
        self.assertIsNotNone(first);self.assertIsNone(self.m.claim())
        self.clock+=1801;second=self.m.claim()
        self.assertEqual(first['id'],second['id']);self.assertNotEqual(first['owner'],second['owner'])
        with self.assertRaises(ValueError):self.m.mark(first['id'],first['owner'],'attention')

    def test_pause_stops_claim_and_manual_trigger(self):
        self.m.save(self.cfg);self.m.trigger('MDT');self.m.save(dict(self.cfg,enabled=False))
        self.assertIsNone(self.m.claim())
        with self.assertRaises(ValueError):self.m.trigger('MDT')

    def test_no_success_cursor_until_all_searches_verified(self):
        self.m.save(self.cfg);self.m.trigger('MDT');claim=self.m.claim()
        with self.assertRaises(ValueError):self.m.complete(claim['id'],claim['owner'])
        self.assertIsNone(self.m.status()['policies'][0]['lastSuccess'])

    def test_empty_verified_search_advances_cursor_without_research(self):
        self.m.save(self.cfg);self.m.trigger('MDT');claim=self.m.claim();run=claim['run']
        self.c.observe(run,'MDT','transcript','https://research.alpha-sense.com/search',0,'Reviewed empty fixed-window search')
        self.c.finish(run,'MDT','transcript',0)
        result=self.m.complete(claim['id'],claim['owner'],fetcher=lambda tk:{'files':[]})
        self.assertEqual(result['newDocuments'],0)
        self.assertEqual(self.m.status()['requests'][0]['status'],'complete')
        self.clock+=86400
        self.m.trigger('MDT');next_claim=self.m.claim()
        self.assertEqual(next_claim['collection']['since'],'2026-09-05')

    def test_event_folder_is_created_only_under_existing_ticker(self):
        self.m.save(dict(self.cfg,workflow='recap',topic='MDT F1Q27 Earnings'))
        self.m.trigger('MDT')
        self.assertTrue((self.c.catalysts/'MDT'/'MDT F1Q27 Earnings').is_dir())
        with self.assertRaises(ValueError):self.m.save(dict(self.cfg,workflow='recap',topic='../escape'))

    def test_auth_pause_persists_and_requires_explicit_resume(self):
        self.m.save(self.cfg);self.m.trigger('MDT');claim=self.m.claim()
        self.c.auth(claim['run'],True);self.m.mark(claim['id'],claim['owner'],'needs_auth','Sign in in Chrome')
        self.assertIsNone(self.m.claim());self.m.retry(claim['id'])
        again=self.m.claim();self.assertEqual(again['id'],claim['id'])
        self.assertEqual(again['collection']['tasks'][0]['status'],'pending')

    def test_invalid_tickers_cadences_and_empty_sources_rejected(self):
        for change in ({'ticker':'BAD/NAME'},{'hours':2},{'kinds':[]},{'lookbackDays':0}):
            with self.assertRaises(ValueError):self.m.save(dict(self.cfg,**change))

    def test_cancel_revokes_worker_and_preserves_audit(self):
        self.m.save(self.cfg);self.m.trigger('MDT');claim=self.m.claim()
        self.assertEqual(claim['lease_until'],self.clock+1800)
        self.m.cancel(claim['id'])
        with self.assertRaises(ValueError):self.m.mark(claim['id'],claim['owner'],'collecting')
        self.assertEqual(self.m.status()['requests'][0]['status'],'cancelled')
        self.assertIsNone(self.m.status()['policies'][0]['lastSuccess'])
        self.assertIsNotNone(self.c.status(claim['run']))
        self.assertNotEqual(self.m.trigger('MDT'),claim['id'])
