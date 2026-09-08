import json
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

    def test_snapshot_distinguishes_saved_sources_from_restricted_staging(self):
        from tests.unit.test_charlie_collector import pdf
        self.m.save(self.cfg); self.m.trigger('MDT'); claim=self.m.claim()
        source=Path(self.tmp.name)/'source.pdf';source.write_bytes(pdf())
        first=self.c.stage(claim['run'],'MDT','transcript',source,'https://research.alpha-sense.com/?docid=one')['documents'][0]
        self.c.handoff(first['id'])
        source.write_bytes(pdf(width=90))
        self.c.stage(claim['run'],'MDT','transcript',source,'https://research.alpha-sense.com/?docid=two',usage='reference_only')
        progress=self.m.status()['requests'][0]['sourceProgress']
        self.assertEqual(progress['documents'],{'handed_off':1,'held':1})
        self.assertNotEqual(progress['tasks'][0]['status'],'complete')

    def test_meeting_requests_include_presentations_and_repair_is_fenced(self):
        payload={'ticker':'MDT','until':'2026-09-07','since':'2026-09-01',
                 'kind':'event','instruction':'Prepare meeting questions',
                 'meetingPrep':{'meetingDate':'2026-09-08'}}
        result=self.m.apply_cloud_command('meeting',{'action':'research_task','payload':payload})
        claim=self.m.claim()
        self.assertEqual(result['refreshRequestId'],claim['id'])
        self.assertIn('presentation',claim['config']['kinds'])
        before=self.c.status(claim['run'])
        # Simulate a request created before presentation support.
        cfg=claim['config'];cfg['kinds'].remove('presentation')
        self.c.db.execute('UPDATE refresh_requests SET config=? WHERE id=?',(json.dumps(cfg),claim['id']))
        self.c.db.execute("DELETE FROM tasks WHERE run=? AND kind='presentation'",(claim['run'],))
        self.c.db.commit()
        with self.assertRaises(ValueError):self.m.include_meeting_presentations(claim['id'],'stale-owner')
        self.m.include_meeting_presentations(claim['id'],claim['owner'])
        self.m.include_meeting_presentations(claim['id'],claim['owner'])
        after=self.c.status(claim['run'])
        self.assertEqual(sum(t['kind']=='presentation' for t in after['tasks']),1)
        self.assertEqual(before['since'],after['since'])
        self.assertEqual(before['until_date'],after['until_date'])
        self.m.cancel(claim['id'])
        with self.assertRaises(ValueError):self.m.include_meeting_presentations(claim['id'],claim['owner'])

    def test_nonmeeting_repair_rejected(self):
        self.m.save(self.cfg);self.m.trigger('MDT');claim=self.m.claim()
        with self.assertRaises(ValueError):self.m.include_meeting_presentations(claim['id'],claim['owner'])

    def test_cloud_trigger_receipt_prevents_replay_after_collection_completes(self):
        self.m.save(self.cfg)
        value={'action':'trigger','payload':{'ticker':'MDT'}}
        first=self.m.apply_cloud_command('cloud-test',value)
        self.c.db.execute("UPDATE refresh_requests SET status='complete'");self.c.db.commit()
        self.assertEqual(self.m.apply_cloud_command('cloud-test',value),first)
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM refresh_requests').fetchone()[0],1)
        with self.assertRaises(ValueError):self.m.apply_cloud_command('cloud-test',{'action':'save','payload':self.cfg})

    def test_cloud_save_and_receipt_commit_together(self):
        self.m.apply_cloud_command('cloud-save',{'action':'save','payload':self.cfg})
        self.assertEqual(len(self.m.status()['policies']),1)
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM cloud_control_receipts').fetchone()[0],1)
        with self.assertRaises(ValueError):self.m.apply_cloud_command('invalid',{'action':'save','payload':{**self.cfg,'hours':8761}})
        self.assertEqual(self.c.db.execute('SELECT COUNT(*) FROM cloud_control_receipts').fetchone()[0],1)

    def test_cloud_batch_is_atomic_and_replay_safe(self):
        (self.c.stocks/'DE').mkdir()
        configs=[self.cfg,{**self.cfg,'ticker':'DE'}]
        first=self.m.apply_cloud_command('batch',{'action':'save_batch','payload':{'policies':configs}})
        self.assertEqual(len(first['policies']),2)
        with self.assertRaises(ValueError):self.m.apply_cloud_command('bad-batch',{'action':'save_batch','payload':{'policies':[{**self.cfg,'hours':4},{**self.cfg,'ticker':'MISSING'}]}})
        self.assertEqual(next(p['hours'] for p in self.m.status()['policies'] if p['ticker']=='MDT'),24)
        self.assertEqual(self.m.apply_cloud_command('batch',{'action':'save_batch','payload':{'policies':configs}}),first)

    def test_cloud_retry_and_cancel_validate_ticker_and_preserve_progress(self):
        self.m.save(self.cfg);rid=self.m.trigger('MDT')
        self.c.db.execute("UPDATE refresh_requests SET status='attention' WHERE id=?",(rid,));self.c.db.commit()
        cmd={'action':'retry','payload':{'ticker':'MDT','refreshRequestId':rid}}
        self.m.apply_cloud_command('retry',cmd);self.m.apply_cloud_command('retry',cmd)
        self.assertEqual(self.m.status()['requests'][0]['status'],'queued')
        with self.assertRaises(ValueError):self.m.apply_cloud_command('wrong',{'action':'cancel','payload':{'ticker':'DE','refreshRequestId':rid}})
        self.m.apply_cloud_command('cancel',{'action':'cancel','payload':{'ticker':'MDT','refreshRequestId':rid}})
        self.assertEqual(self.m.status()['requests'][0]['status'],'cancelled')

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

    def test_pause_blocks_scheduled_work_but_allows_explicit_manual_refresh(self):
        self.m.save(self.cfg);self.m.trigger('MDT');self.m.save(dict(self.cfg,enabled=False))
        self.assertEqual(self.m.due(),[])
        self.assertIsNotNone(self.m.claim())

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
        for change in ({'ticker':'BAD/NAME'},{'hours':8761},{'kinds':[]},{'lookbackDays':0}):
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

    def test_custom_biweekly_monthly_intervals_and_manual_schedule_preservation(self):
        for hours in (3,336,720,8760):
            self.m.save(dict(self.cfg,hours=hours))
            due=self.m.status()['policies'][0]['nextDue']
            self.assertEqual(due,self.clock+hours*3600)
            rid=self.m.trigger('MDT')
            self.assertEqual(self.m.status()['policies'][0]['nextDue'],due)
            self.m.cancel(rid)

    def test_event_refresh_keeps_weekly_policy_and_queues_distinct_events_once(self):
        self.m.save(dict(self.cfg,hours=168))
        cfg={**self.cfg,'workflow':'recap','topic':'MDT clinical event','kinds':['press-release']}
        value={'action':'event_refresh','payload':{'policy':cfg,'event':{'id':'event-a','reason':'A clinical result','url':'https://example.com/event'}}}
        first=self.m.apply_cloud_command('event-command',value)
        self.assertEqual(self.m.apply_cloud_command('event-command-replay',value),first)
        self.assertEqual(self.m.status()['policies'][0]['hours'],168)
        self.assertEqual(self.m.status()['policies'][0]['workflow'],'thesis')
        self.assertEqual(self.m.status()['policies'][0]['nextDue'],self.clock+168*3600)
        claim=self.m.claim();self.assertEqual(claim['config']['eventId'],'event-a')

    def test_add_company_and_refresh_commits_one_request_and_replays_safely(self):
        cfg={**self.cfg,'ticker':'ABBV','createFolder':True,'hours':336}
        cmd={'action':'save_trigger','payload':cfg}
        result=self.m.apply_cloud_command('add-once',cmd)
        self.assertTrue((self.c.stocks/'ABBV').is_dir())
        self.assertEqual(result,self.m.apply_cloud_command('add-once',cmd))
        self.assertEqual(len(self.m.status()['requests']),1)

    def test_managed_completion_keeps_pending_until_dispatch_and_records_receipt(self):
        from unittest.mock import patch
        from catalyst_sources import inventory, fingerprint, already_dispatched
        payload={'ticker':'MDT','until':'2026-09-07','since':'2026-09-01',
                 'kind':'event','instruction':'Prepare meeting questions',
                 'meetingPrep':{'meetingDate':'2026-09-08'}}
        self.m.apply_cloud_command('meeting-receipt',{'action':'research_task','payload':payload})
        claim=self.m.claim();folder=self.c.catalysts/'MDT'/claim['config']['topic']
        (folder/'source.htm').write_text('<html>Original evidence</html>')
        for task in claim['collection']['tasks']:
            self.c.observe(claim['run'],'MDT',task['kind'],'https://research.alpha-sense.com/search',0,'Verified empty search')
            self.c.finish(claim['run'],'MDT',task['kind'],0)
        def dispatch(cfg,docs):
            self.assertTrue((folder/'.charlie-collection-pending').exists())
            self.assertFalse((folder/'.charlie-dispatch-receipt.json').exists())
            return {'activities':[{'activityId':'test'}]}
        with patch('research_task_sources.verify_public_sources',return_value=1), patch('command_source_import.import_sources',return_value=1), patch.object(self.m,'dispatch',side_effect=dispatch):
            self.m.complete(claim['id'],claim['owner'],fetcher=lambda tk:{'files':[]})
        self.assertFalse((folder/'.charlie-collection-pending').exists())
        self.assertTrue(already_dispatched(folder,fingerprint(inventory(folder)[0])))
