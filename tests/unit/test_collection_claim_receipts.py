"""Temporary local ledgers; no browser, production API or paid calls."""
import json
import sqlite3
import unittest
from unittest.mock import patch
from charlie_collector import Collector
from collection_refresh import RefreshManager
from tests.unit import test_collection_refresh as refresh_tests


class ClaimReceiptTests(unittest.TestCase):
    setUp=refresh_tests.RefreshTests.setUp
    queued_pair=refresh_tests.RefreshTests.queued_pair

    def test_lost_response_recovers_same_lease_after_manager_restart_without_new_turn(self):
        first,second=self.queued_pair();original=self.m.claim(attempt_id='worker:2026-10-16T00:25:13.497Z:1')
        self.assertEqual(original['id'],first)
        before=list(self.c.db.execute('SELECT request_id,turn FROM refresh_claim_order'))
        other=Collector(self.c.state,self.c.stocks);self.addCleanup(other.db.close)
        manager=RefreshManager(other,lambda:self.clock)
        self.clock+=100
        with patch.object(manager,'_due',side_effect=AssertionError('Replay must not enqueue due work')):
            recovered=manager.claim(attempt_id='worker:2026-10-16T00:25:13.497Z:1')
        self.assertEqual(recovered,original)
        self.assertEqual([tuple(r) for r in before],[tuple(r) for r in self.c.db.execute('SELECT request_id,turn FROM refresh_claim_order')])
        self.assertIsNone(manager.claim(attempt_id='another-worker'))
        self.assertEqual(self.c.db.execute('SELECT status FROM refresh_requests WHERE id=?',(second,)).fetchone()[0],'queued')

    def test_active_receipt_returns_current_progress_without_mutating_frozen_config(self):
        first,_=self.queued_pair();original=self.m.claim(first,'attempt')
        self.c.db.execute("UPDATE tasks SET status='needs_auth' WHERE run=?",(original['run'],));self.c.db.commit()
        recovered=self.m.claim(first,'attempt')
        self.assertEqual(recovered['config'],original['config'])
        self.assertEqual(recovered['owner'],original['owner'])
        self.assertNotEqual(recovered['collection'],original['collection'])
        self.assertNotIn('policy_config',recovered)

    def test_changed_target_is_rejected_without_exposing_or_claiming_other_work(self):
        first,second=self.queued_pair();self.m.claim(first,'attempt')
        for target in (second,None):
            with self.assertRaisesRegex(ValueError,'different request selection'):self.m.claim(target,'attempt')
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_order').fetchone()[0],1)

    def test_released_receipt_does_not_reclaim_or_change_fair_order(self):
        first,second=self.queued_pair();claimed=self.m.claim(attempt_id='attempt')
        self.m.mark(first,claimed['owner'],'queued','Bounded import pending')
        with self.assertRaisesRegex(ValueError,'Active worker lease required'):self.m.claim(attempt_id='attempt')
        next_claim=self.m.claim(attempt_id='next-attempt')
        self.assertEqual(next_claim['id'],second)

    def test_expired_and_reassigned_receipts_never_renew_or_steal_lease(self):
        first,_=self.queued_pair();old=self.m.claim(first,'attempt');self.clock+=1801
        with self.assertRaisesRegex(ValueError,'Active worker lease required'):self.m.claim(first,'attempt')
        new=self.m.claim(first,'new-attempt')
        self.assertNotEqual(new['owner'],old['owner'])
        with self.assertRaisesRegex(ValueError,'Active worker lease required'):self.m.claim(first,'attempt')
        self.assertEqual(self.m.claim(first,'new-attempt'),new)

    def test_cancelled_and_attention_receipts_do_not_restart_collection(self):
        first,_=self.queued_pair();claimed=self.m.claim(first,'attempt')
        self.m.mark(first,claimed['owner'],'attention','Browser unavailable')
        with self.assertRaises(ValueError):self.m.claim(first,'attempt')
        self.m.cancel(first)
        with self.assertRaises(ValueError):self.m.claim(first,'attempt')
        self.assertEqual(self.c.db.execute('SELECT status FROM refresh_requests WHERE id=?',(first,)).fetchone()[0],'cancelled')

    def test_live_policy_pause_blocks_scheduled_receipt_recovery(self):
        self.m.save(self.cfg);self.clock+=86401;claimed=self.m.claim(attempt_id='scheduled')
        self.assertFalse(claimed['config'].get('manual'))
        self.m.save({**self.cfg,'enabled':False})
        with self.assertRaisesRegex(ValueError,'paused'):self.m.claim(attempt_id='scheduled')

    def test_manual_receipt_retains_existing_manual_pause_exception(self):
        first,_=self.queued_pair();claimed=self.m.claim(first,'manual')
        self.m.save({**self.cfg,'enabled':False})
        self.assertEqual(self.m.claim(first,'manual')['owner'],claimed['owner'])

    def test_null_receipt_remains_null_and_requires_new_attempt_for_new_work(self):
        self.assertIsNone(self.m.claim(attempt_id='empty'))
        first,_=self.queued_pair()
        self.assertIsNone(self.m.claim(attempt_id='empty'))
        self.assertEqual(self.m.claim(attempt_id='next')['id'],first)

    def test_receipt_write_failure_rolls_back_lease_fair_turn_and_worker_claim(self):
        first,_=self.queued_pair()
        self.c.db.execute("CREATE TRIGGER deny_receipt BEFORE INSERT ON refresh_claim_receipts BEGIN SELECT RAISE(ABORT,'fixture write failure'); END");self.c.db.commit()
        with self.assertRaises(sqlite3.IntegrityError):self.m.claim(attempt_id='attempt')
        row=self.c.db.execute('SELECT status,owner,lease_until FROM refresh_requests WHERE id=?',(first,)).fetchone()
        self.assertEqual(tuple(row),('queued',None,None))
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_order').fetchone()[0],0)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_receipts').fetchone()[0],0)

    def test_attempt_id_validation_does_not_claim(self):
        first,_=self.queued_pair()
        for value in ('','x'*161,'has space','../path',True,1,{},[]):
            with self.subTest(value=value),self.assertRaises(ValueError):self.m.claim(attempt_id=value)
        self.assertEqual(self.c.db.execute('SELECT status FROM refresh_requests WHERE id=?',(first,)).fetchone()[0],'queued')

    def test_public_snapshot_never_exposes_private_claim_receipts(self):
        self.queued_pair();claimed=self.m.claim(attempt_id='private-attempt')
        snapshot=json.dumps(self.m.status())
        self.assertNotIn('private-attempt',snapshot);self.assertNotIn(claimed['owner'],snapshot)
        self.assertNotIn('refresh_claim_receipts',snapshot)

    def test_simultaneous_processes_cannot_create_two_claims_for_one_attempt(self):
        import subprocess,sys
        from concurrent.futures import ThreadPoolExecutor
        first,_=self.queued_pair()
        script="""
import json,sys
from pathlib import Path
from charlie_collector import Collector
from collection_refresh import RefreshManager
c=Collector(Path(sys.argv[1]),Path(sys.argv[2]));m=RefreshManager(c,lambda:float(sys.argv[3]))
try:
 r=m.claim(attempt_id='concurrent-attempt');print(json.dumps({'id':r['id'],'owner':r['owner']}))
except ValueError as e:
 if str(e)!='Another collection operation is active':raise
 print(json.dumps({'busy':True}))
finally:c.db.close()
"""
        def run(_):
            return json.loads(subprocess.check_output([sys.executable,'-c',script,str(self.c.state),str(self.c.stocks),str(self.clock)],text=True,timeout=15))
        with ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(run,range(2)))
        final=self.m.claim(attempt_id='concurrent-attempt')
        self.assertEqual(final['id'],first)
        self.assertTrue(any(not r.get('busy') for r in receipts))
        for r in receipts:
            if not r.get('busy'):self.assertEqual(r,{'id':first,'owner':final['owner']})
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_receipts').fetchone()[0],1)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_order').fetchone()[0],1)

    def test_cli_attempt_id_reconciles_the_same_receipt(self):
        import subprocess,sys,time
        self.clock=time.time();self.m.save({**self.cfg,'hours':0});first=self.m.trigger('MDT')
        args=[sys.executable,'collection_refresh.py','--state',str(self.c.state),'--stocks',str(self.c.stocks),
              'claim','--attempt-id','worker:cli:1']
        a=json.loads(subprocess.check_output(args,text=True,timeout=15))
        b=json.loads(subprocess.check_output(args,text=True,timeout=15))
        self.assertEqual(a['id'],first);self.assertEqual(a,b)
        self.assertEqual(self.c.db.execute('SELECT count(*) FROM refresh_claim_order').fetchone()[0],1)
