import unittest
from contextlib import contextmanager
from flask import Flask
from pipeline_recovery import create_blueprint, update_managed

OWNER='11111111-1111-1111-1111-111111111111'
OTHER='22222222-2222-2222-2222-222222222222'
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'job','status':'queued','job_type':'synthesis','agent_owner':None,'agent_managed':False,'recovery_attempts':0,'steps_detail':{'activityId':'activity'}};self.calls=[];outer=self
        class Cursor:
            def execute(self,sql,args=()):
                outer.calls.append((sql,args))
                if sql.startswith("UPDATE research_pipeline_jobs SET status='running',agent_owner"):
                    outer.row.update(status='running',agent_owner=args[0],agent_managed=args[1])
                elif sql.startswith("UPDATE research_pipeline_jobs SET status='queued'"):
                    outer.row.update(status='queued',agent_owner=None,recovery_attempts=outer.row['recovery_attempts']+1)
                elif sql.startswith("UPDATE research_pipeline_jobs SET status='failed'"):
                    outer.row.update(status='failed',agent_owner=None)
            def fetchone(self):return outer.row
            def fetchall(self):return [dict(outer.row)]
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        self.db=db;app=Flask(__name__);app.register_blueprint(create_blueprint(db));self.client=app.test_client()
    def claim(self,owner=OWNER):return self.client.post('/api/agent/claim',json={'jobId':'job','claimToken':owner})
    def test_claim_is_atomic_idempotent_and_competing_owner_blocked(self):
        self.assertEqual(self.claim().status_code,200);self.assertTrue(self.row['agent_managed'])
        self.assertEqual(self.claim().status_code,200);self.assertEqual(self.claim(OTHER).status_code,409)
    def test_only_token_aware_synthesis_is_recoverable(self):
        self.row['job_type']='note';self.assertEqual(self.claim().status_code,200);self.assertFalse(self.row['agent_managed'])
    def test_invalid_token_cannot_claim(self):
        self.assertEqual(self.claim('bad').status_code,400);self.assertFalse(self.calls)
    def test_recovery_preserves_job_id_and_bounds_retries(self):
        for count in range(3):
            self.row.update(status='running',agent_owner=OWNER,recovery_attempts=count,agent_managed=True)
            r=self.client.post('/api/agent/recover-jobs',json={})
            self.assertEqual(r.status_code,200)
            self.assertEqual(self.row['status'],'queued' if count<2 else 'failed')
            self.assertEqual(self.row['id'],'job')
        sql=self.calls[0][0];self.assertIn("agent_lease_until<NOW()",sql);self.assertIn('SKIP LOCKED',sql)
    def test_revoked_owner_cannot_update_progress(self):
        self.claim();before=len(self.calls)
        response=update_managed(self.db,{'jobId':'job','claimToken':OTHER,'status':'running'})
        self.assertEqual(response[1],409);self.assertEqual(len(self.calls),before+1)
    def test_completed_result_must_use_receipt_endpoint(self):
        self.claim();response=update_managed(self.db,{'jobId':'job','claimToken':OWNER,'status':'complete'})
        self.assertEqual(response[1],400)
    def test_lease_renewal_checks_exact_owner_and_running_status(self):
        r=self.client.post('/api/agent/job-leases',json={'jobs':[{'id':'job','claimToken':OWNER}]})
        self.assertEqual(r.status_code,200)
        sql,args=self.calls[-1];self.assertIn("agent_owner=%s AND status='running'",sql);self.assertEqual(args,('job',OWNER))
