"""Decision persistence uses only an isolated disposable PostgreSQL cluster."""
import json
import unittest
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from onepager_decisions import record_decision
from tests.unit import test_onepager_editions as editions
from tests.unit import test_stock_analysis_postgres as pg
from tests.unit.test_onepager_edition_contract import AID, EID

DID = 'b998e805-0bd4-49ce-aafa-2af123e1aace'
OTHER = 'c998e805-0bd4-49ce-aafa-2af123e1aace'


@unittest.skipUnless((pg.PG/'initdb').exists(), 'Disposable PostgreSQL unavailable')
class DecisionPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): editions.EditionPersistenceTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls): editions.EditionPersistenceTests.tearDownClass.__func__(cls)
    client = editions.EditionPersistenceTests.client

    def setUp(self):
        with self.db(True) as (_,c):
            c.execute('DROP TABLE IF EXISTS onepager_decisions; DROP TABLE IF EXISTS onepager_accepted_heads')
        editions.EditionPersistenceTests.setUp(self)
        self.assertEqual(self.client().post(self.url,json=self.body).status_code,201)
        self.record=self.client().get(self.url).json['editions'][0]['record']
        self.decision={k:self.record[k] for k in ('assignmentId','runId','fingerprint','fundamentalsDigest','parentId','correctionIds','selectionDigest')}
        self.decision.update(requestId=DID,editionId=EID,action='accept')

    def submit(self, body=None, eid=EID, db=None, investor='synthetic-investor'):
        return record_decision(db or self.db,AID,eid,body or self.decision,investor_id=investor)

    def test_concurrent_replay_and_restart_do_not_change_originals(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:self.submit(),range(2)))
        self.assertEqual(sum(r['reused'] for r in results),1)
        self.assertEqual(results[0]['decision'],results[1]['decision'])
        self.assertTrue(self.submit()['reused'])
        with self.db() as (_,c):
            c.execute('SELECT edition_id FROM onepager_accepted_heads');self.assertEqual(str(c.fetchone()['edition_id']),EID)
            c.execute('SELECT count(*) AS n FROM onepager_decisions');self.assertEqual(c.fetchone()['n'],1)
            c.execute('SELECT record FROM onepager_editions');self.assertEqual(c.fetchone()['record'],self.record)
            c.execute('SELECT state FROM stock_analysis_runs');self.assertEqual(c.fetchone()['state'],self.run['state'])
            c.execute('SELECT result FROM mp_jobs');self.assertEqual(c.fetchone()['result'],self.assignment['result'])
            c.execute('SELECT proposal FROM onepager_corrections');self.assertEqual(c.fetchone()['proposal']['review'],'pending')

    def test_competing_accepts_cannot_advance_same_parent_twice(self):
        body={**self.body,'requestId':OTHER}
        self.assertEqual(self.client().post(self.url,json=body).status_code,201)
        def attempt(eid):
            request={**self.decision,'editionId':eid,'requestId':DID if eid==EID else OTHER}
            try:return self.submit(request,eid)
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(attempt,[EID,OTHER]))
        self.assertEqual(sum(r is not None for r in results),1)
        with self.db() as (_,c):
            c.execute('SELECT count(*) AS n FROM onepager_decisions');self.assertEqual(c.fetchone()['n'],1)
            c.execute('SELECT count(*) AS n FROM onepager_accepted_heads');self.assertEqual(c.fetchone()['n'],1)

    def test_rejection_never_advances_head_or_allows_second_decision(self):
        body={**self.decision,'action':'reject'}
        self.assertEqual(self.submit(body)['decision']['action'],'reject')
        with self.db() as (_,c):
            c.execute('SELECT count(*) AS n FROM onepager_accepted_heads');self.assertEqual(c.fetchone()['n'],0)
        self.assertTrue(self.submit(body)['reused'])
        for request in (self.decision,{**self.decision,'requestId':OTHER}):
            with self.assertRaises(ValueError):self.submit(request)

    def test_replay_survives_drift_but_cannot_change_actor_or_payload(self):
        original=self.submit()
        with self.db(True) as (_,c):c.execute("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb")
        self.assertEqual(self.submit()['decision'],original['decision'])
        with self.assertRaises(ValueError):self.submit(investor='another-investor')
        with self.assertRaises(ValueError):self.submit({**self.decision,'action':'reject'})

    def test_validation_and_late_failure_roll_back_decision_and_pointer(self):
        for mutation in ("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb",
                         "UPDATE mp_jobs SET result=jsonb_set(result,'{visualFundamentals,price}','101'::jsonb)"):
            self.setUp()
            with self.db(True) as (_,c):c.execute(mutation)
            with self.assertRaises(ValueError):self.submit()
            with self.db() as (_,c):
                c.execute("SELECT to_regclass('onepager_decisions') AS name");self.assertIsNone(c.fetchone()['name'])
        self.setUp()
        @contextmanager
        def fail_commit(commit=False):
            with self.db(commit) as pair:
                yield pair
                raise RuntimeError('Synthetic failure before commit')
        with self.assertRaises(RuntimeError):self.submit(db=fail_commit)
        with self.db() as (_,c):
            c.execute("SELECT to_regclass('onepager_accepted_heads') AS name");self.assertIsNone(c.fetchone()['name'])
        self.assertFalse(self.submit()['reused'])

    def test_auth_scope_and_denied_machine_decision_route(self):
        with self.assertRaises(ValueError):self.submit(investor='')
        with self.assertRaises(ValueError):self.submit({**self.decision,'assignmentId':OTHER})
        self.assertEqual(self.client().post(self.url+'/'+EID+'/decision',json=self.decision).status_code,403)
        with self.db() as (_,c):
            c.execute("SELECT to_regclass('onepager_decisions') AS name");self.assertIsNone(c.fetchone()['name'])

    def test_successor_rollback_preserves_existing_head_and_decision(self):
        from onepager_edition_contract import prepare_edition
        from onepager_corrections import digest
        self.submit()
        selection={**self.body,'requestId':OTHER,'parentId':EID}
        with self.db(True) as (_,c):
            c.execute('SELECT * FROM onepager_corrections');corrections=[dict(r) for r in c.fetchall()]
            c.execute('SELECT * FROM document_files');docs={r['filename']:dict(r) for r in c.fetchall()}
            record=prepare_edition(self.run,self.assignment,selection,corrections,docs,self.assignment['result']['visualFundamentals'],parent_id=EID)
            c.execute('INSERT INTO onepager_editions(id,assignment_id,run_id,payload_hash,record) VALUES(%s,%s,%s,%s,%s::jsonb)',(OTHER,AID,self.run['id'],digest(selection),json.dumps(record)))
        request={**self.decision,'requestId':OTHER,'editionId':OTHER,'parentId':EID,'selectionDigest':record['selectionDigest']}
        @contextmanager
        def fail_commit(commit=False):
            with self.db(commit) as pair:
                yield pair
                raise RuntimeError('Synthetic successor commit failure')
        with self.assertRaises(RuntimeError):self.submit(request,OTHER,fail_commit)
        with self.db() as (_,c):
            c.execute('SELECT edition_id FROM onepager_accepted_heads');self.assertEqual(str(c.fetchone()['edition_id']),EID)
            c.execute('SELECT count(*) AS n FROM onepager_decisions');self.assertEqual(c.fetchone()['n'],1)
        self.assertFalse(self.submit(request,OTHER)['reused'])
        self.assertTrue(self.submit()['reused'])
        with self.db() as (_,c):
            c.execute('SELECT edition_id FROM onepager_accepted_heads');self.assertEqual(str(c.fetchone()['edition_id']),OTHER)

    def test_source_policy_cannot_change_during_decision_validation(self):
        from threading import Event
        from unittest.mock import patch
        import psycopg2
        from onepager_decision_contract import validate_decision
        entered,release=Event(),Event()
        def held(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise AssertionError('Synthetic lock wait expired')
            return validate_decision(*args,**kwargs)
        with patch('onepager_decisions.validate_decision',side_effect=held),ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.submit)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(psycopg2.errors.LockNotAvailable):
                    with self.db(True) as (_,c):
                        c.execute("SET LOCAL lock_timeout='100ms'")
                        c.execute("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb")
            finally:release.set()
            self.assertFalse(future.result()['reused'])

    def test_reads_expose_separate_decision_and_prevent_pending_preview(self):
        for action in ('accept','reject'):
            self.setUp();self.submit({**self.decision,'action':action})
            result=self.client().get(self.url)
            self.assertEqual(result.status_code,200,result.json)
            row=result.json['editions'][0]
            self.assertEqual(row['record'],self.record)
            self.assertEqual(row['status'],'accepted' if action=='accept' else 'rejected')
            self.assertEqual(row['currentAccepted'],action=='accept')
            self.assertFalse(row['stale'])
            self.assertNotIn('synthetic-investor',str(result.json))
            self.assertEqual(result.json['parentId'],EID if action=='accept' else None)
            preview=self.client().get(self.url+'/'+EID+'/preview')
            self.assertEqual(preview.status_code,409);self.assertNotIn('html',preview.json)
            replay=self.client().post(self.url,json=self.body)
            self.assertTrue(replay.json['reused']);self.assertEqual(replay.json['status'],row['status'])

    def test_new_pending_selection_binds_current_parent_and_previews(self):
        self.submit()
        stale=self.client().post(self.url,json={**self.body,'requestId':OTHER})
        self.assertEqual(stale.status_code,400)
        successor={**self.body,'requestId':OTHER,'parentId':EID}
        result=self.client().post(self.url,json=successor)
        self.assertEqual(result.status_code,201,result.json)
        preview=self.client().get(self.url+'/'+OTHER+'/preview')
        self.assertEqual(preview.status_code,200,preview.json)
        self.assertEqual(preview.json['status'],'pending')
        self.assertIn('not accepted',preview.json['html'])

    def test_inconsistent_pointer_or_decided_content_fails_read_closed(self):
        for mutation in ('DELETE FROM onepager_accepted_heads',
                "UPDATE onepager_accepted_heads SET decision_id='c998e805-0bd4-49ce-aafa-2af123e1aace'",
                "UPDATE onepager_editions SET record=jsonb_set(record,'{caveat}','\"Changed\"'::jsonb)"):
            self.setUp();self.submit()
            with self.db(True) as (_,c):c.execute(mutation)
            result=self.client().get(self.url)
            self.assertEqual(result.status_code,400,result.json)
            self.assertNotIn('editions',result.json)

    def test_investor_route_and_version_export_preserve_originals(self):
        from flask import Flask
        from onepager_editions import create_blueprint
        app=Flask(__name__);app.register_blueprint(create_blueprint(self.db, lambda:'synthetic-investor'))
        client=app.test_client();path=self.url+'/'+EID
        self.assertEqual(client.get(path+'/export').status_code,409)
        result=client.post(path+'/decision',json=self.decision)
        self.assertEqual(result.status_code,200,result.json)
        self.assertNotIn('investorId',result.json['decision'])
        self.assertTrue(client.post(path+'/decision',json=self.decision).json['reused'])
        for layout in ('brief','illustrated'):
            response=client.get(path+'/export?layout='+layout)
            self.assertEqual(response.status_code,200,response.data[:300])
            self.assertIn(b'Investor-accepted editorial edition',response.data)
            self.assertIn(EID.encode(),response.data)
            self.assertIn('attachment',response.headers['Content-Disposition'])
            self.assertIn('no-store',response.headers['Cache-Control'])
        self.assertEqual(client.get(path+'/preview').status_code,409)
        with self.db(True) as (_,c):c.execute("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb")
        response=client.get(path+'/export')
        self.assertEqual(response.status_code,409)
        self.assertNotIn('html',response.json)

    def test_rejected_export_and_forged_decision_fail(self):
        from flask import Flask
        from onepager_editions import create_blueprint
        app=Flask(__name__);app.register_blueprint(create_blueprint(self.db, lambda:'synthetic-investor'))
        client=app.test_client();path=self.url+'/'+EID
        self.assertEqual(client.post(path+'/decision',json={**self.decision,'investorId':'forged'}).status_code,409)
        self.assertEqual(client.post(path+'/decision',json={**self.decision,'action':'reject'}).status_code,200)
        self.assertEqual(client.get(path+'/export').status_code,409)

    def test_stable_view_tracks_exact_decision_without_writing_or_weakening_routes(self):
        from flask import Flask
        from onepager_editions import create_blueprint
        commits=[]
        @contextmanager
        def observed(commit=False):
            commits.append(commit)
            with self.db(commit) as pair: yield pair
        app=Flask(__name__);app.register_blueprint(create_blueprint(observed))
        client=app.test_client();path=self.url+'/'+EID
        pending=client.get(path+'/view?layout=brief')
        self.assertEqual(pending.status_code,200,pending.json)
        self.assertEqual(pending.json['status'],'pending')
        self.assertIn('Pending editorial preview',pending.json['html'])
        with self.db() as (_,c):
            c.execute("SELECT to_regclass('onepager_decisions') AS name");self.assertIsNone(c.fetchone()['name'])
        self.submit()
        accepted=client.get(path+'/view?layout=brief')
        self.assertEqual(accepted.status_code,200,accepted.json)
        self.assertEqual(accepted.json['status'],'accepted')
        self.assertEqual(accepted.json['id'],EID)
        self.assertEqual(accepted.json['assignmentId'],AID)
        self.assertIn('Investor-accepted editorial edition',accepted.json['html'])
        self.assertEqual(accepted.headers['Cache-Control'],'private, no-store')
        self.assertNotIn('Content-Disposition',accepted.headers)
        self.assertEqual(client.get(path+'/preview').status_code,409)
        self.assertEqual(client.get(path+'/view?accepted=true').status_code,400)
        self.assertEqual(client.get(path.replace(AID,OTHER)+'/view').status_code,404)
        self.assertFalse(any(commits))
        with self.db() as (_,c):
            c.execute('SELECT state FROM stock_analysis_runs');self.assertEqual(c.fetchone()['state'],self.run['state'])
            c.execute('SELECT result FROM mp_jobs');self.assertEqual(c.fetchone()['result'],self.assignment['result'])
            c.execute('SELECT count(*) AS n FROM onepager_decisions');self.assertEqual(c.fetchone()['n'],1)

    def test_stable_view_keeps_historical_edition_after_successor_acceptance(self):
        self.submit()
        successor={**self.body,'requestId':OTHER,'parentId':EID}
        self.assertEqual(self.client().post(self.url,json=successor).status_code,201)
        record=next(x['record'] for x in self.client().get(self.url).json['editions'] if x['id']==OTHER)
        self.submit({**self.decision,'requestId':OTHER,'editionId':OTHER,'parentId':EID,'selectionDigest':record['selectionDigest']},OTHER)
        response=self.client().get(self.url+'/'+EID+'/view?layout=brief')
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['id'],EID)
        self.assertIn(DID,response.json['html'])
        self.assertNotIn(OTHER,response.json['html'])

    def test_rejected_view_never_substitutes_original_or_another_edition(self):
        self.submit({**self.decision,'action':'reject'})
        r=self.client().get(self.url+'/'+EID+'/view?layout=brief')
        self.assertEqual(r.status_code,409)
        self.assertIn('no other edition was substituted',r.json['error'])
        self.assertNotIn('html',r.json)

    def test_stable_view_rechecks_source_policy_bytes_and_saved_decision(self):
        for mutation in ("UPDATE document_files SET metadata='{\"usage\":\"reference_only\"}'::jsonb",
                         "UPDATE document_files SET file_data='Y2hhbmdlZA=='",
                         'DELETE FROM onepager_accepted_heads'):
            self.setUp();self.submit()
            with self.db(True) as (_,c):c.execute(mutation)
            r=self.client().get(self.url+'/'+EID+'/view?layout=brief')
            self.assertEqual(r.status_code,409,r.json)
            self.assertNotIn('html',r.json)

    def test_recovery_view_renders_complete_saved_corrections_before_and_after_decision(self):
        from tests.unit.test_research_review_brief import fixture
        from onepager_edition_contract import baseline, prepare_edition
        run,assignment,record,corrections,docs,financials=fixture()
        assignment['result']['visualFundamentals']=financials
        body={k:v for k,v in baseline(run,assignment,financials).items() if k!='ticker'}
        body.update(requestId=EID,correctionIds=[x['id'] for x in corrections])
        with self.db(True) as (_,c):
            c.execute('DELETE FROM onepager_editions; DELETE FROM onepager_corrections')
            c.execute('UPDATE stock_analysis_runs SET state=%s::jsonb,sources=%s::jsonb WHERE id=%s',(json.dumps(run['state']),json.dumps(run['sources']),run['id']))
            c.execute('UPDATE mp_jobs SET input=%s::jsonb,result=%s::jsonb WHERE id=%s',(json.dumps(assignment['input']),json.dumps(assignment['result']),AID))
            for row in corrections:
                c.execute('INSERT INTO onepager_corrections VALUES(%s,%s,%s,%s,%s::jsonb)',(row['id'],row['run_id'],row['fingerprint'],row['payload_hash'],json.dumps(row['proposal'])))
        r=self.client().post(self.url,json=body);self.assertEqual(r.status_code,201,r.json)
        path=self.url+'/'+EID+'/view?layout=recovery'
        pending=self.client().get(path);self.assertEqual(pending.status_code,200,pending.json)
        record=self.client().get(self.url).json['editions'][0]['record']
        intent={k:record[k] for k in ('assignmentId','runId','fingerprint','fundamentalsDigest','parentId','correctionIds','selectionDigest')}
        self.submit({**intent,'requestId':DID,'editionId':EID,'action':'accept'})
        accepted=self.client().get(path);self.assertEqual(accepted.status_code,200,accepted.json)
        self.assertIn('Pending editorial preview',pending.json['html'])
        self.assertIn('Investor-accepted editorial edition',accepted.json['html'])
        for r in (pending,accepted):
            self.assertIn('Qualified FY2029 assumption',r.json['html'])
            self.assertIn('Charts not extracted.',r.json['html'])
            self.assertEqual(r.json['id'],EID)
