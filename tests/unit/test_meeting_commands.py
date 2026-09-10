import base64
import hashlib
import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch
from flask import Flask
from meeting_command_plan import batch
from meeting_commands import source_documents, job_id, prepare, drain, create_blueprint
from research_commands import plan

CID='00000000-0000-4000-8000-000000000021'
class MeetingTests(unittest.TestCase):
    def request(self,**changes):
        return dict(requestId=CID,tickers=['ABT','MDT'],date='2026-09-08',meetingDate='2026-09-10',days=90,**changes)
    def test_guided_instructions_and_stable_per_company_ids(self):
        bid,commands=batch(self.request())
        self.assertEqual(len(commands),2)
        self.assertNotEqual(commands[0]['id'],commands[1]['id'])
        self.assertEqual(commands,batch(self.request())[1])
        self.assertIn('ABT',commands[0]['payload']['instruction'])
        self.assertNotIn('{ticker}',commands[0]['payload']['instruction'])
        self.assertEqual(commands[0]['payload']['meetingPrep']['meetingDate'],'2026-09-10')
        self.assertIn('missing source types',commands[0]['payload']['instruction'])
        p=commands[0]['payload']
        self.assertEqual(plan({**p,'date':p['until'],'days':90}),p)
    def test_invalid_batches_rejected(self):
        for change in ({'tickers':[]},{'tickers':['ABT','abt']},{'tickers':['ABT']*11},{'tickers':['../ABT']},{'meetingDate':'bad'},{'focuses':['shell']},{'note':'x'*1001},{'days':False}):
            data=self.request();data.update(change)
            with self.assertRaises((ValueError,TypeError)):batch(data)
    def fixture(self):
        raw=b'<html><body>Reported revenue increased 10 percent.</body></html>'
        digest=hashlib.sha256(raw).hexdigest()
        row={'filename':'release.html','file_data':base64.b64encode(raw).decode()}
        bridge={'ready':[{'filename':row['filename'],'sha256':digest}],'blocked':[]}
        return bridge,row
    def test_source_verification_and_html_extraction(self):
        b,r=self.fixture();docs=source_documents(b,[r]);self.assertEqual(docs[0]['extractedText'],'Reported revenue increased 10 percent.')
        self.assertEqual(docs[0]['sha256'],b['ready'][0]['sha256'])
    def test_missing_changed_duplicate_and_excess_sources_block(self):
        b,r=self.fixture()
        for rows in ([],[r,r],[{**r,'file_data':base64.b64encode(b'changed').decode()}]):
            with self.assertRaises(ValueError):source_documents(b,rows)
        for bridge in ({**b,'blocked':[{}]},{**b,'ready':b['ready']*21}):
            with self.assertRaises(ValueError):source_documents(bridge,[r])
    def db(self,cur):
        @contextmanager
        def db(commit=False):yield Mock(),cur
        return db
    def test_existing_prepared_job_is_not_duplicated(self):
        cur=Mock();cur.fetchone.return_value={'id':job_id(CID)}
        self.assertEqual(prepare(self.db(cur),{'id':CID,'ticker':'ABT'}),job_id(CID))
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
    def test_batch_replay_and_conflict(self):
        data=self.request();_,commands=batch(data)
        existing=[{'id':c['id'],'input':{'action':'research_task','meetingBatchId':CID,'payload':c['payload']}} for c in commands]
        cur=Mock();cur.fetchone.return_value={'id':1};cur.fetchall.return_value=existing
        app=Flask(__name__);app.register_blueprint(create_blueprint(self.db(cur),Mock(),lambda:True))
        client=app.test_client()
        self.assertEqual(client.post('/api/research/meeting-commands',json=data).status_code,202)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
        changed={**data,'days':30}
        self.assertEqual(client.post('/api/research/meeting-commands',json=changed).status_code,409)
    def test_all_coverage_validated_before_any_insert(self):
        cur=Mock();cur.fetchone.side_effect=[{'id':1},None]
        app=Flask(__name__);app.register_blueprint(create_blueprint(self.db(cur),Mock(),lambda:True))
        self.assertEqual(app.test_client().post('/api/research/meeting-commands',json=self.request()).status_code,400)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
    def test_retry_only_failed_managed_job(self):
        cur=Mock();cur.fetchone.return_value=None
        app=Flask(__name__);app.register_blueprint(create_blueprint(self.db(cur),Mock(),lambda:True))
        self.assertEqual(app.test_client().post('/api/research/meeting-commands/'+CID+'/retry').status_code,409)
        sql=cur.execute.call_args.args[0]
        self.assertIn("stage='command_meeting'",sql);self.assertIn("status='failed'",sql)
    def test_no_duplicate_worker_when_other_process_owns_lock(self):
        @contextmanager
        def denied(*args):yield False
        run=Mock();db=Mock()
        with patch('amendment_ownership.worker_session',denied):drain(db,run)
        run.assert_not_called();db.assert_not_called()
    def test_changed_frozen_document_fails_before_model(self):
        @contextmanager
        def owned(*args):yield True
        cur=Mock();cur.fetchone.return_value={'id':CID,'status':'queued','input':{'meetingId':1,'docs':[{'id':2,'sha256':'original'}]},'result':{}}
        cur.fetchall.return_value=[];run=Mock()
        with patch('amendment_ownership.worker_session',owned):drain(self.db(cur),run)
        run.assert_not_called();self.assertIn('removed or changed',cur.execute.call_args.args[1][0])
    def test_abandoned_run_reuses_checkpoint(self):
        @contextmanager
        def owned(*args):yield True
        b,r=self.fixture();r={**r,'id':2}
        cp={'stage':'synthesizing','analyses':[{'facts':[]}]} 
        inp={'meetingId':1,'docs':[{'id':2,'sha256':b['ready'][0]['sha256']}],'recoveryAttempts':0}
        cur=Mock();cur.fetchone.return_value={'id':CID,'status':'running','input':inp,'result':cp};cur.fetchall.return_value=[r];run=Mock()
        with patch('amendment_ownership.worker_session',owned):drain(self.db(cur),run)
        self.assertEqual(run.call_args.args[2],cp);self.assertEqual(run.call_args.args[1]['recoveryAttempts'],1)


class ExistingPipelineBridgeTests(unittest.TestCase):
    def pipeline(self):
        # Load only the pure orchestration function: importing app_v3 initializes
        # real application services, so it is deliberately excluded from QA.
        import ast
        import threading
        from pathlib import Path
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute_mp_pipeline_job')
        memory_patch=patch('meeting_memory.freeze',return_value=None)
        memory_patch.start();self.addCleanup(memory_patch.stop)
        evidence_patch=patch('meeting_source_support.excerpts',return_value={'sources':[],'limitations':[]})
        evidence_patch.start();self.addCleanup(evidence_patch.stop)
        env={'threading':threading,'get_db':Mock()}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'app_v3.py','exec'),env)
        env['_run_mp_pipeline_job']=env['_execute_mp_pipeline_job']
        env['_mp_update_job']=Mock()
        env['_mp_analyze_one_doc']=Mock(return_value=({'facts':['source-backed fact']},10))
        env['_mp_synthesize_inline']=Mock(return_value=({'changes':['reported change']},20))
        env['_mp_questions_inline']=Mock(return_value=([{'topic':'Margins','questions':[{'question':'What changed?','source':'release.txt','source_filenames':['release.txt'],'context':'Reported margin change','follow_up_angle':'Which drivers persist?','priority':'high'}]}],30))
        env['_mp_save_results_inline']=Mock(return_value={'id':7,'version':1})
        return env
    def test_managed_pack_uses_existing_analysis_and_save_pipeline(self):
        env=self.pipeline()
        env['_run_mp_pipeline_job'](CID,'fake-key',1,'ABT','Abbott','Health',
            [{'id':2,'filename':'release.txt','extractedText':'Reported facts'}],[],
            'Meeting focus: margins',[],'model',managed=True)
        env['_mp_save_results_inline'].assert_called_once()
        self.assertTrue(env['_mp_save_results_inline'].call_args.kwargs['managed'])
        final=env['_mp_update_job'].call_args.kwargs
        self.assertEqual(final['status'],'done');self.assertEqual(final['result']['questionSetId'],7)
        self.assertEqual(env['_mp_synthesize_inline'].call_args.args[-1],'Meeting focus: margins')
        self.assertEqual(env['_mp_questions_inline'].call_args.args[4]['assignmentContext'],'Meeting focus: margins')
    def test_shared_memory_reaches_question_stage_and_saved_receipt(self):
        env=self.pipeline()
        snapshot={'ticker':'ABT','snapshotHash':'fixture','entries':[]}
        with patch('meeting_memory.freeze',return_value=snapshot):
            env['_run_mp_pipeline_job'](CID,'fake-key',1,'ABT','Abbott','Health',
                [{'id':2,'filename':'release.txt','extractedText':'Reported facts'}],[],
                'Meeting focus: margins',[],'model',managed=True)
        self.assertEqual(env['_mp_questions_inline'].call_args.kwargs['company_context'],snapshot)
        self.assertEqual(env['_mp_update_job'].call_args.kwargs['result']['companyMemory'],snapshot)

    def test_success_keeps_cache_reuse_receipt(self):
        env=self.pipeline();env['_mp_analyze_one_doc'].return_value=({'facts':['existing fact'],'_cacheHit':True},0)
        env['_run_mp_pipeline_job'](CID,'fake-key',1,'ABT','Abbott','Health',
            [{'id':2,'filename':'release.txt','extractedText':'Reported facts'}],[],
            '2026-06-11 through 2026-09-08. Meeting assignment: Focus on reimbursement.',[], 'model',managed=True)
        context=env['_mp_questions_inline'].call_args.args[4]
        self.assertIn('Focus on reimbursement.',context['assignmentContext'])
        self.assertEqual(context['researchWindow'],'2026-06-11 through 2026-09-08')
        self.assertEqual(env['_mp_update_job'].call_args.kwargs['result']['cachedSources'],1)
    def test_failed_questions_resume_without_repeating_document_analysis(self):
        env=self.pipeline();env['_mp_questions_inline'].side_effect=ValueError('test interruption')
        args=(CID,'fake-key',1,'ABT','Abbott','Health',[{'id':2,'filename':'release.txt','extractedText':'facts'}],[],'window',[],'model')
        env['_run_mp_pipeline_job'](*args,managed=True)
        checkpoint=env['_mp_update_job'].call_args.kwargs['result']
        self.assertEqual(checkpoint['failedAt'],'generating')
        env['_mp_questions_inline'].side_effect=None
        env['_run_mp_pipeline_job'](*args,managed=True,resume_from=checkpoint)
        env['_mp_analyze_one_doc'].assert_called_once();env['_mp_synthesize_inline'].assert_called_once()
        self.assertEqual(env['_mp_update_job'].call_args.kwargs['status'],'done')
    def test_saved_meeting_receipt_prevents_duplicate_question_set(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_save_results_inline')
        cur=Mock();cur.fetchone.return_value={'id':7,'version':1}
        @contextmanager
        def db(commit=False):yield None,cur
        env={'get_db':db,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'app_v3.py','exec'),env)
        self.assertEqual(env['_mp_save_results_inline'](1,[],{},0,'model',managed=True),{'id':7,'version':1})
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))


class PackQualityTests(unittest.TestCase):
    def test_unknown_citation_and_missing_followup_fail(self):
        from meeting_commands import validate_pack
        q={'question':'What changed?','context':'Margin fell','source':'release.txt','source_filenames':['release.txt'],'priority':'high','follow_up_angle':'Which costs persist?'}
        docs=[{'filename':'release.txt'}]
        validate_pack([{'topic':'Margins','questions':[q]}],docs)
        for change in ({'source_filenames':['invented.pdf']},{'follow_up_angle':''},{'priority':'urgent'}):
            with self.assertRaises(ValueError):validate_pack([{'topic':'Margins','questions':[{**q,**change}]}],docs)

class MeetingOwnershipTests(unittest.TestCase):
    def test_lost_owner_cannot_publish_progress(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_update_job')
        cur=Mock();cur.rowcount=0
        @contextmanager
        def db(commit=False):yield None,cur
        env={'get_db':db,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'app_v3.py','exec'),env)
        with self.assertRaises(ValueError):env['_mp_update_job'](CID,_owner='old',status='done',result={'topics':[]})
        sql,values=cur.execute.call_args.args
        self.assertIn("input->>'workerToken'",sql);self.assertEqual(values[-1],'old')
    def test_lost_owner_cannot_save_question_set(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_save_results_inline')
        cur=Mock();cur.fetchone.return_value=None
        @contextmanager
        def db(commit=False):yield None,cur
        env={'get_db':db,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'app_v3.py','exec'),env)
        with self.assertRaises(ValueError):env['_mp_save_results_inline'](1,[],{},0,'model',managed=True,job_id=CID,owner='old')
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))

    def test_revision_preserves_prior_version_and_rejects_stale_editor(self):
        topics=[{'topic':'Growth','questions':[dict(question='What changed?',context='Release',source='Release',follow_up_angle='Why?',priority='high',source_filenames=['release.html'])]}]
        cur=Mock();cur.fetchone.side_effect=[{'status':'done','input':{'meetingId':24},'result':{'questionSetId':21,'synthesis':{'old':True}}},{'id':21,'version':1},{'id':22,'version':2},{'company_id':1}]
        cur.fetchall.return_value=[{'filename':'release.html'}]
        app=Flask(__name__);app.register_blueprint(create_blueprint(MeetingTests().db(cur),Mock(),lambda:True))
        response=app.test_client().post('/api/research/meeting-commands/'+job_id(CID)+'/revision',json={'expectedQuestionSetId':21,'reason':'Correct future context','topics':topics})
        self.assertEqual(response.status_code,200)
        calls=cur.execute.call_args_list
        update=next(c for c in calls if 'UPDATE mp_jobs SET result' in c.args[0])
        result=json.loads(update.args[1][0]);self.assertNotIn('synthesis',result);self.assertEqual(result['previousQuestionSetId'],21)
        self.assertFalse(any('DELETE' in c.args[0] or 'UPDATE mp_question_sets' in c.args[0] for c in calls))
        cur.reset_mock();cur.fetchone.side_effect=[{'status':'done','input':{'meetingId':24},'result':{}},{'id':22,'version':2}]
        response=app.test_client().post('/api/research/meeting-commands/'+job_id(CID)+'/revision',json={'expectedQuestionSetId':21,'reason':'stale','topics':topics})
        self.assertEqual(response.status_code,409)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
