import unittest
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
import company_memory as cm
from test_research_conversations import ConversationTests


class MemoryTests(unittest.TestCase):
    def test_preserves_attribution_and_long_case_without_clipping(self):
        case = {'revision': 7, 'created_at': '2026-09-10', 'body': {
            'thesis': 'x'*60001+'END', 'assumptions': [{'evidenceType':'broker_estimate', 'claim':'Forecast'}],
            'evidenceLinks': [{'source':'Wells Fargo', 'before':'old', 'after':'new'}]}}
        snap = cm.assemble('ABT', case, {'updated_at':'yesterday','analysis':{'thesis':{'summary':'Conflicting view'}}})
        self.assertEqual(snap['entries'][0]['body'], case['body'])
        text = cm.render(snap)
        self.assertIn('END', text)
        self.assertIn('Wells Fargo', text)
        self.assertIn('Conflicting view', text)
        self.assertIn('Surface disagreements', text)
        self.assertEqual(snap, cm.assemble('ABT', case, {'updated_at':'yesterday','analysis':{'thesis':{'summary':'Conflicting view'}}}))
        changed = {**case, 'revision':8}
        self.assertNotEqual(snap['snapshotHash'], cm.assemble('ABT',changed)['snapshotHash'])

    def test_no_case_is_explicit_and_ticker_scoped(self):
        self.assertEqual(cm.assemble('ABT')['entries'], [])
        class Cur:
            def execute(self, sql, args=()):
                self.sql=sql
                if 'WHERE ticker=' in sql: self.assertTicker=args
            def fetchone(self):
                if 'to_regclass' in self.sql:return {'name':None}
                return None
        cur=Cur()
        @contextmanager
        def db():yield None,cur
        self.assertEqual(cm.load(db,'abt')['entries'],[])
        self.assertEqual(cur.assertTicker, ('ABT',))
        with self.assertRaises(ValueError):cm.load(db,'ABT/UNH')


class MemoryConversationTests(ConversationTests):
    def setUp(self):
        super().setUp()
        import research_conversations as rc
        self.snapshot=cm.assemble('MDT',{'revision':3,'created_at':'today','body':{'thesis':'Private saved view'}})
        self.app=Flask(__name__);self.app.testing=True
        self.bp=rc.create_blueprint(self.store.db,lambda prompt:self.calls.append(prompt) or 'Reply',lambda ticker:self.snapshot)
        self.app.register_blueprint(self.bp);self.client=self.app.test_client()

    def test_snapshot_is_frozen_in_job_and_used_by_reply(self):
        self.assertEqual(self.post().status_code,202)
        stored=self.store.jobs[self.payload['requestId']]['input']
        self.assertEqual(stored['companyMemory']['snapshotHash'],self.snapshot['snapshotHash'])
        self.work()
        self.assertIn('Private saved view',self.calls[0])
        self.assertEqual(self.post().status_code,200)

    def test_lookup_failure_does_not_enqueue_or_save_message(self):
        import research_conversations as rc
        app=Flask('failure')
        def fail(ticker):raise RuntimeError('private error')
        app.register_blueprint(rc.create_blueprint(self.store.db,lambda _:None,fail))
        r=app.test_client().post('/api/research/conversations/messages',json=self.payload)
        self.assertEqual(r.status_code,503)
        self.assertEqual(self.store.jobs,{})
        self.assertEqual(self.store.chats,{})

    def test_current_question_drives_focused_retrieval_and_freezes_result(self):
        import research_conversations as rc
        lookups=[]
        def focused(ticker,message):
            lookups.append((ticker,message));return self.snapshot
        app=Flask('focused')
        app.register_blueprint(rc.create_blueprint(self.store.db,lambda _: 'Reply',load_focused_memory=focused))
        response=app.test_client().post('/api/research/conversations/messages',json=self.payload)
        self.assertEqual(response.status_code,202)
        self.assertEqual(lookups,[(self.payload['ticker'].upper(),self.payload['message'])])
        stored=self.store.jobs[self.payload['requestId']]['input']['companyMemory']
        self.assertEqual(stored['snapshotHash'],self.snapshot['snapshotHash'])
