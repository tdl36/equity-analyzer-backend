import base64
import hashlib
import json
import unittest
from unittest.mock import Mock
from command_thesis_bridge import resolve, match_sources

CID='00000000-0000-4000-8000-000000000001'

class CommandBridgeTests(unittest.TestCase):
    def setUp(self):
        self.data=b'Original evidence, including figures and reported periods.'
        self.source={'filename':'event.pdf','sha256':hashlib.sha256(self.data).hexdigest()}
        self.doc={'filename':'event.pdf','file_data':base64.b64encode(self.data).decode()}
        self.command={'ticker':'UNH','input':{'action':'research_task','payload':{'instruction':'Review new evidence'}},'result':{'topic':'UNH event'}}
        self.activity={'id':'activity-1','output':{'synthesisMarkdown':'Recap','evidenceSnapshot':{'sources':[self.source]}}}

    def cursor(self):
        cur=Mock();cur.fetchone.side_effect=[self.command,self.activity];cur.fetchall.return_value=[self.doc]
        return cur

    def test_exact_input_matches_original(self):
        ready,blocked=match_sources([self.source],[self.doc])
        self.assertEqual(ready,[self.source]);self.assertEqual(blocked,[])

    def test_original_hash_matches_transformed_text_input(self):
        transformed={**self.source,'sha256':hashlib.sha256(b'extracted text').hexdigest(),'originalSha256':self.source['sha256']}
        self.assertEqual(match_sources([transformed],[self.doc])[0],[self.source])

    def test_missing_different_ambiguous_and_corrupt_sources_block(self):
        for docs in ([],[{**self.doc,'file_data':base64.b64encode(b'changed').decode()}],
                     [self.doc,self.doc],[{**self.doc,'file_data':'not base64'}]):
            ready,blocked=match_sources([self.source],docs)
            self.assertFalse(ready);self.assertEqual(len(blocked),1)
        self.assertFalse(match_sources([self.source,self.source],[self.doc])[0])

    def test_latest_exact_topic_and_ticker_are_queried(self):
        cur=self.cursor();result=resolve(cur,CID,'UNH')
        self.assertEqual(result['ready'],[self.source])
        self.assertEqual(cur.execute.call_args_list[1].args[1],('UNH','UNH event'))
        self.assertIn('ORDER BY created_at DESC LIMIT 1',cur.execute.call_args_list[1].args[0])
        self.assertEqual(result['instructions'],'Review new evidence')

    def test_other_ticker_incomplete_and_untracked_recaps_rejected(self):
        with self.assertRaises(ValueError):resolve(self.cursor(),CID,'DE')
        self.activity['output']['synthesisMarkdown']=''
        with self.assertRaises(ValueError):resolve(self.cursor(),CID)
        self.activity['output']={'synthesisMarkdown':'Recap'}
        with self.assertRaises(ValueError):resolve(self.cursor(),CID)

    def test_revision_changes_with_recapped_inputs_or_activity(self):
        first=resolve(self.cursor(),CID)['revision']
        self.activity['id']='activity-2'
        self.assertNotEqual(first,resolve(self.cursor(),CID)['revision'])

    def test_noncommand_or_missing_event_not_guessed(self):
        self.command['input']['action']='another_action'
        with self.assertRaises(ValueError):resolve(self.cursor(),CID)
        self.command['input']['action']='research_task';self.command['result']={}
        with self.assertRaises(ValueError):resolve(self.cursor(),CID)

class CommandBridgeRouteTests(unittest.TestCase):
    setUp=CommandBridgeTests.setUp
    cursor=CommandBridgeTests.cursor
    def client(self, cur):
        from contextlib import contextmanager
        from flask import Flask
        from research_amendments import create_blueprint
        @contextmanager
        def db(commit=False):yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db,Mock(),lambda value:'test-key'))
        return app.test_client()

    def test_context_route_read_only(self):
        cur=self.cursor();cur.fetchone.side_effect=[self.command,self.activity,{'analysis':{'thesis':{'summary':'Saved thesis'}}}]
        response=self.client(cur).get('/api/research/commands/'+CID+'/thesis-context')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['documents']['uploaded'],[self.source])
        self.assertTrue(all('SELECT' in c.args[0] for c in cur.execute.call_args_list))

    def test_stale_command_revision_rejects_before_thread(self):
        from unittest.mock import patch
        cur=self.cursor();cur.fetchone.side_effect=[None,self.command,self.activity]
        with patch('research_amendments.threading.Thread') as thread:
            response=self.client(cur).post('/api/research/amendments/UNH',json={
                'requestId':'00000000-0000-4000-8000-000000000002','filenames':['event.pdf'],
                'commandId':CID,'commandRevision':'stale'})
            self.assertEqual(response.status_code,409);thread.assert_not_called()

    def test_verified_sources_and_parent_link_persist_before_dispatch(self):
        from unittest.mock import patch
        revision=resolve(self.cursor(),CID)['revision']
        cur=self.cursor();cur.fetchone.side_effect=[None,self.command,self.activity,None,{'analysis':{'thesis':{'summary':'Saved thesis'}}}]
        cur.fetchall.side_effect=[[self.doc],[{'filename':'event.pdf'}]]
        with patch('research_amendments.threading.Thread') as thread:
            response=self.client(cur).post('/api/research/amendments/UNH',json={
                'requestId':'00000000-0000-4000-8000-000000000002','filenames':['event.pdf'],
                'commandId':CID,'commandRevision':revision})
            self.assertEqual(response.status_code,202,response.json)
            self.assertEqual(thread.call_args.kwargs['args'][-1],{'event.pdf':self.source['sha256']})
            inserted=next(c for c in cur.execute.call_args_list if 'INSERT INTO mp_jobs' in c.args[0])
            self.assertEqual(json.loads(inserted.args[1][-1])['commandId'],CID)
