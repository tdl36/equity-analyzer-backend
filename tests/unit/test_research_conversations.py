import copy
import json
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
import research_conversations as rc


class Store:
    def __init__(self): self.chats={};self.jobs={};self.result=[]
    @contextmanager
    def db(self, commit=False): yield None,self
    def execute(self, sql, args=()):
        self.result=[]
        if 'pg_advisory' in sql:return
        if sql.startswith('SELECT stage,input,status'):
            row=self.jobs.get(args[0]);self.result=[row] if row else []
        elif sql.startswith('SELECT * FROM content_chats') or sql.startswith('SELECT messages FROM content_chats'):
            row=self.chats.get(args[0]);self.result=[row] if row else []
        elif sql.startswith('SELECT status,input FROM mp_jobs'):
            self.result=[j for j in self.jobs.values() if j['stage']==args[0] and j['input']['conversationId']==args[1]][-1:]
        elif sql.startswith('SELECT name,sector,playbook'):
            self.result=[{'name':'Healthcare analyst','sector':'Healthcare','playbook':{}}] if args[0]=='health' else []
        elif sql.startswith('INSERT INTO content_chats'):
            cid,tk,kind,messages=args;self.chats[cid]={'id':cid,'ticker':tk,'content_type':kind,'messages':json.loads(messages)}
        elif sql.startswith('INSERT INTO mp_jobs'):
            jid,stage,tk,value=args;self.jobs[jid]={'id':jid,'stage':stage,'ticker':tk,'status':'queued','input':json.loads(value)}
        elif sql.startswith("UPDATE mp_jobs SET status='running'"):
            j=self.jobs.get(args[0])
            if j and j['stage']==args[1] and j['status']=='queued':j['status']='running';self.result=[{'id':args[0]}]
        elif sql.startswith('SELECT status FROM mp_jobs'):
            j=self.jobs.get(args[0]);self.result=[j] if j else []
        elif sql.startswith('UPDATE content_chats SET messages'):
            self.chats[args[1]]['messages']=json.loads(args[0])
        elif sql.startswith("UPDATE mp_jobs SET status='completed'"):
            self.jobs[args[1]]['status']='completed'
        elif sql.startswith("UPDATE mp_jobs SET status='cancelled'"):
            for j in self.jobs.values():
                if j['stage']==args[0] and j['input']['conversationId']==args[1] and j['status'] in ('queued','running'):
                    j['status']='cancelled';self.result.append({'id':j['id']})
        elif sql.startswith("UPDATE mp_jobs SET status='failed'"):
            j=self.jobs[args[1]]
            if j['status'] in ('queued','running'):j['status']='failed';j['error']=args[0]
        else: raise AssertionError(sql)
    def fetchone(self):return self.result[0] if self.result else None
    def fetchall(self):return self.result


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.store=Store();self.calls=[]
        self.app=Flask(__name__);self.app.testing=True
        self.bp=rc.create_blueprint(self.store.db,lambda prompt:self.calls.append(prompt) or 'Proposed wording, not applied.')
        self.app.register_blueprint(self.bp);self.client=self.app.test_client()
        self.payload={'ticker':'mdt','contentType':'thesis','content':'Saved thesis text.','message':'Challenge the conclusion.', 'requestId':str(uuid.uuid4()),'conversationId':str(uuid.uuid4()),'analystId':'health'}
        self.thread=patch.object(rc.threading,'Thread');self.mockthread=self.thread.start();self.addCleanup(self.thread.stop)
    def post(self,p=None):return self.client.post('/api/research/conversations/messages',json=p or self.payload)
    def work(self):
        kwargs=self.mockthread.call_args.kwargs;kwargs['target'](*kwargs['args'])
    def test_duplicate_request_starts_one_worker_and_saves_one_message(self):
        self.assertEqual(self.post().status_code,202);self.assertEqual(self.post().status_code,200)
        self.assertEqual(self.mockthread.call_count,1)
        self.assertEqual(len(self.store.chats[self.payload['conversationId']]['messages']),1)
    def test_same_id_different_message_conflicts(self):
        self.post();p={**self.payload,'message':'Different'}
        self.assertEqual(self.post(p).status_code,409);self.assertEqual(self.mockthread.call_count,1)
    def test_concurrent_conversation_reply_rejected(self):
        self.post();p={**self.payload,'requestId':str(uuid.uuid4())}
        self.assertEqual(self.post(p).status_code,409)
    def test_reply_is_saved_without_editing_research(self):
        self.post();self.work();self.assertEqual(self.store.jobs[self.payload['requestId']]['status'],'completed')
        messages=self.store.chats[self.payload['conversationId']]['messages'];self.assertEqual(messages[-1]['role'],'assistant')
        self.assertIn('Healthcare analyst',self.calls[0]);self.assertIn('Saved thesis text.',self.calls[0])
    def test_cancelled_job_does_not_invoke_provider(self):
        self.post();r=self.client.post('/api/research/conversations/'+self.payload['conversationId']+'/stop')
        self.assertEqual(r.json['stopped'],1);self.work();self.assertEqual(self.calls,[])
    def test_cancellation_during_model_call_cannot_deliver_reply(self):
        self.post()
        def call(prompt):
            self.store.jobs[self.payload['requestId']]['status']='cancelled';return 'Late reply'
        other=rc.create_blueprint(self.store.db,call)
        kwargs=self.mockthread.call_args.kwargs;other.run_reply(*kwargs['args'])
        self.assertEqual(len(self.store.chats[self.payload['conversationId']]['messages']),1)
    def test_changed_research_requires_new_conversation(self):
        self.post();self.work();p={**self.payload,'requestId':str(uuid.uuid4()),'content':'New thesis'}
        self.assertEqual(self.post(p).status_code,409)
    def test_wrong_company_cannot_reuse_conversation(self):
        self.post();self.work();p={**self.payload,'requestId':str(uuid.uuid4()),'ticker':'DE'}
        self.assertEqual(self.post(p).status_code,409)
    def test_invalid_or_oversized_payload_never_starts_work(self):
        for field,value in [('content','x'*120001),('message',''),('requestId','bad'),('analystId',42),('contentType','command')]:
            with self.subTest(field=field):self.assertEqual(self.post({**self.payload,field:value}).status_code,400)
        self.assertEqual(self.mockthread.call_count,0)
    def test_model_prompt_preserves_full_context_and_bounds_old_history(self):
        p=rc.validate({**self.payload,'content':'START'+('x'*119990)+'END'})
        text=rc.model_prompt(p,[{'role':'user','content':str(i)} for i in range(30)],{'name':'Test'})
        self.assertIn(p['content'],text);self.assertNotIn('"content": "0"',text);self.assertIn('"content": "10"',text)
