import copy,json,unittest,uuid
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
import research_edits as edits
import research_evidence

SOURCE='The company raised annual revenue guidance to $540 million for fiscal 2027.'
DOC={'filename':'release.txt','file_data':'encoded','file_type':'text/plain','extracted_text':SOURCE}
SID=research_evidence.source_catalog([DOC])[0]['id']

class Store:
    def __init__(self):
        self.jobs={};self.notes={'note-1':{'id':'note-1','ticker':'MDT','version':'1.0','status':'published','note_markdown':'Old paragraph.\n\nUnchanged paragraph.','sources_markdown':'Original sources','charts':[],'metadata':{}}}
        self.reviews={'review-1':{'id':'review-1','ticker':'MDT','mode':'review','state':{'ticker':'MDT','thesis':['Old paragraph.'],'price':100,'rating':'BUY'},'metadata':{}}};self.latest='review-1';self.rows=[]
    @contextmanager
    def db(self,commit=False):
        old=copy.deepcopy((self.jobs,self.notes,self.reviews,self.latest))
        try:yield None,self
        except Exception:
            self.jobs,self.notes,self.reviews,self.latest=old;raise
    def execute(self,sql,args=()):
        self.rows=[]
        if 'pg_advisory_xact_lock' in sql:return
        if sql.startswith('SELECT stage,ticker,input') or sql.startswith('SELECT * FROM mp_jobs'):
            r=self.jobs.get(args[0]);self.rows=[r] if r else []
        elif sql.startswith('SELECT id FROM mp_jobs'):
            self.rows=[r for r in self.jobs.values() if r['stage']==args[0] and r['ticker']==args[1] and r['input']['targetId']==args[2] and r['status'] in ('queued','running','awaiting_approval')]
        elif sql.startswith('SELECT * FROM research_notes') or sql.startswith('SELECT * FROM investment_reviews'):
            store=self.notes if 'research_notes' in sql else self.reviews;r=store.get(args[0]);self.rows=[r] if r and r['ticker']==args[1] else []
        elif sql.startswith('SELECT id FROM investment_reviews'):self.rows=[{'id':self.latest}]
        elif sql.startswith('SELECT filename,file_data,file_type'):self.rows=[copy.deepcopy(DOC)]
        elif sql.startswith('SELECT name,sector,playbook'):self.rows=[{'name':'Test analyst','sector':'Healthcare','playbook':{}}]
        elif sql.startswith('INSERT INTO mp_jobs'):
            self.jobs[args[0]]={'id':args[0],'stage':args[1],'ticker':args[2],'status':'queued','input':json.loads(args[3]),'result':{}}
        elif sql.startswith("UPDATE mp_jobs SET status='running'"):
            r=self.jobs[args[0]]
            if r['status']=='queued':r['status']='running';self.rows=[{'id':r['id']}]
        elif sql.startswith('UPDATE mp_jobs SET status=%s'):
            status,result,error,jid,stage=args;r=self.jobs[jid]
            if r['status'] in ('queued','running'):r.update(status=status,result=json.loads(result),error=error)
        elif sql.startswith("UPDATE mp_jobs SET status='dismissed'"):self.jobs[args[0]]['status']='dismissed'
        elif sql.startswith("UPDATE mp_jobs SET status='applied'"):self.jobs[args[1]].update(status='applied',result=json.loads(args[0]))
        elif sql.startswith('INSERT INTO research_notes'):
            ident,tk,version,md,sources,changes,charts,metadata=args;self.notes[ident]={'id':ident,'ticker':tk,'version':version,'note_markdown':md,'sources_markdown':sources,'charts':json.loads(charts),'metadata':json.loads(metadata),'status':'draft'}
        elif sql.startswith('INSERT INTO investment_reviews'):
            ident,tk,mode,state,md,html,pdf,qc,changelog,metadata=args;self.reviews[ident]={'id':ident,'ticker':tk,'mode':mode,'state':json.loads(state),'metadata':json.loads(metadata),'review_markdown':md};self.latest=ident
        else:raise AssertionError(sql)
    def fetchone(self):return self.rows[0] if self.rows else None
    def fetchall(self):return self.rows

class EditTests(unittest.TestCase):
    def setUp(self):
        self.db=Store();self.kind='note';self.calls=[]
        def model(prompt,key,tokens):
            self.calls.append(prompt)
            if prompt.startswith('Independently'):return {'checks':[{'id':'0','verdict':'pass','issue':''}]}
            return {'changes':[{'path':'blocks.0' if self.kind=='note' else 'thesis.0','after':SOURCE,'reason':'Correct guidance.','source_id':SID,'source_excerpt':SOURCE}]}
        self.rendered=[]
        def render(tk,state,mode):self.rendered.append(copy.deepcopy(state));return 'New memo','<p>New memo</p>','pdf',{'consistency':[]}
        app=Flask(__name__);app.testing=True;self.bp=edits.create_blueprint(self.db.db,model,lambda _: 'test-key',render);app.register_blueprint(self.bp);self.client=app.test_client()
        self.thread=patch.object(edits.threading,'Thread');self.mock=self.thread.start();self.addCleanup(self.thread.stop)
        self.extract=patch.object(edits.notegen,'extract_file_text',return_value=SOURCE);self.extract.start();self.addCleanup(self.extract.stop)
        self.payload={'requestId':str(uuid.uuid4()),'kind':'note','targetId':'note-1','filenames':['release.txt'],'instruction':'Correct guidance.','analystId':'health'}
    def submit(self,payload=None,ticker='MDT'):return self.client.post('/api/research/edits/'+ticker,json=payload or self.payload)
    def work(self):p=self.mock.call_args.kwargs;p['target'](*p['args'])
    def decide(self,**data):return self.client.post('/api/research/edits/'+self.payload['requestId']+'/decide',json=data or {'action':'apply','acceptedIds':['0']})
    def prepare(self):self.assertEqual(self.submit().status_code,202);self.work();self.assertEqual(self.db.jobs[self.payload['requestId']]['status'],'awaiting_approval')
    def test_note_apply_creates_draft_and_preserves_published_original(self):
        old=copy.deepcopy(self.db.notes['note-1']);self.prepare();r=self.decide();self.assertEqual(r.status_code,200)
        self.assertEqual(self.db.notes['note-1'],old);new=self.db.notes[r.json['createdId']]
        self.assertEqual(new['status'],'draft');self.assertEqual(new['note_markdown'],SOURCE+'\n\nUnchanged paragraph.')
        self.assertIn('release.txt',new['sources_markdown'])
        self.assertEqual(self.decide().json['createdId'],r.json['createdId']);self.assertEqual(len(self.db.notes),2)
    def test_review_creates_new_structured_version_without_changing_model_inputs(self):
        self.kind='review';self.payload.update(kind='review',targetId='review-1');old=copy.deepcopy(self.db.reviews['review-1']);self.prepare();r=self.decide()
        self.assertEqual(r.status_code,200);self.assertEqual(self.db.reviews['review-1'],old)
        new=self.db.reviews[r.json['createdId']];self.assertEqual(new['state']['thesis'],[SOURCE]);self.assertEqual(new['state']['price'],100);self.assertEqual(new['state']['rating'],'BUY')
        self.assertEqual(new['metadata']['readiness']['status'],'needs_review');self.assertEqual(new['metadata']['evidence'],{})
    def test_changed_note_cannot_apply_old_proposal(self):
        self.prepare();self.db.notes['note-1']['note_markdown']='User changed it';self.assertEqual(self.decide().status_code,409);self.assertEqual(len(self.db.notes),1)
    def test_status_change_cannot_apply_old_proposal(self):
        self.prepare();self.db.notes['note-1']['status']='superseded';self.assertEqual(self.decide().status_code,409)
    def test_new_review_blocks_old_version_proposal(self):
        self.kind='review';self.payload.update(kind='review',targetId='review-1');self.prepare();self.db.latest='newer-review';self.assertEqual(self.decide().status_code,409)
    def test_unmatched_or_failed_review_blocks_apply(self):
        self.prepare();c=self.db.jobs[self.payload['requestId']]['result']['changes'][0]
        for key in ('passageMatched','reviewPassed'):
            c[key]=False;self.assertEqual(self.decide().status_code,409);c[key]=True
    def test_duplicate_submission_does_not_start_second_model_job(self):
        self.assertEqual(self.submit().status_code,202);self.assertEqual(self.submit().status_code,200);self.assertEqual(self.mock.call_count,1)
        self.assertEqual(self.submit({**self.payload,'instruction':'Different'}).status_code,409)
    def test_document_id_cannot_be_used_for_other_company(self):self.assertEqual(self.submit(ticker='DE').status_code,404)
    def test_dismiss_before_worker_preserves_original(self):
        self.submit();self.decide(action='dismiss');self.work();self.assertEqual(self.calls,[]);self.assertEqual(len(self.db.notes),1)
    def test_malformed_selection_never_writes(self):
        self.prepare()
        for ids in [[],['missing'],['0','0'],[{}]]:self.assertEqual(self.decide(action='apply',acceptedIds=ids).status_code,409)
        self.assertEqual(len(self.db.notes),1)
    def test_numeric_and_personal_judgment_fields_are_not_model_editable(self):
        fields=edits.editable('review',{'rating':'BUY','price':100,'thesis':['Test'],'kpis':[{'current':5,'note':'Context'}]})
        self.assertEqual(fields,{'thesis.0':'Test','kpis.0.note':'Context'})
        with self.assertRaises(ValueError):edits.validate_changes({'changes':[{'path':'price','after':'120','reason':'x'}]},'review',{'price':100},[])
    def test_fabricated_source_cannot_pass_text_match(self):
        c=edits.validate_changes({'changes':[{'path':'blocks.0','after':'New','reason':'Correction','source_id':SID,'source_excerpt':'Invented quotation that does not appear in the provided source at all.'}]},'note',{'blocks':['Old']},research_evidence.source_catalog([DOC]))
        self.assertFalse(c[0]['passageMatched'])
