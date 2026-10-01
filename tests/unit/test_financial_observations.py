import base64
import copy
import hashlib
import json
import unittest
import uuid
from contextlib import contextmanager
from flask import Flask
from financial_observations import candidates, numbers, selection, resolve
from investment_case import create_blueprint
from operating_model import evaluate
from test_operating_model import fixture

RUN='6df05f68-72dd-4dc6-aa44-d0f85020a43b'
QUOTE='Synthetic consolidated revenue for fiscal 2025 was USD 1.0 billion, compared with 900 million previously.'
RAW=QUOTE.encode()
SHA=hashlib.sha256(RAW).hexdigest()

def research():
    return dict(id=RUN,ticker='SYNTH',status='complete',sources=[dict(id='s',filename='synthetic.txt',text=QUOTE,originalHash=SHA,extractionHash=SHA)],state={'sections':[{'claims':[dict(id='financials:0',basis='reported_fact',review='supported',statement='Synthetic revenue',evidence=[dict(sourceId='s',excerpt=QUOTE,matched=True)])]}]})

def chosen():
    return dict(researchRunId=RUN,claimId='financials:0',sourceId='s',excerptHash=hashlib.sha256(QUOTE.encode()).hexdigest(),token='1.0',unit='billions',fiscalYear=2025,currency='USD',basis='Reported consolidated revenue',locator='Revenue section',confirmed=True)

class ObservationTests(unittest.TestCase):
    def setUp(self):
        outer=self
        self.run=research();self.doc=dict(filename='synthetic.txt',file_data=base64.b64encode(RAW).decode(),metadata={'usage':'research'})
        self.rows=[]
        class Cursor:
            def execute(self,sql,args=()):
                self.result=[]
                if sql.startswith(('CREATE','SELECT pg_advisory')):return
                if 'to_regclass' in sql:self.result=[{'name':'company_research_runs'}]
                elif 'FROM company_research_runs' in sql:self.result=[outer.run] if args[0]==RUN else []
                elif 'FROM document_files' in sql:self.result=[outer.doc] if outer.doc else []
                elif 'WHERE request_id=' in sql:self.result=[r for r in outer.rows if r['request_id']==args[0]]
                elif sql.startswith('SELECT') and 'investment_case_versions' in sql:
                    self.result=sorted([r for r in outer.rows if r['ticker']==args[0] and (len(args)==1 or r['revision']==args[1])],key=lambda r:r['revision'],reverse=True)
                elif sql.startswith('INSERT INTO investment_case_versions'):
                    outer.rows.append(dict(ticker=args[0],revision=args[1],request_id=args[2],payload_hash=args[3],body=json.loads(args[4]),created_at='2026-10-01'))
                else:raise AssertionError(sql)
            def fetchone(self):return copy.deepcopy(self.result[0]) if self.result else None
            def fetchall(self):return copy.deepcopy(self.result)
        self.cur=Cursor()
        @contextmanager
        def db(commit=False):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));self.client=app.test_client()

    def test_numbers_preserve_printed_tokens_and_avoid_percent_partial_matches(self):
        self.assertEqual(numbers('Revenue 1,250.25 and 12% or -10 or 10abc; year 2025'),['1,250.25','2025'])
        self.assertEqual(selection(chosen())['valueMillions'],'1000.0')
        for unit,token in [('units','1000000000'),('thousands','1000000'),('millions','1000'),('billions','1')]:
            self.assertEqual(float(selection({**chosen(),'unit':unit,'token':token})['valueMillions']),1000)

    def test_only_supported_reported_passages_are_candidates(self):
        for field,value in [('basis','broker_estimate'),('review','needs_review')]:
            run=research();run['state']['sections'][0]['claims'][0][field]=value
            self.assertEqual(candidates(run,'SYNTH'),[])
        run=research();run['sources'][0]['text']='A different original.'
        self.assertEqual(candidates(run,'SYNTH'),[])
        with self.assertRaises(ValueError):candidates(research(),'OTHER')
        run=research();run['status']='running'
        with self.assertRaises(ValueError):candidates(run,'SYNTH')

    def test_receipt_is_server_derived_and_semantic_judgment_is_not_verified(self):
        result=resolve({**chosen(),'excerpt':'Forged','checks':['verified'],'originalHash':'fake'},'SYNTH',self.cur)
        self.assertEqual(result['excerpt'],QUOTE);self.assertEqual(result['originalHash'],SHA)
        self.assertIn('not independently verified',result['interpretation'])
        self.assertEqual(result['checks'],['passage_matched','printed_number_matched','unit_conversion_checked'])
        for change in ({'token':'1000'},{'confirmed':False},{'excerptHash':'fake'},{'unit':'percent'},{'fiscalYear':2025.5},{'currency':'US dollars'}):
            with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):resolve({**chosen(),**change},'SYNTH',self.cur)

    def test_source_permission_hash_and_availability_are_rechecked(self):
        for doc in [None,{**self.doc,'metadata':{'nested':{'usage':'reference_only'}}},{**self.doc,'file_data':base64.b64encode(b'Changed original').decode()}]:
            self.doc=doc
            with self.assertRaises(ValueError):resolve(chosen(),'SYNTH',self.cur)

    def test_model_link_cannot_survive_changed_value_currency_or_period(self):
        data={**fixture(),'baseRevenueObservation':chosen()}
        self.assertIn('baseRevenueObservation',evaluate(data))
        for change in ({'baseRevenue':'1001'},{'baseYear':2024},{'currency':'EUR'}):
            with self.assertRaises(ValueError):evaluate({**data,**change})
        # Dropping the link deliberately makes this a manual input again.
        self.assertNotIn('baseRevenueObservation',evaluate({**data,'baseRevenueObservation':None}))

    def test_api_preview_link_and_save_recompute_proof_and_replay_once(self):
        url='/api/research/operating-model/SYNTH/revenue-observation'
        self.assertEqual(len(self.client.get(url+'?runId='+RUN).json['candidates']),1)
        receipt=self.client.post(url,json=chosen());self.assertEqual(receipt.status_code,200)
        self.assertEqual(len(self.rows),0)
        model={**fixture(),'baseRevenueObservation':receipt.json['observation']}
        p=self.client.post('/api/research/operating-model/preview',json={**model,'ticker':'SYNTH'})
        self.assertEqual(p.status_code,200);self.assertEqual(p.json['model']['results']['base']['impliedPrice'],'21.95')
        case='/api/research/investment-case/SYNTH';payload=dict(requestId=str(uuid.uuid4()),revision=0,body={'operatingModel':model})
        saved=self.client.post(case,json=payload);self.assertEqual(saved.status_code,200)
        self.assertEqual(saved.json['body']['operatingModel']['baseRevenueObservation']['excerpt'],QUOTE)
        self.doc=None
        self.assertTrue(self.client.post(case,json=payload).json['replayed'])
        self.assertEqual(len(self.rows),1)
        p=self.client.post(case,json={**payload,'requestId':str(uuid.uuid4()),'revision':1})
        self.assertEqual(p.status_code,409);self.assertEqual(len(self.rows),1)
        self.assertEqual(self.client.post('/api/research/operating-model/preview',json={**model,'ticker':'OTHER'}).status_code,400)

    def test_restore_keeps_historical_receipt_without_claiming_current_availability(self):
        case='/api/research/investment-case/SYNTH'
        self.client.post(case,json=dict(requestId=str(uuid.uuid4()),revision=0,body={'operatingModel':{**fixture(),'baseRevenueObservation':chosen()}}))
        self.client.post(case,json=dict(requestId=str(uuid.uuid4()),revision=1,body={}))
        self.doc=None
        r=self.client.post(case,json=dict(requestId=str(uuid.uuid4()),revision=2,mode='restore',sourceRevision=1))
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['body']['operatingModel']['baseRevenueObservation']['originalHash'],SHA)
        self.assertEqual(r.json['revision'],3);self.assertEqual(len(self.rows),3)

    def test_missing_run_and_malformed_inputs_are_errors_without_writes(self):
        url='/api/research/operating-model/SYNTH/revenue-observation'
        for data in [None,[],{}, {**chosen(),'researchRunId':str(uuid.uuid4())}]:
            self.assertEqual(self.client.post(url,json=data).status_code,400)
        self.assertEqual(self.rows,[])
