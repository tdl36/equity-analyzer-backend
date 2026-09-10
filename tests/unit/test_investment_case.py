import unittest
import uuid
from contextlib import contextmanager
import json
from flask import Flask
from investment_case import validate, scenario_bridge, create_blueprint


class CaseTests(unittest.TestCase):
    def test_scenarios_are_decimal_price_returns_not_total_returns(self):
        data=dict(referencePrice='100',period='FY27 adjusted diluted EPS',currency='USD',asOf='2026-09-10',bearEPS='4',bearPE='15',baseEPS='5',basePE='20',bullEPS='6',bullPE='25')
        self.assertEqual(scenario_bridge(data)['bear'],{'impliedPrice':'60.00','priceReturnPct':'-40.00'})
        self.assertEqual(scenario_bridge(data)['base']['priceReturnPct'],'0.00')
        for change in ({'referencePrice':'0'},{'baseEPS':'NaN'},{'bullPE':True},{'bearEPS':'-1'},{'asOf':'yesterday'},{'currency':''}):
            with self.subTest(change=change),self.assertRaises(ValueError):scenario_bridge({**data,**change})

    def test_assumptions_preserve_attribution_and_require_unique_ids(self):
        a=dict(id=str(uuid.uuid4()),claim='Margins recover',evidenceType='interpretation',contrary='Cost trend remains elevated')
        self.assertEqual(validate({'assumptions':[a]})['assumptions'][0]['contrary'],a['contrary'])
        with self.assertRaises(ValueError):validate({'assumptions':[a,a]})
        with self.assertRaises(ValueError):validate({'assumptions':[{**a,'evidenceType':'verified_by_ai'}]})

    def test_revision_conflicts_idempotency_and_immutable_history(self):
        rows=[]
        class Cursor:
            def execute(self,sql,args=()):
                self.rows=[]
                if sql.startswith(('CREATE TABLE','SELECT pg_advisory')):return
                if 'WHERE request_id=' in sql:self.rows=[r for r in rows if r['request_id']==args[0]]
                elif sql.startswith('SELECT'):
                    self.rows=sorted([r for r in rows if r['ticker']==args[0]],key=lambda r:r['revision'],reverse=True)
                elif sql.startswith('INSERT INTO investment_case_versions'):
                    rows.append(dict(ticker=args[0],revision=args[1],request_id=args[2],payload_hash=args[3],body=json.loads(args[4]),created_at='2026-09-10T00:00:00Z'))
                else:raise AssertionError(sql)
            def fetchone(self):return self.rows[0] if self.rows else None
            def fetchall(self):return self.rows
        @contextmanager
        def db(commit=False):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client();url='/api/research/investment-case/ABBV'
        self.assertEqual(client.get(url).json['revision'],0)
        p=dict(requestId=str(uuid.uuid4()),revision=0,body={'thesis':'Original belief'})
        self.assertEqual(client.post(url,json=p).json['revision'],1)
        self.assertTrue(client.post(url,json=p).json['replayed'])
        self.assertEqual(client.post(url,json={**p,'body':{'thesis':'Changed'}}).status_code,409)
        self.assertEqual(client.post(url,json={**p,'requestId':str(uuid.uuid4())}).status_code,409)
        p2=dict(requestId=str(uuid.uuid4()),revision=1,body={'thesis':'Revised belief'})
        self.assertEqual(client.post(url,json=p2).status_code,200)
        self.assertEqual(client.get(url).json['versions'][1]['body']['thesis'],'Original belief')
        self.assertEqual(len(rows),2)
        self.assertEqual(client.get('/api/research/investment-case/PFE').json['revision'],0)
