"""Synthetic origin validation, locked reads and durable submission replay."""
import copy
import unittest
from unittest.mock import patch
from research_assignment_plan import plan
from research_assignment_origin import verify
from onepager_edition_contract import baseline
from tests.unit.test_onepager_corrections import fixture
from tests.unit.test_onepager_edition_contract import AID

class OriginTests(unittest.TestCase):
    def setUp(self):
        self.run=fixture()[0]
        self.assignment={'id':AID,'ticker':self.run['ticker'],'input':{'ticker':self.run['ticker']},'result':{'reportId':self.run['id'],'visualFundamentals':{'price':100}}}
        self.origin={'kind':'onepager',**baseline(self.run,self.assignment,{'price':100})}
        self.plan={'ticker':self.run['ticker'],'since':'2026-10-01','until':'2026-10-09','outputs':['visual'],'origin':self.origin}
    def test_bounded_origin_and_mac_without_site_packages(self):
        import subprocess,sys,json
        self.assertEqual(plan(self.plan)['origin'],self.origin)
        result=subprocess.run([sys.executable,'-S','-c','from research_assignment_plan import plan; import json; plan(json.loads('+repr(json.dumps(self.plan))+'))'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for changed in [None,{**self.origin,'ticker':'OTHER'},{**self.origin,'kind':'meeting'},{**self.origin,'approved':True},{**self.origin,'fingerprint':'bad'},{**self.origin,'parentId':'bad'}]:
            with self.subTest(changed=changed),self.assertRaises((ValueError,TypeError)):
                plan({**self.plan,'origin':changed})
        with self.assertRaises(ValueError):plan({**self.plan,'outputs':['thesis']})
    def test_locked_current_origin_rejects_changed_report_financials_and_head(self):
        class Cursor:
            def __init__(cur,rows):cur.rows=iter(rows);cur.queries=[]
            def execute(cur,q,args):cur.queries.append(q)
            def fetchone(cur):return next(cur.rows)
        with patch('onepager_decisions.read_lineage',return_value=(None,{})):
            cur=Cursor([self.assignment,self.run]);verify(cur,self.origin)
            self.assertTrue(all('FOR SHARE' in q for q in cur.queries))
            for field in ('report','financials','missing'):
                a=copy.deepcopy(self.assignment);r=copy.deepcopy(self.run)
                if field=='report':r['state']['report']['changed']='new'
                if field=='financials':a['result']['visualFundamentals']['price']=101
                if field=='missing':r=None
                with self.subTest(field=field),self.assertRaises(ValueError):verify(Cursor([a,r]),self.origin)
        with patch('onepager_decisions.read_lineage',return_value=(AID,{})),self.assertRaises(ValueError):verify(Cursor([self.assignment,self.run]),self.origin)
