import copy
import unittest
from contextlib import contextmanager
from flask import Flask
from research_amendments import create_blueprint, STAGE

class DecisionTests(unittest.TestCase):
    def setUp(self):
        baseline={'thesis':{'summary':'Original'},'conclusion':'Retained'}
        self.state={'analysis':copy.deepcopy(baseline),'job':{'id':'job','ticker':'DE','stage':STAGE,'status':'awaiting_approval','input':{'baseline':baseline},'result':{'changes':[{'id':'0','path':'thesis.summary','before':'Original','after':'Updated','passageMatched':True,'reviewPassed':True}]}},'failAudit':False}
        state=self.state
        class Cursor:
            def execute(self,sql,args):
                import json
                self.sql=sql
                if sql.startswith('UPDATE portfolio_analyses'): state['analysis']=json.loads(args[0])
                if sql.startswith("UPDATE mp_jobs SET status='applied'"):
                    if state['failAudit']: raise RuntimeError('synthetic audit failure')
                    state['job']['status']='applied';state['job']['result']=json.loads(args[0])
            def fetchone(self):
                return copy.deepcopy(state['job']) if 'mp_jobs' in self.sql else {'analysis':copy.deepcopy(state['analysis'])}
        @contextmanager
        def get_db(**kwargs):
            before=copy.deepcopy(state)
            try: yield None,Cursor()
            except Exception:
                state.clear();state.update(before);raise
        app=Flask(__name__);app.config['TESTING']=True
        app.register_blueprint(create_blueprint(get_db,lambda *a:None,lambda k:'key'))
        self.client=app.test_client()
    def decide(self,ids=None):
        return self.client.post('/api/research/amendment/job/decide',json={'action':'apply','acceptedIds':ids if ids is not None else ['0']})
    def test_apply_saves_selected_edit_and_audit_in_same_transaction(self):
        self.assertEqual(self.decide().status_code,200)
        self.assertEqual(self.state['analysis']['thesis']['summary'],'Updated')
        self.assertEqual(self.state['job']['input']['baseline']['thesis']['summary'],'Original')
        self.assertEqual(self.state['job']['result']['appliedSnapshot'],self.state['analysis'])
    def test_duplicate_decision_is_idempotent(self):
        self.decide(); snapshot=copy.deepcopy(self.state)
        self.assertEqual(self.decide().status_code,200)
        self.assertEqual(snapshot,self.state)
    def test_conflict_preserves_live_analyst_edit(self):
        self.state['analysis']['conclusion']='Analyst changed this'
        self.assertEqual(self.decide().status_code,409)
        self.assertEqual(self.state['analysis']['thesis']['summary'],'Original')
        self.assertEqual(self.state['analysis']['conclusion'],'Analyst changed this')
    def test_audit_failure_rolls_back_thesis_write(self):
        self.state['failAudit']=True
        with self.assertRaises(RuntimeError): self.decide()
        self.assertEqual(self.state['analysis']['thesis']['summary'],'Original')
        self.assertEqual(self.state['job']['status'],'awaiting_approval')
    def test_malformed_decision_rejected(self):
        self.assertEqual(self.decide([{}]).status_code,400)
        self.assertEqual(self.client.post('/api/research/amendment/job/decide',json=[]).status_code,400)

if __name__=='__main__': unittest.main()
