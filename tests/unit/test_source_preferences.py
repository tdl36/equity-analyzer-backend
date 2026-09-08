import json,sqlite3,unittest
from source_preferences import policy,decision
from source_selection_gate import check
from research_commands import plan
class SourceTests(unittest.TestCase):
    def test_scoped_rules_and_fail_closed_alias(self):
        p={'mode':'preferred','rules':[{'name':'JPMorgan','disposition':'preferred'},{'name':'JPMorgan','ticker':'ABT','disposition':'excluded'}]}
        self.assertEqual(decision(p,'jpmorgan','NVDA','meeting'),'include')
        self.assertEqual(decision(p,'J.P. Morgan','NVDA','meeting'),'review')
        self.assertEqual(decision(p,'JPMorgan','ABT','meeting'),'excluded')
        self.assertEqual(decision(p,None,'ABT','meeting'),'review')
    def test_invalid_and_duplicate_preferences(self):
        for p in ({'mode':'whatever'},{'mode':'auto','rules':[{'name':'X','disposition':'best'}]},{'mode':'auto','rules':[{'name':'X','disposition':'preferred'}]*2}):
            with self.assertRaises(ValueError):policy(p)
    def test_policy_survives_cloud_command_replan(self):
        d=dict(ticker='ABT',instruction='Prepare research',date='2026-09-08',days=7,kind='event',sourcePolicy={'mode':'review','rules':[]})
        p=plan(d);self.assertEqual(plan({**p,'date':p['until'],'days':7}),p)
    def test_gate_requires_exact_document_and_publisher_approval(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.execute('CREATE TABLE refresh_requests(run,config)')
        cfg={'eventId':'command','researchCommand':{'kind':'event','sourcePolicy':{'mode':'review','rules':[]}}}
        db.execute('INSERT INTO refresh_requests VALUES(?,?)',('run',json.dumps(cfg)))
        for candidates in ([],[{'url':'wrong','publisher':'Bank','decision':'include'}],[{'url':'url','publisher':'Bank','decision':'exclude'}]):
            with self.assertRaises(ValueError):check(db,'run','ABT','broker-report','Bank','url',lambda _:dict(candidates=candidates))
        check(db,'run','ABT','broker-report','Bank','url',lambda _:dict(candidates=[{'url':'url','publisher':'Bank','decision':'include'}]))
        check(db,'run','ABT','transcript',None,'url',lambda _:self.fail('Primary source should not need broker approval'))
    def test_preference_save_conflict_and_validation_do_not_write(self):
        from flask import Flask,Blueprint
        from unittest.mock import Mock
        from contextlib import contextmanager
        from source_preferences import create_routes,DEFAULT,revision
        cur=Mock();cur.fetchone.return_value=None
        @contextmanager
        def db(commit=False):yield Mock(),cur
        app=Flask(__name__);bp=Blueprint('prefs_test',__name__);create_routes(bp,db);app.register_blueprint(bp)
        c=app.test_client()
        self.assertEqual(c.get('/api/research/source-preferences').json['policy'],DEFAULT)
        self.assertEqual(c.put('/api/research/source-preferences',json={'revision':'old','policy':DEFAULT}).status_code,409)
        self.assertFalse(any('INSERT' in call.args[0] for call in cur.execute.call_args_list))
        self.assertEqual(c.put('/api/research/source-preferences',json={'revision':revision(DEFAULT),'policy':{'mode':'review','rules':[]}}).status_code,200)
        self.assertTrue(any('INSERT' in call.args[0] for call in cur.execute.call_args_list))
    def test_shortlist_requires_explicit_current_revision(self):
        from flask import Flask,Blueprint
        from unittest.mock import Mock
        from contextlib import contextmanager
        from source_preferences import create_routes,revision
        cid='00000000-0000-4000-8000-000000000001';url='https://research.alpha-sense.com/doc/observed'
        value={'commandId':cid,'ticker':'ABT','candidates':[{'url':url,'publisher':'Bank','title':'Report','reason':'Unique model','decision':'pending'}]}
        cur=Mock();cur.fetchone.side_effect=[{'input':{'payload':{'ticker':'ABT'}}},{'value':value}]*2
        @contextmanager
        def db(commit=False):yield Mock(),cur
        app=Flask(__name__);bp=Blueprint('shortlist_test',__name__);create_routes(bp,db);app.register_blueprint(bp)
        c=app.test_client();path='/api/research/commands/'+cid+'/source-shortlist'
        self.assertEqual(c.put(path,json={'revision':'old','choices':{url:'include'}}).status_code,409)
        r=c.put(path,json={'revision':revision(value),'choices':{url:'include'}})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json['candidates'][0]['decision'],'include')
    def test_subsector_stock_and_named_analyst_precedence(self):
        from source_preferences import resolve
        p={'mode':'preferred','subsectors':{'UNH':'Healthcare services','ABT':'Medtech'},'rules':[
            {'name':'Wells Fargo','subsector':'Healthcare services','disposition':'preferred'},
            {'name':'Wells Fargo','ticker':'UNH','disposition':'standard'},
            {'name':'Wells Fargo','ticker':'UNH','analyst':'Jane Analyst','disposition':'preferred'}]}
        self.assertEqual(decision(p,'Wells Fargo','ABT','meeting'),'review')
        self.assertEqual(decision(p,'Wells Fargo','UNH','meeting'),'review')
        self.assertEqual(decision(p,'Wells Fargo','UNH','meeting','Jane Analyst','Report byline: Jane Analyst'),'include')
        self.assertEqual(decision(p,'Wells Fargo','UNH','meeting','Different Author','Observed byline'),'review')
        p['subsectors']['HUM']='Healthcare services'
        r=resolve(p,'Wells Fargo','HUM','meeting');self.assertEqual(r['decision'],'include');self.assertEqual(r['rule']['subsector'],'Healthcare services')
    def test_missing_authorship_never_silently_bypasses_analyst_exclusion(self):
        p={'mode':'auto','rules':[{'name':'Bank','disposition':'preferred'},{'name':'Bank','analyst':'A','disposition':'excluded'}]}
        self.assertEqual(decision(p,'Bank','UNH','event'),'review')
        self.assertEqual(decision(p,'Bank','UNH','event','A',None),'review')
        self.assertEqual(decision(p,'Bank','UNH','event','A','Visible author byline'),'excluded')
        self.assertEqual(decision(p,'Bank','UNH','event','B','Visible author byline'),'include')
    def test_classification_and_scope_validation(self):
        for p in ({'mode':'auto','subsectors':{'bad/ticker':'Services'}},{'mode':'auto','rules':[{'name':'Bank','ticker':'UNH','subsector':'Services','disposition':'preferred'}]}):
            with self.assertRaises(ValueError):policy(p)
