import copy
import unittest
from contextlib import contextmanager
from flask import Flask
from research_meeting_handoff import preview,prepare,save,create_blueprint

RID='6977b8d3-e286-5848-b377-159dcbfc0f21'

def fixture():
    return {'id':RID,'ticker':'SYNTH','status':'complete','sources':[{'id':'s1','filename':'synthetic.txt'}],
        'state':{'report':{'diligence_questions':['What changed in pricing?','How will you measure returns?','Unavailable']},
        'citations':{'/diligence_questions/0':{'statement':'What changed in pricing?','review':'supported','passageMatched':True,'evidence':[{'sourceId':'s1','matched':True}]}}}}

def data(run):return {'fingerprint':preview(run)['fingerprint'],'indices':[1,0],'meetingDate':'2026-10-12','meetingType':'management'}

class Cursor:
    def __init__(self,run):self.run=run;self.existing=None;self.statements=[];self.row=None
    def execute(self,sql,args=()):
        self.statements.append((sql,args))
        if 'FROM stock_analysis_runs' in sql:self.row=self.run
        elif 'FROM research_meeting_handoffs' in sql:self.row=self.existing
        elif 'INSERT INTO mp_companies' in sql:self.row={'id':1}
        elif 'INSERT INTO mp_meetings' in sql:self.row={'id':2}
        elif 'INSERT INTO mp_question_sets' in sql:self.row={'id':3}
        elif 'INSERT INTO research_meeting_handoffs' in sql:self.existing={'meeting_id':args[2],'question_set_id':args[3]}
    def fetchone(self):return self.row

class HandoffTests(unittest.TestCase):
    def test_exact_wording_and_honest_provenance(self):
        run=fixture();original=copy.deepcopy(run);p=preview(run)
        self.assertEqual([q['index'] for q in p['questions']],[0,1]);self.assertTrue(p['questions'][0]['sourceMatched']);self.assertFalse(p['questions'][1]['sourceMatched'])
        _,_,topics=prepare(run,data(run));qs=topics[0]['questions']
        self.assertEqual([q['question'] for q in qs],run['state']['report']['diligence_questions'][:2])
        self.assertTrue(all(q['priority']=='high' and not q['supporting_quotes'] and not q['source_filenames'] for q in qs))
        self.assertEqual(qs[0]['research_origin']['runId'],RID);self.assertEqual(run,original)
    def test_invalid_or_stale_selection_never_writes(self):
        run=fixture();d=data(run)
        for patch in ({'indices':[]},{'indices':[0,0]},{'indices':[True]},{'indices':[2]},{'indices':[9]},{'meetingDate':'2026-02-30'},{'meetingDate':'20261012'},{'meetingType':'unknown'},{'fingerprint':'old'}):
            c=Cursor(run)
            with self.subTest(patch=patch),self.assertRaises(ValueError):save(c,run,{**d,**patch})
            self.assertEqual(c.statements,[])
        changed=copy.deepcopy(run);changed['state']['report']['diligence_questions'][0]='Changed question?'
        with self.assertRaises(ValueError):prepare(changed,d)
        with self.assertRaises(ValueError):preview({**run,'status':'running'})
    def test_duplicate_reload_and_reordered_selection_reuse(self):
        run=fixture();c=Cursor(run);d=data(run)
        a=save(c,run,d);b=save(c,run,{**d,'indices':[0,1]})
        self.assertEqual(a['meetingId'],b['meetingId']);self.assertFalse(a['reused']);self.assertTrue(b['reused'])
        self.assertEqual(sum('INSERT INTO mp_meetings' in q for q,_ in c.statements),1)
        self.assertEqual(sum('INSERT INTO mp_past_questions' in q for q,_ in c.statements),2)
        self.assertTrue(all("'planned'" in q for q,_ in c.statements if 'INSERT INTO mp_past_questions' in q))
        c.existing={'meeting_id':None,'question_set_id':3}
        with self.assertRaises(ValueError):save(c,run,d)
    def test_date_and_selection_define_distinct_meetings(self):
        run=fixture();d=data(run);ident=prepare(run,d)[0]
        for patch in ({'meetingDate':'2026-10-13'},{'indices':[1]},{'meetingType':'expert'}):self.assertNotEqual(ident,prepare(run,{**d,**patch})[0])
    def test_get_is_read_only_and_post_atomic(self):
        run=fixture();cur=Cursor(run);commits=[]
        @contextmanager
        def db(commit=False):
            commits.append(commit);yield None,cur
        app=Flask(__name__);app.register_blueprint(create_blueprint(db));client=app.test_client();url=f'/api/research/stock-analysis-run/{RID}/meeting-prep'
        self.assertEqual(client.get(url).status_code,200);self.assertEqual(commits,[False]);self.assertEqual(len(cur.statements),1)
        r=client.post(url,json=data(run));self.assertEqual(r.status_code,200);self.assertFalse(r.json['reused']);self.assertEqual(commits,[False,True])
        self.assertTrue(any('CREATE TABLE IF NOT EXISTS' in q for q,_ in cur.statements))
        self.assertTrue(client.post(url,json=data(run)).json['reused'])
        self.assertEqual(client.post(url,json=[]).status_code,400)
        cur.run=None;self.assertEqual(client.get(url).status_code,404)
