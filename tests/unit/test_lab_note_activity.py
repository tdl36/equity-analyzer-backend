import json,unittest,uuid
from datetime import datetime,timezone,timedelta
from flask import Flask
from tests.unit import test_pipeline_lab_note_review as fixtures
from tests.unit import test_stock_analysis_postgres as pg
from lab_note_activity import inventory
from activity_snapshot import assemble,create_blueprint

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class Inventory(unittest.TestCase):
    setUpClass=classmethod(fixtures.Review.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.Review.tearDownClass.__func__)
    def setUp(self):
        with self.db(True) as (_,c):c.execute('DELETE FROM pipeline_lab_note_jobs')
    def seed(self,status='queued',revision=None,accepted=None,days=0):
        ident=str(uuid.uuid4())
        with self.db(True) as (_,c):
            c.execute('INSERT INTO pipeline_lab_note_plans VALUES(%s,%s::jsonb,%s,NOW())',(ident,json.dumps({'ticker':'ABT','context':'PRIVATE SOURCE BODY'}),'a'*64))
            c.execute("INSERT INTO pipeline_lab_note_jobs(note_id,status,created_at,updated_at) VALUES(%s,%s,NOW()-%s*INTERVAL '1 day',NOW()-%s*INTERVAL '1 day')",(ident,status,days,days))
            for rev in range(1,(revision or 0)+1):
                c.execute("INSERT INTO pipeline_lab_note_versions VALUES(%s,%s,%s,'ABT',%s::jsonb,%s,NOW()-%s*INTERVAL '1 day')",(ident,rev,str(uuid.uuid4()),json.dumps({'markdown':'PRIVATE NOTE'}),'b'*64,days))
            if accepted:c.execute("INSERT INTO pipeline_lab_note_acceptances VALUES(%s,%s,'PRIVATE INVESTOR',NOW()-%s*INTERVAL '1 day')",(ident,accepted,days))
        return ident
    def test_review_is_exact_revision_and_reads_do_not_change_jobs(self):
        old=self.seed('draft',2,1,20);accepted=self.seed('draft',1,1);missing=self.seed('draft');queued=self.seed();blocked=self.seed('attention')
        with self.db() as (_,c):rows,truncated=inventory(c)
        found={r['id']:r for r in rows}
        self.assertEqual(found[old]['status'],'needs_review');self.assertEqual(found[accepted]['status'],'complete')
        self.assertEqual(found[missing]['status'],'attention');self.assertEqual(found[queued]['status'],'queued')
        self.assertEqual(found[blocked]['status'],'attention');self.assertEqual(truncated,[])
        self.assertNotIn('PRIVATE',json.dumps(rows,default=str))
        now=datetime.now(timezone.utc);items={r['jobId']:r for r in assemble(rows,{},now)}
        self.assertEqual(items[old]['bucket'],'attention')
        with self.db() as (_,c):
            c.execute('SELECT status FROM pipeline_lab_note_jobs WHERE note_id=%s',(queued,));self.assertEqual(c.fetchone()['status'],'queued')
    def test_dedup_requires_current_matching_parent_and_retains_draft_review(self):
        ident=self.seed('running')
        with self.db() as (_,c):rows,_=inventory(c)
        child=rows[0];now=datetime.now(timezone.utc)
        parent={**child,'id':'parent','source':'mp_jobs','stage':'research_assignment','lab_note_id':ident,'lab_note_status':'running','status':'running','updated_at':now}
        self.assertEqual(len(assemble([parent,child],{},now)),1)
        self.assertEqual(len(assemble([{**parent,'updated_at':now-timedelta(days=2)},child],{},now)),2)
        self.assertEqual(len(assemble([{**parent,'ticker':'CVS'},child],{},now)),2)
        self.assertEqual(len(assemble([{**parent,'lab_note_status':'draft','status':'complete'},{**child,'status':'needs_review','note_status':'draft'}],{},now)),2)
    def test_independent_backlog_limits_keep_active_and_errors(self):
        for _ in range(202):self.seed('draft',1)
        active=self.seed('running');blocked=self.seed('attention')
        with self.db() as (_,c):rows,truncated=inventory(c)
        self.assertIn('Full Lab notes (review)',truncated)
        self.assertEqual(sum(r['status']=='needs_review' for r in rows),200)
        self.assertTrue({active,blocked}<={r['id'] for r in rows})
    def test_missing_dependency_is_reported_without_mutation(self):
        self.seed()
        with self.db(True) as (_,c):c.execute('ALTER TABLE pipeline_lab_note_acceptances RENAME TO unavailable_acceptances')
        try:
            app=Flask('lab-activity');app.register_blueprint(create_blueprint(self.db))
            d=app.test_client().get('/api/activity').json
            self.assertIn('Full Lab notes',d['unavailable'])
        finally:
            with self.db(True) as (_,c):c.execute('ALTER TABLE unavailable_acceptances RENAME TO pipeline_lab_note_acceptances')
