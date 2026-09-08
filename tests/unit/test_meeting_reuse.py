import base64,hashlib,unittest
from contextlib import contextmanager
from unittest.mock import Mock
from meeting_reuse import create
class ReuseTests(unittest.TestCase):
    def fixture(self,changed=False):
        raw=b'Original source text';digest=hashlib.sha256(raw).hexdigest()
        inp={'commandId':'old','meetingId':1,'ticker':'MDT','docs':[{'id':2,'filename':'a.txt','sha256':digest}],'pastQuestions':[],'unresolvedQuestions':[]}
        cur=Mock();cur.fetchone.side_effect=[None,{'status':'done','input':inp},{'input':{'payload':{'since':'2026-06-11','until':'2026-09-08'}}},{'company_id':1},{'id':3},{'id':4}]
        cur.fetchall.return_value=[{'id':2,'file_data':base64.b64encode(b'Changed' if changed else raw).decode(),'filename':'a.txt','extracted_text':raw.decode()}]
        @contextmanager
        def db(**kwargs):yield None,cur
        data={'requestId':'00000000-0000-4000-8000-000000000031','meetingDate':'2026-09-09','format':'one_on_one','audience':'generalist'}
        return db,cur,data
    def test_verified_sources_create_separate_meeting_without_collection(self):
        db,cur,data=self.fixture();result=create(db,'base',data)
        self.assertEqual(result['meetingId'],3)
        sql=' '.join(c.args[0] for c in cur.execute.call_args_list)
        self.assertNotIn('UPDATE mp_documents',sql);self.assertNotIn('refresh_requests',sql)
        inserted=[c.args[1] for c in cur.execute.call_args_list if 'INSERT INTO mp_jobs' in c.args[0]]
        self.assertIn('one_on_one',inserted[-1][-1]);self.assertIn('saved_meeting',inserted[0][2])
    def test_changed_source_rejected_before_inserts(self):
        db,cur,data=self.fixture(True)
        with self.assertRaisesRegex(ValueError,'changed or is missing'):create(db,'base',data)
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
    def test_idempotent_replay_does_not_create_another_meeting(self):
        db,cur,data=self.fixture();from meeting_command_plan import meeting_options
        cur.fetchone.side_effect=[{'input':{'reuseIdentity':{'baseJobId':'base','options':meeting_options(data)}}}]
        self.assertTrue(create(db,'base',data)['reused'])
        self.assertFalse(any('INSERT' in c.args[0] for c in cur.execute.call_args_list))
