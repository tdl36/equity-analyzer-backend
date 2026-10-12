import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from onepager_handoff import prepare,submit,workspace
from thesis_handoff import save,read

A='11111111-1111-4111-8111-111111111111';R='22222222-2222-4222-8222-222222222222'
PACK=dict(assignmentId=A,runId=R,ticker='TEST',fingerprint='frozen',fundamentalsDigest='financials',parentId=None)
CLAIM=dict(path='/summary/one_liner',statement='Proposed wording',basis='interpretation',reason='Synthetic rationale',sourceId='S1',excerpt='Synthetic source passage reviewed separately.',locator='p1')

class HandoffTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.calls=[];self.fail=None;self.changed=False
  self.records={}
  prepare(A,self.root,self.call);save(self.root/'proposal.json',{'claims':[CLAIM]});self.calls=[]
 def call(self,path,body=None):
  self.calls.append((path,copy.deepcopy(body)))
  if body is None:
   if path.endswith('/corrections'):return dict(runId=R,ticker='TEST',fingerprint='frozen',fields=[{'path':CLAIM['path'],'before':'Original'}],sources=[{'id':'S1','filename':'one.pdf','sha256':'abc'}])
   return {**PACK,**({'parentId':A} if self.changed else {})}
  key=body['requestId'];self.records.setdefault(key,copy.deepcopy(body));assert self.records[key]==body
  result={'id':key,'status':'pending'}
  if self.fail and path.endswith(self.fail):self.fail=None;raise ValueError('Simulated lost response')
  return result
 def test_pending_submission_never_approves_and_retains_identities(self):
  first=submit(self.root,self.call);before=copy.deepcopy(self.records)
  second=submit(self.root,self.call)
  self.assertEqual(first,second);self.assertEqual(before,self.records);self.assertEqual(len(self.records),2)
  self.assertTrue(all('/decision' not in p for p,_ in self.calls));self.assertIn('work=assignment%3A'+A,first['reviewUrl'])
 def test_lost_correction_response_replays_exact_body(self):
  self.fail='/corrections'
  with self.assertRaises(ValueError):submit(self.root,self.call)
  prior=copy.deepcopy(self.records);submit(self.root,self.call)
  for key,value in prior.items():self.assertEqual(self.records[key],value)
  self.assertEqual(len(self.records),2)
 def test_lost_edition_response_reconciles_after_parent_change(self):
  self.fail='/onepager-editions'
  with self.assertRaises(ValueError):submit(self.root,self.call)
  self.changed=True;self.calls=[];submit(self.root,self.call)
  self.assertEqual(len(self.calls),1);self.assertIsNotNone(self.calls[0][1])
 def test_stale_before_submission_does_not_write(self):
  self.changed=True
  with self.assertRaisesRegex(ValueError,'parent changed'):submit(self.root,self.call)
  self.assertFalse(self.records)
 def test_changed_submission_content_and_client_review_flags_rejected(self):
  submit(self.root,self.call)
  save(self.root/'proposal.json',{'claims':[{**CLAIM,'statement':'Changed'}]})
  with self.assertRaisesRegex(ValueError,'content changed'):submit(self.root,self.call)
  save(self.root/'proposal.json',{'claims':[{**CLAIM,'review':'supported'}]})
  with self.assertRaisesRegex(ValueError,'fields'):submit(self.root,self.call)
 def test_no_overwrite_preparation_or_watched_workspaces(self):
  with self.assertRaisesRegex(ValueError,'already prepared'):prepare(A,self.root,self.call)
  with self.assertRaisesRegex(ValueError,'private local'):workspace(Path.cwd()/'private-test')
 def test_mismatched_receipt_is_not_saved_as_complete(self):
  def call(path,body=None):
   if body:return {'id':A,'status':'pending'}
   return self.call(path,body)
  with self.assertRaisesRegex(ValueError,'receipt mismatch'):submit(self.root,call)
  self.assertEqual(read(self.root/'submission.json')['receipts'],[])
