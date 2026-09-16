"""Isolated tests; no application/database imports or provider calls."""
import unittest
from types import SimpleNamespace
import anthropic
import httpx
from summary_lab import ask_with_recovery

class RecoveryTests(unittest.TestCase):
 def error(self,status=200,kind='overloaded_error'):
  return anthropic.APIStatusError('private provider text',response=httpx.Response(status,request=httpx.Request('POST','https://example.test')),body={'error':{'type':kind,'message':'private source and key'}})
 def run_case(self,outcomes):
  attempts=[]; waits=[]; snapshots=[]; state={'progress':'Reading part 1','parts':{'0':{'record':'retained'}}}
  class Fake:
   def __init__(self,**kwargs): attempts.append(kwargs);self.messages=self
   def __enter__(self): return self
   def __exit__(self,*args): pass
   def stream(self,**kwargs): return self
   def __iter__(self):
    value=outcomes.pop(0)
    if isinstance(value,Exception): raise value
    yield object()
   def get_final_message(self):return SimpleNamespace(stop_reason='end_turn',content=[SimpleNamespace(type='text',text='complete result')])
  def call():return ask_with_recovery('test','test','system','source',100,state,lambda s:snapshots.append(dict(s)),client_factory=Fake,sleep=waits.append)
  return call,state,attempts,waits,snapshots
 def test_stream_overload_http_200_recovers(self):
  call,state,attempts,waits,snapshots=self.run_case([self.error(),True]);self.assertEqual(call(),'complete result');self.assertEqual(waits,[10]);self.assertEqual(len(attempts),2);self.assertNotIn('providerIssue',state);self.assertEqual(state['progress'],'Reading part 1');self.assertIn('0',state['parts'])
 def test_overload_exhaustion_is_bounded_and_clear(self):
  call,state,attempts,waits,snapshots=self.run_case([self.error() for _ in range(3)])
  with self.assertRaisesRegex(ValueError,'temporarily overloaded'):call()
  self.assertEqual(waits,[10,30]);self.assertEqual(len(attempts),3);self.assertNotIn('private',str(state))
 def test_auth_is_not_retried(self):
  call,state,attempts,waits,snapshots=self.run_case([self.error(401,'authentication_error')])
  with self.assertRaisesRegex(ValueError,'Settings'):call()
  self.assertEqual(waits,[]);self.assertEqual(len(attempts),1)
 def test_rate_limit_is_retried(self):
  call,state,attempts,waits,snapshots=self.run_case([self.error(429,'rate_limit_error'),True]);call();self.assertEqual(waits,[10])
if __name__=='__main__':unittest.main()
