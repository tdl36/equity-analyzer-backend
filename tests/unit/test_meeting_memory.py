import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch
import company_memory
import meeting_memory


class MemoryFreezeTests(unittest.TestCase):
    def setUp(self):
        self.row={'ticker':'ABT','status':'running','input':{'companyMemoryRequested':True,'workerToken':'owner'}}
        self.writes=0
        self.snapshot=company_memory.assemble('ABT',{'revision':4,'created_at':'today','body':{'thesis':'Test cash conversion','evidenceLinks':[{'publisher':'Wells Fargo'}]}})
    @contextmanager
    def db(self,commit=False):yield None,self
    def execute(self,sql,args=()):
        if sql.startswith('UPDATE'):
            self.row['input']=json.loads(args[0]);self.writes+=1
    def fetchone(self):return self.row
    def test_recovery_keeps_original_revision_even_after_case_changes(self):
        with patch('company_memory.load',return_value=self.snapshot) as load:
            first=meeting_memory.freeze(self.db,'job','ABT','owner')
            self.snapshot['entries'][0]['revision']=5
            retry=meeting_memory.freeze(self.db,'job','ABT','owner')
        self.assertEqual(retry['entries'][0]['revision'],4)
        self.assertEqual(self.writes,1);load.assert_called_once()
    def test_wrong_ticker_or_worker_never_reads_company_context(self):
        with patch('company_memory.load') as load:
            for ticker,owner in [('MDT','owner'),('ABT','other')]:
                with self.assertRaises(ValueError):meeting_memory.freeze(self.db,'job',ticker,owner)
        load.assert_not_called();self.assertEqual(self.writes,0)
    def test_old_job_is_not_reinterpreted(self):
        self.row['input']={}
        with patch('company_memory.load') as load:self.assertIsNone(meeting_memory.freeze(self.db,'job','ABT'))
        load.assert_not_called()
    def test_failure_prevents_snapshot_write(self):
        with patch('company_memory.load',side_effect=RuntimeError('private detail')):
            with self.assertRaisesRegex(ValueError,'Retry the meeting pack'):
                meeting_memory.freeze(self.db,'job','ABT','owner')
        self.assertEqual(self.writes,0)
    def test_prompt_keeps_private_attribution_without_promoting_it_to_evidence(self):
        prompt=meeting_memory.instruction(self.snapshot)
        self.assertIn('Wells Fargo',prompt)
        self.assertIn('must never satisfy a source citation',prompt)
        self.assertIn('open questions',prompt)
        self.assertIn('comprehensive business coverage',prompt)
