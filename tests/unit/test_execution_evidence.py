import json
import unittest
from agent_execution import execution_evidence


class EvidenceTests(unittest.TestCase):
    def test_legacy_and_malformed_logs_never_prove_no_execution(self):
        for logs in (None, 'broken', {}, [None, 'text'], [{'message': 'complete'}]):
            result = execution_evidence(logs)
            self.assertFalse(result['enrolled'])
            self.assertEqual(result['providerOutcome'], 'unverified')
            self.assertEqual(result['reservationCount'], 0)

    def test_all_logs_examined_but_identifiers_validated_and_bounded(self):
        ids = ['00000000-0000-4000-8000-%012d' % n for n in range(7)]
        entries = [{'event': 'execution_queued'}] + [
            {'event': 'execution_started', 'executionId': i} for i in ids]
        entries += [{'event': 'execution_started', 'executionId': 'private arbitrary text'}]
        entries += [{'event': 'execution_started', 'executionId': ids[0]}]
        entries += [{'message': 'saved step'}] * 110
        result = execution_evidence(json.dumps(entries))
        self.assertTrue(result['enrolled'])
        self.assertEqual(result['reservationCount'], 9)
        self.assertEqual(result['reservationIds'], ids[:5])
        self.assertTrue(result['reservationIdsTruncated'])
        self.assertNotIn('private arbitrary text', json.dumps(result))
        self.assertEqual(result['providerOutcome'], 'unverified')
