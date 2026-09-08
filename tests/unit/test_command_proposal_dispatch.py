import unittest
from contextlib import contextmanager
from unittest.mock import Mock,patch
from flask import Flask,jsonify
from command_proposal_dispatch import identifiers,proposal_input,advance
from research_commands import plan,favorites,DEFAULTS

CID='00000000-0000-4000-8000-000000000001'
class DispatchTests(unittest.TestCase):
    def bridge(self):return {'commandId':CID,'ticker':'UNH','revision':'hash','instructions':'Review source evidence','ready':[{'filename':'release.pdf','sha256':'hash'}],'blocked':[]}
    def db(self):
        cur=Mock();cur.fetchall.return_value=[{'id':CID,'ticker':'UNH'}]
        @contextmanager
        def db(commit=False):yield None,cur
        return db,cur
    def test_ids_and_payload_stable(self):
        self.assertEqual(identifiers(CID),identifiers(CID))
        self.assertNotEqual(*identifiers(CID))
        value=proposal_input(self.bridge())
        self.assertEqual(value['filenames'],['release.pdf']);self.assertEqual(value['commandRevision'],'hash')
    def test_missing_or_excess_sources_never_silently_subset(self):
        for change in ({'blocked':[{'filename':'missing.pdf'}]},{'ready':[]},{'ready':[{'filename':str(i)} for i in range(11)]}):
            with self.assertRaises(ValueError):proposal_input({**self.bridge(),**change})
    def test_disabled_by_default_and_strictly_boolean(self):
        data={'ticker':'UNH','instruction':'Review','kind':'filing','date':'2026-09-08','days':1}
        self.assertFalse(plan(data)['autoProposal'])
        self.assertTrue(plan({**data,'autoProposal':True})['autoProposal'])
        with self.assertRaises(ValueError):plan({**data,'autoProposal':'yes'})
        self.assertFalse(favorites(DEFAULTS)[0]['autoProposal'])
    def test_repeated_advancement_uses_same_proposal_id(self):
        db,cur=self.db();app=Flask(__name__)
        with app.app_context(),patch('command_thesis_bridge.resolve',return_value=self.bridge()):
            submit=Mock(side_effect=lambda tk,p:(jsonify(jobId=p['requestId']),202))
            first=advance(db,submit);second=advance(db,submit)
            self.assertEqual(first,second);self.assertEqual(first[0]['status'],'complete')
            self.assertEqual(submit.call_args_list[0].args[1]['requestId'],submit.call_args_list[1].args[1]['requestId'])
            self.assertIn('NOT EXISTS',cur.execute.call_args_list[0].args[0])
            self.assertIn('NULLS FIRST',cur.execute.call_args_list[0].args[0])
    def test_blocked_bridge_does_not_submit(self):
        db,cur=self.db();submit=Mock()
        with patch('command_thesis_bridge.resolve',return_value={**self.bridge(),'blocked':[{'filename':'missing'}]}):
            result=advance(db,submit)
        submit.assert_not_called();self.assertEqual(result[0]['status'],'blocked')
    def test_rejected_or_wrong_receipt_stays_visible(self):
        app=Flask(__name__)
        with app.app_context(),patch('command_thesis_bridge.resolve',return_value=self.bridge()):
            for response in ((jsonify(error='Another proposal is active'),409),(jsonify(jobId='wrong'),200)):
                db,cur=self.db();result=advance(db,Mock(return_value=response))
                self.assertEqual(result[0]['status'],'blocked')
