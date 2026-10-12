import copy
import unittest
from unittest.mock import patch
from research_visual_view import current_view,assignment_view

class VisualViewTests(unittest.TestCase):
    def test_frozen_snapshot_and_original_link_no_mutation(self):
        run={'id':'report','ticker':'UNH'};row={'id':'assignment','ticker':'UNH','report_id':'report','input':{'since':'2026-08-01'},'fundamentals':{'asOf':'2026-10-10','retrievedAt':'2026-10-10T12:00:00Z'}}
        before=copy.deepcopy(row)
        with patch('research_visual_view.render',return_value='current') as render:
            result=current_view(run,row)
            self.assertEqual(result['rendererVersion'],'stock-infographic-v20');self.assertEqual(result['html'],'current')
            self.assertEqual(render.call_args.args[1]['_visualFundamentals'],row['fundamentals'])
        self.assertEqual(result['originalUrl'],'/api/research/assignments/assignment/artifact/visual?render=original');self.assertEqual(row,before)
        for changed in ({**row,'ticker':'SYK'},{**row,'report_id':'other'}):
            with self.assertRaises(ValueError):current_view(run,changed)
    def test_ambiguous_parents_are_not_chosen_implicitly(self):
        class C:
            def execute(self,*args):pass
            def fetchone(self):return {'name':'mp_jobs'}
            def fetchall(self):return [{'id':'a'},{'id':'b'}]
        with self.assertRaisesRegex(ValueError,'Multiple'):assignment_view(C(),{'id':'r','ticker':'UNH'})
