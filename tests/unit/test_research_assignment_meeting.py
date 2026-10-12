import copy
import unittest
from research_assignment_plan import plan
from research_assignment_meeting import selection
from research_meeting_handoff import prepare
from tests.unit.test_research_meeting_handoff import fixture


class SharedMeetingTests(unittest.TestCase):
    def test_explicit_options_frozen_and_old_defaults_unchanged(self):
        p={'ticker':'TEST','since':'2026-10-01','until':'2026-10-12'}
        self.assertNotIn('meeting_prep',plan(p)['outputs'])
        p.update(outputs=['meeting_prep','thesis','visual'],meeting={'date':'2026-10-20','type':'management'})
        saved=plan(p);p['meeting']['date']='2026-11-20'
        self.assertEqual(saved['meeting']['date'],'2026-10-20')
        for value in (None,{}, {'date':'bad','type':'management'},{'date':'2026-10-20','type':'unknown'}):
            with self.assertRaises(ValueError):plan({**p,'meeting':value})

    def test_only_supported_questions_with_exact_pack_no_rewrite(self):
        run=fixture();run['sources'][0]['originalHash']='a'*64;before=copy.deepcopy(run)
        body=selection(run,{'date':'2026-10-20','type':'management'})
        self.assertEqual(body['indices'],[0]);self.assertEqual(run,before)
        _,identity,topics=prepare(run,body)
        self.assertEqual(topics[0]['questions'][0]['question'],run['state']['report']['diligence_questions'][0])
        self.assertEqual(identity['expectedSources'],body['expectedSources'])
        run['sources'][0]['originalHash']='b'*64
        with self.assertRaises(ValueError):prepare(run,body)

    def test_no_supported_questions_is_visible_failure(self):
        run=fixture();run['state']['citations']={}
        with self.assertRaisesRegex(ValueError,'No source-matched'):selection(run,{'date':'2026-10-20','type':'management'})
