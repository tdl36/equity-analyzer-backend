import unittest
from meeting_command_plan import batch,meeting_options,profile_instruction
class MeetingProfileTests(unittest.TestCase):
    def test_legacy_payload_keeps_original_shape(self):
        self.assertEqual(meeting_options({'meetingDate':'2026-09-09'}),{'meetingDate':'2026-09-09','focuses':['thesis','earnings','followups'],'note':''})
    def test_format_and_audience_are_frozen_with_collection_instructions(self):
        _,jobs=batch({'requestId':'00000000-0000-4000-8000-000000000021','tickers':['ABT'],'date':'2026-09-08','meetingDate':'2026-09-09','format':'hosted_pm','audience':'generalist'})
        p=jobs[0]['payload']
        self.assertEqual(p['meetingPrep']['format'],'hosted_pm')
        self.assertIn('35–45',p['instruction']);self.assertIn('generalist portfolio managers',p['instruction'])
    def test_invalid_profile_rejected(self):
        for fields in ({'format':'bad'},{'audience':'bad'}):
            with self.assertRaises(ValueError):meeting_options({'meetingDate':'2026-09-09',**fields})
    def test_long_formats_cover_durable_debates(self):
        for fmt in ('one_on_one','hosted_pm'):
            self.assertIn('capital allocation',profile_instruction({'format':fmt}))

class ManualMeetingProfileTests(unittest.TestCase):
    def test_manual_prompt_has_shared_counts_without_legacy_conflicts(self):
        from meeting_command_plan import manual_question_prompt
        legacy='Prepare 25-30 sophisticated questions\n- **Prioritized**: high (8-10), medium (10-12)\nPreserve this source context'
        for fmt,count in [('conference','12–15'),('one_on_one','25–30'),('hosted_pm','35–45')]:
            prompt=manual_question_prompt(legacy,{'format':fmt,'audience':'generalist'})
            self.assertIn(count,prompt);self.assertIn('generalist portfolio managers',prompt)
            self.assertIn('Preserve this source context',prompt)
            self.assertNotIn('25-30 sophisticated',prompt);self.assertNotIn('medium (10-12)',prompt)
