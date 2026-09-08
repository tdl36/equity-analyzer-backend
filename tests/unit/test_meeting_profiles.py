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
