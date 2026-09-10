import unittest
from catalyst_identity import possible_duplicate, clinical_context


class IdentityTests(unittest.TestCase):
    def signal(self, title, ticker='ABBV', timestamp=1789041600):
        return dict(title=title, ticker=ticker, publishedAt=timestamp, category='clinical')

    def test_generic_headlines_never_collapse_distinct_trials(self):
        a=self.signal('Positive Phase 3 clinical trial results')
        self.assertIsNone(possible_duplicate(a,[{'input':a}]))

    def test_asset_match_is_only_a_review_candidate(self):
        a=self.signal('Positive Phase 3 atogepant results in menstrual migraine')
        row={'id':'first','input':a}
        b=self.signal('Atogepant met endpoint in Phase 3 LUNA study for migraine')
        self.assertEqual(possible_duplicate(b,[row]),row)
        for change in ({'ticker':'PFE'},{'publishedAt':1789128000},
                       {'title':'Negative Phase 3 atogepant results in migraine'},
                       {'title':'Positive Phase 2 atogepant results in migraine'}):
            self.assertIsNone(possible_duplicate({**b,**change},[row]))

    def test_clinical_requires_medical_context(self):
        self.assertFalse(clinical_context('Phase II drilling results'))
        self.assertFalse(clinical_context('Phase 3 positive results'))
        self.assertTrue(clinical_context('Phase 3 study met primary endpoint'))
