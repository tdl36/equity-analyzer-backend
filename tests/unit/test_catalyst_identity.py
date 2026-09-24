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

    def test_phase_2b_3_trial_name_and_new_asset_suffix_deduplicate(self):
        a=self.signal('Merck announces positive Phase 2b/3 BRUNELLO study results for intismeran',ticker='MRK')
        b=self.signal('BRUNELLO Phase 2b/3 trial of intismeran met its primary endpoint',ticker='MRK')
        row={'id':'first','input':a}
        self.assertEqual(possible_duplicate(b,[row]),row)

    def test_similar_nonclinical_headlines_only_match_same_day_and_category(self):
        a=dict(title='ABBV cuts full-year earnings guidance to $10.50 after quarterly results',ticker='ABBV',publishedAt=1789041600,category='guidance')
        b=dict(title='Quarterly results: ABBV lowers annual earnings guidance to $10.50',ticker='ABBV',publishedAt=1789042600,category='guidance')
        row={'id':'first','input':a}
        self.assertEqual(possible_duplicate(b,[row]),row)
        self.assertIsNone(possible_duplicate({**b,'category':'legal'},[row]))
        self.assertIsNone(possible_duplicate({**b,'publishedAt':1789128000},[row]))
