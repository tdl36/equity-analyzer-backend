import copy
import unittest
from processing_proposal_coverage import summarize


class Coverage(unittest.TestCase):
    def frozen(self):
        return {'plan': {'since': '2026-07-01', 'until': '2026-10-01'},
                'sources': [{'documentId': str(i)} for i in range(1,5)],
                'sourceContext': {
                    '1': {'role':'collection','publishedDate':'2026-10-01','sourceKind':'transcript','referencedBySavedThesis':False},
                    '2': {'role':'supplemental','publishedDate':'2024-02-01','sourceKind':'broker-report','referencedBySavedThesis':True},
                    '3': {'role':'supplemental','publishedDate':'2026-10-02','sourceKind':'press-release','referencedBySavedThesis':False},
                    '4': {'role':'collection','publishedDate':None,'sourceKind':'','referencedBySavedThesis':False}}}

    def test_mixed_history_dates_and_hash_references_are_distinct(self):
        frozen=self.frozen();unchanged=copy.deepcopy(frozen);c=summarize(frozen)
        self.assertEqual(c['roles'], {'collection':2,'supplemental':2,'unknown':0})
        self.assertEqual(c['publication'], {'earliest':'2024-02-01','latest':'2026-10-02','known':3,'unknown':1,'beforeWindow':1,'withinWindow':1,'afterWindow':1})
        self.assertEqual(c['thesisReferences'], {'referenced':1,'noReference':3,'unknown':0})
        self.assertEqual(c['incorporation'], 'not_assessed')
        self.assertEqual(c['exhaustiveCoverage'], 'not_assessed')
        self.assertEqual(frozen,unchanged)

    def test_legacy_metadata_is_unknown_not_inferred_from_window(self):
        frozen=self.frozen();frozen.pop('sourceContext');c=summarize(frozen)
        self.assertEqual(c['metadataRecorded'],0)
        self.assertEqual(c['publication']['unknown'],4)
        self.assertIsNone(c['publication']['earliest'])
        self.assertEqual(c['roles']['unknown'],4)
        self.assertEqual(c['thesisReferences']['unknown'],4)
        self.assertEqual(c['thesisReferences']['noReference'],0)

    def test_partial_invalid_metadata_and_extra_rows_are_not_evidence(self):
        frozen=self.frozen()
        frozen['sourceContext']={'1':{'publishedDate':'2026-02-30','role':'unexpected','sourceKind':'unknown-new-kind','referencedBySavedThesis':'true'},'9':self.frozen()['sourceContext']['2']}
        c=summarize(frozen)
        self.assertEqual(c['metadataRecorded'],1)
        self.assertEqual(c['publication']['unknown'],4)
        self.assertEqual(c['kinds']['unknown'],4)
        self.assertEqual(c['thesisReferences']['unknown'],4)

    def test_window_boundaries_and_no_source_means_no_coverage(self):
        frozen=self.frozen();frozen['sources']=[{'documentId':'1'}]
        for stamp in ['2026-07-01','2026-10-01']:
            frozen['sourceContext']['1']['publishedDate']=stamp
            self.assertEqual(summarize(frozen)['publication']['withinWindow'],1)
        frozen['sources']=[];c=summarize(frozen)
        self.assertEqual(c['total'],0);self.assertIsNone(c['publication']['latest'])
        self.assertEqual(c['exhaustiveCoverage'],'not_assessed')
