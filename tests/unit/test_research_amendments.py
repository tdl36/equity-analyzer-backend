import copy
import unittest
from research_amendments import editable_fields, validate_changes, apply_changes
from research_evidence import source_catalog

class AmendmentsTests(unittest.TestCase):
    def setUp(self):
        self.baseline={'thesis':{'summary':'Old judgment','pillars':[{'title':'Pricing','description':'Old margin view','confidence':'High'}]},'conclusion':'Old conclusion','documentHistory':[{'filename':'old.pdf'}]}
        self.sources=source_catalog([{'filename':'release.txt','extracted_text':'Reported operating margin increased to twenty percent in fiscal 2025.'}])
    def change(self,**extra):
        c={'path':'thesis.summary','after':'Updated judgment','reason':'New margin evidence','source_id':self.sources[0]['id'],'source_excerpt':self.sources[0]['text']}
        c.update(extra)
        return c
    def validated(self):
        c=validate_changes({'changes':[self.change()]},self.baseline,self.sources)
        c[0]['reviewPassed']=True
        return c
    def test_selective_apply_preserves_unselected_fields_and_original(self):
        original=copy.deepcopy(self.baseline)
        out=apply_changes(self.baseline,self.baseline,self.validated(),['0'])
        self.assertEqual(out['thesis']['summary'],'Updated judgment')
        self.assertEqual(out['thesis']['pillars'],original['thesis']['pillars'])
        self.assertEqual(out['documentHistory'],original['documentHistory'])
        self.assertEqual(self.baseline,original)
    def test_unrelated_analyst_edit_also_blocks_stale_proposal(self):
        live=copy.deepcopy(self.baseline);live['conclusion']='New analyst judgment'
        with self.assertRaisesRegex(ValueError,'changed'): apply_changes(live,self.baseline,self.validated(),['0'])
    def test_unmatched_evidence_cannot_apply(self):
        c=self.validated();c[0]['passageMatched']=False
        with self.assertRaises(ValueError): apply_changes(self.baseline,self.baseline,c,['0'])
    def test_failed_independent_review_cannot_apply(self):
        c=self.validated();c[0]['reviewPassed']=False
        with self.assertRaises(ValueError): apply_changes(self.baseline,self.baseline,c,['0'])
    def test_bookkeeping_and_array_replacement_forbidden(self):
        for path in ['documentHistory','thesis.pillars','thesis.pillars.0.confidence','__proto__']:
            with self.assertRaises(ValueError): validate_changes({'changes':[self.change(path=path)]},self.baseline,self.sources)
    def test_duplicate_path_rejected(self):
        with self.assertRaises(ValueError): validate_changes({'changes':[self.change(),self.change()]},self.baseline,self.sources)
    def test_unknown_or_repeated_ids_rejected(self):
        for ids in [['invented'],['0','0'],[]]:
            with self.assertRaises(ValueError): apply_changes(self.baseline,self.baseline,self.validated(),ids)
    def test_no_change_is_valid_not_a_synthetic_edit(self):
        self.assertEqual(validate_changes({'changes':[]},self.baseline,self.sources),[])
        self.assertEqual(validate_changes({'changes':[self.change(after='Old judgment')]},self.baseline,self.sources),[])
    def test_invented_quotation_stays_blocked(self):
        c=validate_changes({'changes':[self.change(source_excerpt='Completely invented quotation that is not in the release.')]},self.baseline,self.sources)
        self.assertFalse(c[0]['passageMatched'])
    def test_malformed_model_output_fails_closed(self):
        for raw in [{},{'changes':'wrong'},{'changes':[self.change(after='')]},{'changes':[None]}]:
            with self.assertRaises(ValueError): validate_changes(raw,self.baseline,self.sources)

if __name__=='__main__': unittest.main()
