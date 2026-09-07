import unittest
from research_evidence import source_catalog, build_snapshot, quality_status, workspace_payload, research_context

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.docs=source_catalog([{'filename':'release.txt','extracted_text':'Revenue increased by 12 percent for the fiscal year ended December 2025.'}])
        self.sid=self.docs[0]['id']
    def snapshot(self, excerpt, sid=None):
        return build_snapshot({'facts':[{'statement':'Revenue grew 12%.','type':'reported_fact','source_id':sid or self.sid,'source_excerpt':excerpt}]},self.docs)
    def test_matching_passage_retains_identity_without_document_body(self):
        s=self.snapshot(self.docs[0]['text'])
        self.assertEqual(s['claims'][0]['status'],'passage_matched')
        self.assertNotIn('text',s['sources'][0])
        self.assertEqual(s['sources'][0]['id'],self.sid)
    def test_fabricated_quote_and_unknown_source_fail(self):
        for quote,sid in [('Revenue increased by 25 percent for fiscal 2025.',self.sid),(self.docs[0]['text'],'invented')]:
            self.assertEqual(self.snapshot(quote,sid)['claims'][0]['status'],'needs_evidence')
    def test_short_quote_is_not_evidence(self):
        self.assertEqual(self.snapshot('Revenue')['claims'][0]['status'],'needs_evidence')
    def test_filename_is_not_source_identity(self):
        changed=source_catalog([{'filename':'release.txt','extracted_text':'Different period and different financial results.'}])
        self.assertNotEqual(changed[0]['id'],self.sid)
    def test_failed_qc_and_dropped_documents_block_readiness(self):
        s=self.snapshot(self.docs[0]['text'])
        self.assertEqual(quality_status(s,{})['status'],'needs_review')
        self.assertEqual(quality_status(s,{'verdict':'ship','findings':[]},dropped=['omitted.pdf'])['status'],'needs_review')
        self.assertEqual(quality_status(s,{'verdict':'ship','findings':[]})['status'],'checks_passed')
    def test_legacy_never_passes(self):
        p=workspace_payload('DE',[{'id':'legacy','state':{'thesis':['Existing judgment']},'qc':{'verdict':'ship','findings':[]}}])
        self.assertEqual(p['current']['quality']['status'],'needs_review')
        self.assertIsNone(p['prior'])
    def test_claims_need_their_own_source_links(self):
        s=build_snapshot({'thesis':['Unsupported judgment'],'facts':[{'statement':'Supported fact','source_id':self.sid,'source_excerpt':self.docs[0]['text']}]},self.docs)
        self.assertEqual(quality_status(s,{'verdict':'ship','findings':[]})['matchedCount'],1)
        self.assertEqual(quality_status(s,{'verdict':'ship','findings':[]})['status'],'needs_review')
    def test_readiness_keeps_independent_and_arithmetic_findings(self):
        q=quality_status(self.snapshot(self.docs[0]['text']),{'verdict':'revise','findings':[{'severity':'high','issue':'Wrong period'}]}, {'consistency':['Invalid scenario probabilities']})
        self.assertIn('Wrong period',q['issues'])
        self.assertIn('Invalid scenario probabilities',q['issues'])
    def test_malformed_collections_are_not_treated_as_evidence(self):
        s=build_snapshot({'facts':'invalid','thesis':None,'evidence_links':{}},self.docs)
        self.assertEqual(s['claims'],[])
        self.assertEqual(quality_status(s,{'verdict':'ship','findings':'invalid'})['status'],'needs_review')

class ResearchContextTests(unittest.TestCase):
    def test_thesis_without_review_is_preserved_with_both_inventories(self):
        context=research_context({'company':'Deere','analysis':'{"thesis":{"summary":"Analyst judgment"}}'},
            [{'filename':'release.pdf','file_data':'PRIVATE'}],
            {'timestamp':'2026-09-07','manifest':{'DE':[{'filename':'call.pdf','folder':'main','path':'PRIVATE'}]}},'DE')
        self.assertEqual(context['savedThesis']['thesis']['summary'],'Analyst judgment')
        self.assertEqual(len(context['documents']['uploaded']),1)
        self.assertEqual(len(context['documents']['local']),1)
        self.assertNotIn('PRIVATE',str(context))
        self.assertNotIn('current',context)
    def test_no_manifest_is_unknown_not_asserted_empty_folder(self):
        context=research_context()
        self.assertIsNone(context['savedThesis'])
        self.assertIsNone(context['documents']['localUpdatedAt'])
    def test_malformed_saved_analysis_does_not_hide_document_inventory(self):
        context=research_context({'analysis':'invalid'},[{'filename':'report.pdf'}])
        self.assertIsNone(context['savedThesis']['thesis'])
        self.assertEqual(context['documents']['uploaded'][0]['filename'],'report.pdf')

if __name__=='__main__': unittest.main()
