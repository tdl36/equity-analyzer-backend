import copy
import json
import unittest
from research_source_context import validate, freeze, attach, note_context, GUIDANCE


def fixture():
    pack=[{'documentId':'1','filename':'one.txt','sha256':'a'*64}]
    context={'1':{'role':'supplemental','publishedDate':'2024-03-01','publisher':'Publisher','sourceKind':'transcript','referencedBySavedThesis':False}}
    expected=[{'id':'1','filename':'one.txt','sha256':'a'*64}]
    return pack,context,expected


class SourceContextTests(unittest.TestCase):
    def test_exact_frozen_identity_and_metadata_are_copied_into_sources(self):
        pack,context,expected=fixture();value=freeze(pack,context)
        context['1']['publishedDate']='2025-01-01'
        self.assertEqual(value['sources'][0]['publishedDate'],'2024-03-01')
        sources=[{'id':'report-id','filename':'one.txt','originalHash':'a'*64,'text':'Exact evidence'}]
        before=copy.deepcopy(sources);result=attach(sources,value,expected)
        self.assertEqual(result[0]['selectionContext']['role'],'supplemental')
        self.assertEqual(result[0]['text'],before[0]['text']);self.assertEqual(sources,before)
        with self.assertRaises(ValueError):attach([{**sources[0],'originalHash':'b'*64}],value,expected)
        with self.assertRaises(ValueError):validate(value,[{**expected[0],'id':'2'}])

    def test_contract_rejects_drift_extra_fields_invalid_dates_and_oversize(self):
        pack,context,expected=fixture();value=freeze(pack,context)
        for field,change in [('role','other'),('publishedDate','2026-02-30'),('publisher','x'*201),('sourceKind','filing'),('referencedBySavedThesis',1),('sha256','b'*64),('id','2'),('filename','other.txt')]:
            bad=copy.deepcopy(value);bad['sources'][0][field]=change
            with self.subTest(field=field),self.assertRaises(ValueError):validate(bad,expected)
        for change in [{'version':True}, {'version':2}, {'privateBody':'hidden'}, {'sources':value['sources']*2}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate({**value,**change},expected)
        with self.assertRaises(ValueError):validate(value,None)
        bad=copy.deepcopy(value);bad['sources'][0]['publisher']='x'*128001
        with self.assertRaisesRegex(ValueError,'128 KB'):validate(bad,expected)

    def test_full_note_context_carries_same_manifest_with_explicit_uncertainty(self):
        pack,context,expected=fixture();value=freeze(pack,context)
        a={'processingEvidenceContext':value,'thesisBaseline':{'analysis':'Frozen thesis'}}
        saved=json.loads(note_context(a,expected))
        self.assertEqual(saved['sourceEvidence'],value);self.assertEqual(saved['sourceContextGuidance'],GUIDANCE)
        self.assertEqual(saved['thesisBaseline'],a['thesisBaseline'])
        a['thesisBaseline']={'analysis':'x'*500000}
        with self.assertRaisesRegex(ValueError,'500 KB'):note_context(a,expected)

    def test_unknowns_are_not_inferred_and_legacy_context_remains_byte_identical(self):
        pack,context,expected=fixture();context['1'].update(publishedDate=None,sourceKind='',referencedBySavedThesis=None)
        value=freeze(pack,context);self.assertIsNone(value['sources'][0]['publishedDate'])
        self.assertIsNone(validate(None,None));sources=[{'old':'unchanged'}];self.assertIs(attach(sources,None,None),sources)
        a={'thesisBaseline':{'z':'frozen','a':[1,2]}}
        self.assertEqual(note_context(a,None),json.dumps(a['thesisBaseline'],default=str))
