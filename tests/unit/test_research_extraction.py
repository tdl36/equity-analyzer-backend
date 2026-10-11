import copy
import hashlib
import unittest
from research_extraction import frozen_source,PER_SOURCE_LIMIT
from research_source_budget import fit_sources
class ExtractionTests(unittest.TestCase):
    def test_full_returned_text_has_exact_ranges_and_not_full_document_claim(self):
        text='Exact extracted text.\n';doc={'filename':'one.pdf','metadata':{'sourceUrl':'https://example.com'},'textExtraction':{'ocrPages':[1],'limitations':['Local OCR used']}}
        before=copy.deepcopy(doc);r=frozen_source(doc,'a'*64,text,'one')
        self.assertEqual(doc,before);self.assertEqual(r['coverage']['mode'],'full_returned_extraction')
        self.assertEqual(r['coverage']['includedRanges'],[[0,len(text)]]);self.assertEqual(r['coverage']['omittedCharacters'],0)
        self.assertEqual(r['coverage']['fullExtractionHash'],hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(r['textExtraction']['limitations'],['Local OCR used'])
    def test_silent_clipped_file_and_pdf_gap_stop_before_research(self):
        for text in ['',None,'a'*(PER_SOURCE_LIMIT+1),'Opening [... middle of document omitted to fit context ...] closing']:
            with self.subTest(size=len(text) if text else 0),self.assertRaises(ValueError):frozen_source({'filename':'one.txt'},'a'*64,text,'one')
        self.assertEqual(len(frozen_source({'filename':'one.txt'},'a'*64,'a'*PER_SOURCE_LIMIT,'one')['text']),PER_SOURCE_LIMIT)
    def test_format_limitations_do_not_assert_spreadsheet_recalculation(self):
        doc=frozen_source({'filename':'one.docx'},'a'*64,'text','one')
        sheet=frozen_source({'filename':'one.xlsx'},'a'*64,'value','one')
        decoded=frozen_source({'filename':'one.txt'},'a'*64,'lost \ufffd value','one')
        self.assertIn('tracked changes',doc['textExtraction']['limitations'][0])
        self.assertIn('not recalculated',sheet['textExtraction']['limitations'][0])
        self.assertIn('decoding loss',decoded['textExtraction']['limitations'][0])
    def test_pack_budget_replaces_coverage_only_where_text_is_omitted(self):
        pack=[frozen_source({'filename':str(i)+'.txt'},str(i)*64,'text '*n,str(i)) for i,n in enumerate([20000,20000,100])]
        fitted=fit_sources(pack)
        self.assertLessEqual(sum(len(x['text']) for x in fitted),160000)
        self.assertEqual(fitted[2]['coverage']['mode'],'full_returned_extraction')
        self.assertEqual(fitted[0]['coverage']['mode'],'bounded_excerpts')
        self.assertEqual(fitted[0]['coverage']['fullExtractionHash'],pack[0]['extractionHash'])
