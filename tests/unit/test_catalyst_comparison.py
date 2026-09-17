import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('catalyst_comparison',ROOT/'catalyst_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CatalystComparisonTests(unittest.TestCase):
    def record(self, **kwargs):
        return dict(topic='Oil costs',statement='Management reiterated previously guided oil costs.',
                    speaker='Management',kind='management_statement',quote='We previously guided oil costs of $150 million.',
                    interpretation='A reiteration alone does not establish a guidance increase.',uncertainty='',question='What changes the cost outlook?',**kwargs)

    def test_long_text_partition_has_no_loss(self):
        text='Full transcript Ω\n'*18000
        self.assertEqual(''.join(m.chunks(text)),text)
        self.assertGreater(len(text),50000)

    def test_rejects_fabricated_quote(self):
        record=self.record();record['quote']='Invented statement that does not appear in the source.'
        accepted,rejected=m.checked_records({'records':[record]},'We previously guided oil costs of $150 million.')
        self.assertFalse(accepted);self.assertEqual(len(rejected),1)

    def test_missing_or_duplicate_review_fails_closed(self):
        import json
        for checks in ([],[{'index':0,'supported':True},{'index':0,'supported':True}]):
            good,bad=m.review_records(lambda *a:json.dumps({'checks':checks}),[self.record()],'source')
            self.assertFalse(good);self.assertEqual(len(bad),1)

    def test_repairs_failed_record_and_rechecks(self):
        import json
        original=self.record();original['statement']='Management raised oil guidance.'
        responses=iter([{'records':[original]}, {'checks':[{'index':0,'supported':False,'reason':'Not new guidance'}]},
                        {'records':[self.record()]},{'checks':[{'index':0,'supported':True}]}])
        good,bad=m.extract_records(self.record()['quote'],lambda *a:json.dumps(next(responses)))
        self.assertEqual(good[0]['statement'],self.record()['statement']);self.assertEqual(bad,[])

    def test_render_keeps_attribution_and_escapes_source(self):
        record=self.record(filename='<script>x</script>.pdf',page=7,segment=1,baselineComparison='')
        result=m.render([record],[],False)
        self.assertIn('No saved thesis baseline',result)
        self.assertIn('Prior machine interpretation:',result)
        self.assertIn('Management',result)
        self.assertNotIn('<script>',result)
        self.assertEqual(result.count('<section data-version='),3)

    def test_failed_parallel_work_never_raises(self):
        with patch.object(m,'generate',side_effect=RuntimeError('private detail')):
            result=m.run_safely([],'',None)
        self.assertEqual(result['status'],'failed');self.assertNotIn('private detail',str(result))

    def test_pdf_partition_pages_restored(self):
        source={'filename':'transcript.pdf','extractionHash':'a','pages':[{'page':1,'text':'full source'}]}
        with tempfile.TemporaryDirectory() as cache, patch('recap_validation.catalog',return_value=([source],[])),patch.object(m,'extract_records',return_value=([self.record()],0)):
            result=m.generate([{'type':'pdf','pageStart':7}], '', None, checkpoint_root=cache)
        self.assertEqual(result['records'][0]['page'],7)

    def test_rejected_interpretation_preserves_exact_source_words(self):
        import json
        responses=iter([{'records':[self.record()]}, {'checks':[{'index':0,'supported':False}]},
                        {'records':[self.record()]}, {'checks':[{'index':0,'supported':False,'reason':'Paraphrase unsupported'}]}])
        good,issues=m.extract_records(self.record()['quote'],lambda *a:json.dumps(next(responses)))
        self.assertEqual(good[0]['statement'],self.record()['quote'])
        self.assertEqual(good[0]['interpretation'],'')
        self.assertIn('original source wording',good[0]['uncertainty'])
        self.assertEqual(len(issues),1)

    def test_brief_selects_ids_without_rewriting_facts(self):
        import json
        records=[self.record() for _ in range(8)]
        original=json.dumps(records)
        ids=m.select_brief(records,lambda *a:json.dumps({'ids':[7,6,5,4,3,2,1]}))
        self.assertEqual(ids,[7,6,5,4,3,2,1]);self.assertEqual(json.dumps(records),original)

    def test_source_spans_supply_original_words_without_model_transcription(self):
        text='Previously guided costs were $150 million, according to management. This remains our assumption.'
        raw=self.record();raw.pop('quote');raw.update(startSpan=1,endSpan=1)
        records,issues=m.checked_records({'records':[raw]},text)
        self.assertEqual(records[0]['quote'],text)
        self.assertFalse(issues)

    def test_document_date_context_reaches_every_page(self):
        source={'filename':'call.pdf','extractionHash':'hash','pages':[{'page':1,'text':'Conference September 15, 2026. Bill Brown.'},{'page':8,'text':'Next year we expect growth.'}]}
        seen=[]
        def extract(text,call,instructions,context):seen.append(context);return [],[]
        with tempfile.TemporaryDirectory() as cache, patch('recap_validation.catalog',return_value=([source],[])),patch.object(m,'extract_records',side_effect=extract):
            m.generate([{'type':'pdf'}],'',None,checkpoint_root=cache)
        self.assertEqual(len(seen),2)
        self.assertTrue(all('September 15, 2026' in c['openingHeader'] for c in seen))

    def test_summary_paths_do_not_import_trial(self):
        for path in ('summary_comparison.py','src/summary-comparison.jsx','summary_bulk.py'):
            self.assertNotIn('catalyst_comparison', (ROOT/path).read_text())

if __name__=='__main__':unittest.main()
