import unittest
from research_package_rewrite import specification, prompt, preview, validate, source_coverage
from model_registry import MODELS

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.field={'id':'brief','label':'Brief','text':'Draft brief','limit':1600}
        self.context={'id':'thesis','label':'Thesis','text':'Draft investment case','limit':40000}
        self.data={'fields':[self.field],'contextFields':[self.context],'instructions':'Explain the mechanism','writingPolicy':'synthesis-v1'}

    def test_context_is_costed_frozen_and_cannot_be_returned_as_a_revision(self):
        spec=specification(self.data)
        self.assertIn('Return ONLY requested fields',prompt(spec,[],{}))
        model=next(k for k,v in MODELS.items() if 'input' in v and 'output' in v)
        first=preview(spec,[],{},model,'parent')
        changed=specification({**self.data,'contextFields':[{**self.context,'text':'Changed judgment'}]})
        self.assertNotEqual(first['fingerprint'],preview(changed,[],{},model,'parent')['fingerprint'])
        self.assertEqual(first['contextFieldCount'],1)
        plain=specification({k:v for k,v in self.data.items() if k!='contextFields'})
        self.assertGreater(first['estimatedUsd'],preview(plain,[],{},model,'parent')['estimatedUsd'])
        self.assertEqual(validate({'fields':[{'id':'brief','text':'Alternative'}]},spec),{'brief':'Alternative'})
        with self.assertRaises(ValueError):validate({'fields':[{'id':'thesis','text':'Unrequested replacement'}]},spec)

    def test_invalid_context_and_combined_budget_fail_before_generation(self):
        for context in (None,[self.field],[self.context,self.context],[{**self.context,'text':'x'*40001}]):
            with self.subTest(context=context), self.assertRaises(ValueError):specification({**self.data,'contextFields':context})
        fields=[{**self.context,'id':str(i),'text':'x'*32000} for i in range(2)]
        with self.assertRaises(ValueError):specification({**self.data,'fields':fields[:1],'contextFields':fields[1:]})
        legacy=specification({k:v for k,v in self.data.items() if k!='contextFields'})
        self.assertNotIn('contextFields',legacy)

    def test_source_coverage_is_honest_even_without_freeform_warning(self):
        sources=[{'id':'a','filename':'a.pdf','coverage':{'mode':'bounded_excerpts','omittedCharacters':100}},
                 {'id':'b','filename':'b.docx','coverage':{'mode':'full_returned_extraction','omittedCharacters':0},'textExtraction':{'limitations':['Images not extracted.']}},
                 {'id':'c','filename':'legacy.pdf'}]
        rows=source_coverage(sources)
        self.assertEqual([r['status'] for r in rows],['omitted_text','recorded_extraction','unknown'])
        self.assertTrue(all(r['warnings'] for r in rows))
        self.assertNotIn('text',rows[0]);self.assertIn('Images not extracted.',rows[1]['warnings'])
