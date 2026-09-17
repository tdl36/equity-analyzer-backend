import unittest
from catalyst_editorial import validate_note,render_note,render_review
class EditorialTests(unittest.TestCase):
 def setUp(self):
  self.records=[dict(filename='one.pdf',page=3,speaker='Bill Brown, CEO',topic='Margins',statement='Costs unchanged.',quote='The oil headwind was already guided last quarter.',uncertainty='Prior flag',interpretation='Prior interpretation')]
  self.note=dict(title='MMM conference',participants='Bill Brown, CEO',blocks=[dict(heading='Protecting earnings',text='The estimate was already guided.',kind='management',refs=[0])])
 def test_shareable_note_does_not_leak_audit(self):
  rendered=render_note(validate_note(self.note,self.records),self.records)
  self.assertEqual(rendered.count('Bill Brown'),1)
  self.assertEqual(rendered.count('one.pdf'),1)
  self.assertIn('[p. 3]',rendered)
  self.assertNotIn('Prior flag',rendered)
  self.assertNotIn('text-matched',rendered)
 def test_review_keeps_flags_and_original(self):
  rendered=render_review(self.records,[],False)
  self.assertIn('Prior flag',rendered)
  self.assertIn('already guided last quarter',rendered)
  self.assertIn('No saved thesis baseline',rendered)
 def test_invalid_refs_fail_closed(self):
  for refs in [[],[4],[-1],[True],['0']]:
   self.note['blocks'][0]['refs']=refs
   with self.assertRaises(ValueError):validate_note(self.note,self.records)
 def test_raw_html_escaped(self):
  self.note['blocks'][0]['text']='<script>bad</script>'
  self.assertNotIn('<script>',render_note(self.note,self.records))
 def test_attribution_required(self):
  self.note['blocks'][0]['kind']='fact-ish'
  with self.assertRaises(ValueError):validate_note(self.note,self.records)
 def test_missing_model_verdicts_prevent_sharing(self):
  import json
  from catalyst_editorial import build_editorial
  def call(prompt,tokens):
   return json.dumps({'checks':[]}) if 'Independently review EVERY' in prompt else json.dumps(self.note)
  with self.assertRaisesRegex(ValueError,'did not pass'):
   build_editorial(self.records,call)
 def test_complete_model_verdicts_produce_two_distinct_views(self):
  import json
  from catalyst_editorial import build_editorial
  def call(prompt,tokens):
   return json.dumps({'checks':[{'index':0,'supported':True},{'index':1,'supported':True}]}) if 'Independently review EVERY' in prompt else json.dumps(self.note)
  result=build_editorial(self.records,call)
  self.assertEqual(set(result),{'pm','comprehensive'})
class ContextReferenceTests(unittest.TestCase):
 def test_small_context_keeps_exact_originals_without_model_call(self):
  import json
  from catalyst_editorial import prepare_context
  context=[{'id':19,'filename':'Broker B.pdf','quote':'Unmodified source quotation.'}]
  self.assertEqual(json.loads(prepare_context(context,lambda *_:self.fail('Unnecessary compression'))),context)
 def test_missing_ids_fail_closed(self):
  from catalyst_editorial import prepare_context
  with self.assertRaisesRegex(ValueError,'lost evidence IDs'):
   prepare_context([{'id':19,'quote':'x'*600}],lambda *_:'{"records":[{"id":0,"summary":"wrong ID"}]}',budget=200)
 def test_compression_preserves_ids_publisher_and_page(self):
  import json
  from catalyst_editorial import prepare_context
  rows=[{'id':19,'filename':'Broker B.pdf','page':2,'speaker':'Broker B','quote':'x'*600}]
  result=json.loads(prepare_context(rows,lambda *_:'{"records":[{"id":19,"summary":"Short faithful notes"}]}',budget=200))
  self.assertEqual(result[0]['id'],19);self.assertEqual(result[0]['filename'],'Broker B.pdf');self.assertEqual(result[0]['page'],2)

if __name__=='__main__':unittest.main()
