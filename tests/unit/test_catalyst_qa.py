import json,unittest
from catalyst_qa import build_qa,validate,render_qa

class QATests(unittest.TestCase):
 def setUp(self):
  self.text='Analyst: What is the outlook for next year? CFO: We expect margins to improve, but have not quantified the change.'
  self.row=dict(question='What is next year’s outlook?',answer='Management expects margins to improve but did not quantify the change.',asker='Analyst',respondent='CFO',questionQuote='What is the outlook for next year?',answerQuotes=['We expect margins to improve, but have not quantified the change.'])
  self.row.update(questionSpanIds=[0],answerSpanIds=[0])
  self.data={'hasQA':True,'exchanges':[self.row]}
  self.sources=[{'filename':'test.pdf','pages':[{'page':4,'text':self.text}]}]
 def test_nonmatching_quote_fails(self):
  self.row['answerQuotes']=['Margins will double next year.']
  with self.assertRaises(ValueError):validate(self.data,self.text)
 def test_missing_exchange_verdict_fails(self):
  def call(prompt,tokens):return json.dumps({'complete':True,'checks':[]}) if 'Independently compare' in prompt else json.dumps(self.data)
  with self.assertRaisesRegex(ValueError,'review failed'):build_qa(self.sources,call)
 def test_missing_question_coverage_fails(self):
  def call(prompt,tokens):return json.dumps({'complete':False,'checks':[{'index':0,'supported':True}]}) if 'Independently compare' in prompt else json.dumps(self.data)
  with self.assertRaisesRegex(ValueError,'coverage'):build_qa(self.sources,call)
 def test_passed_exchange_keeps_original_page_and_escapes_html(self):
  def call(prompt,tokens):return json.dumps({'complete':True,'checks':[{'index':0,'supported':True}]}) if 'Independently compare' in prompt else json.dumps(self.data)
  result=build_qa(self.sources,call);self.assertEqual(result['exchanges'][0]['pages'],[4]);result['exchanges'][0]['question']='<script>bad</script>'
  self.assertNotIn('<script>',render_qa(result))
 def test_no_qa_does_not_invent_dialogue(self):
  def call(prompt,tokens):return json.dumps({'complete':True,'checks':[]}) if 'Independently compare' in prompt else json.dumps({'hasQA':False,'exchanges':[]})
  self.assertEqual(build_qa(self.sources,call)['status'],'not_applicable')
if __name__=='__main__':unittest.main()
