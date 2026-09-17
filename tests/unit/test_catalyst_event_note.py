import unittest,json
from catalyst_event_note import HEADINGS,choose_mode,validate_note,render_event_note,render_event_views,build_event_note
class EventTests(unittest.TestCase):
 def setUp(self):
  self.records=[dict(filename='Goldman.pdf',page=1,quote='The trial met its primary endpoint. Full details remain pending.',statement='Positive topline result.',topic='Readout',speaker='Source')]
  self.note=dict(title='Company — event takeaway',sections=[dict(heading=h,paragraphs=[dict(text='My takeaway is that further evidence is needed.',refs=[0])]) for h in HEADINGS],privateReviewNotes=['Goldman source uncertainty'])
 def test_modes(self):
  self.assertEqual(choose_mode([{'name':'a broker report.pdf'},{'name':'press release.pdf'}]),'event')
  self.assertEqual(choose_mode([{'name':'Earnings call transcript.pdf'}]),'transcript')
  self.assertEqual(choose_mode([{'name':'ELV CFO WFC Conf.pdf'}]),'transcript')
 def test_exact_five_sections(self):
  self.note['sections'].pop()
  with self.assertRaises(ValueError):validate_note(self.note,self.records)
 def test_recommendations_and_broker_labels_rejected(self):
  for text in ('Goldman recommends it.','I recommend BUY.','Our price target is 180.'):
   self.note['sections'][0]['paragraphs'][0]['text']=text
   with self.assertRaises(ValueError):validate_note(self.note,self.records)
 def test_private_provenance_never_in_shareable_html(self):
  public=render_event_note(validate_note(self.note,self.records));self.assertNotIn('Goldman',public);self.assertNotIn('Source',public)
  private=render_event_views(self.note,self.records,[]).split('<section data-version="quick">')[1];self.assertIn('Goldman',private);self.assertIn(self.records[0]['quote'],private)
 def test_missing_review_fails_closed(self):
  def call(prompt,tokens):return json.dumps({'checks':[]}) if 'Independently review EVERY' in prompt else json.dumps(self.note)
  with self.assertRaisesRegex(ValueError,'failed review'):build_event_note(self.records,call)
if __name__=='__main__':unittest.main()
