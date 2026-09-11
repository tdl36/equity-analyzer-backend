import unittest
from thesis_baseline import draft
class BaselineTests(unittest.TestCase):
 def test_copy_is_deterministic_and_preserves_attribution(self):
  a={'thesis':{'summary':'Summary','pillars':[{'title':'Margin','description':'Prior interpretation'}]},'threats':[{'description':'Cost inflation'}]}
  result=draft('ABT',a)
  self.assertEqual(result,draft('ABT',a));self.assertEqual(result['body']['assumptions'][0]['support'],'Prior interpretation')
  self.assertEqual(result['body']['assumptions'][0]['evidenceType'],'interpretation')
  self.assertIn('Cost inflation',result['body']['changeConditions'])
  self.assertNotIn('assumptions',a)
 def test_no_silent_clipping_or_missing_pillars(self):
  for a in [{},{'thesis':{'pillars':[]}},{'thesis':{'summary':'x'*12001,'pillars':[{'description':'Text'}]}}]:
   with self.assertRaises(ValueError):draft('ABT',a)
