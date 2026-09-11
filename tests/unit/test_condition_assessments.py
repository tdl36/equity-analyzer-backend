import unittest
from condition_assessments import validate
from research_evidence import source_catalog
class Tests(unittest.TestCase):
 def test_frozen_identity_and_provenance(self):
  quote='Management reported improving operating margins but elevated costs remain.'
  sources=source_catalog([{'filename':'Transcript','extracted_text':quote}])
  baseline={'_investmentCase':{'revision':2,'underweightReviews':[{'id':'w','revision':3,'body':{'title':'Review','reviewConditions':[{'id':'c','trigger':'Margin improves'}]}}]}}
  row={'work_id':'w','condition_id':'c','assessment':'partly_met','reason':'Margins improved; sustainability unresolved.','source_id':sources[0]['id'],'source_excerpt':quote}
  result=validate({'condition_assessments':[row]},baseline,sources)[0]
  self.assertTrue(result['passageMatched']);self.assertEqual(result['workRevision'],3)
  self.assertNotIn('reviewPassed',result)
  for rows in [[row,row],[{**row,'work_id':'other'}],[{**row,'assessment':'verified'}]]:
   with self.assertRaises(ValueError):validate({'condition_assessments':rows},baseline,sources)
  self.assertFalse(validate({'condition_assessments':[{**row,'source_excerpt':'invented quotation not contained anywhere in the original document'}]},baseline,sources)[0]['passageMatched'])
