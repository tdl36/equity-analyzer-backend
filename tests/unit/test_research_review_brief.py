import unittest
from types import SimpleNamespace
from unittest.mock import patch
from research_review_brief import render
from tests.unit.test_onepager_edition_evidence import fixture
from onepager_edition_evidence import pending_evidence

class RecoveryBriefTests(unittest.TestCase):
 def test_incomplete_or_forged_context_fails_closed(self):
  a=fixture();p=pending_evidence(*a)
  with self.assertRaisesRegex(ValueError,'explicit pending'):render(a[0],a[1]['input'],preview=p)
  with self.assertRaises(ValueError):render(a[0],a[1]['input'],preview={})
 def test_complete_selected_claims_keep_pending_labels_references_and_escape(self):
  class Claims(dict):
   def get(self,path):return {'statement':'<script>Conditional assumption</script>','reviewOrigin':'pending_editorial','evidence':[{'sourceId':'s','matched':True}]}
  run={'ticker':'TEST','id':'report','sources':[{'id':'s','filename':'source.pdf','originalHash':'hash'}]}
  with patch('research_review_brief.check_render_inputs'),patch('research_review_brief.render_evidence',return_value=(run,Claims())):
   text=render(run,{'since':'2026-01-01','until':'2026-10-11'},preview=SimpleNamespace(edition_id='edition'))
  self.assertIn('Pending investor review',text);self.assertIn('author assumptions, not targets',text)
  self.assertNotIn('<script>',text);self.assertIn('&lt;script&gt;',text);self.assertIn('[1]',text)
