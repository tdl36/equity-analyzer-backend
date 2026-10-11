import unittest
from catalyst_attribution import attribution
from catalyst_watch import candidate
class AttributionTests(unittest.TestCase):
 def test_saved_company_names_and_explicit_symbols_remain_eligible(self):
  for ticker,name,title in [('ABBV','AbbVie Inc.','AbbVie raises annual guidance'),('MSFT','Microsoft Corporation','Microsoft reports data breach'),('UNH','UnitedHealth Group Incorporated','UnitedHealth raises guidance'),('DE','Deere & Company','NYSE: DE raises guidance')]:
   self.assertTrue(attribution(ticker,title,name)['matched'])
 def test_other_issuers_relationships_and_short_symbols_are_not_identity(self):
  for ticker,name,title in [('PFE','Pfizer Inc.','Roche receives FDA approval'),('MSFT','Microsoft Corporation','Microsoft rival cuts guidance'),('A',None,'A company receives FDA approval'),('ON',None,'On quarterly results guidance falls')]:
   self.assertFalse(attribution(ticker,title,name)['matched'])
 def test_missing_name_keeps_signal_reviewable_without_fabricated_resolution(self):
  a={'headline':'Pfizer receives FDA approval','url':'https://example.com/a','datetime':1000,'related':'PFE'}
  self.assertTrue(candidate('PFE',a,900,1100)['requiresReview'])
  self.assertFalse(candidate('PFE',a,900,1100,'Pfizer Inc.')['requiresReview'])
  self.assertEqual(candidate('PFE',a,900,1100,'Pfizer Inc.')['issuerAttribution']['basis'],'saved_company_name_prefix')
