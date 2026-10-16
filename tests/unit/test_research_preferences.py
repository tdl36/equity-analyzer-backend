import unittest
from research_preferences import validate,instruction,read,save
from tests.unit import test_stock_analysis_postgres as pg

class PreferenceContractTests(unittest.TestCase):
 def test_style_scope_and_bounds(self):
  self.assertEqual(validate({'rules':[' Plain targets ','Plain targets']})['rules'],['Plain targets'])
  self.assertIn('never override evidence',instruction({'rules':['Plain targets']}))
  self.assertEqual(instruction({}), '')
  for value in [{'rules':['']},{'rules':['x'*501]},{'rules':['x']*21},{'rules':[],'approve':True}]:
   with self.assertRaises(ValueError):validate(value)

@unittest.skipUnless((pg.PG/'initdb').exists(),'Disposable PostgreSQL unavailable')
class PreferenceStorageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):pg.StockPostgresTests.setUpClass.__func__(cls)
 @classmethod
 def tearDownClass(cls):pg.StockPostgresTests.tearDownClass.__func__(cls)
 def test_replay_conflict_remove_and_persistence(self):
  with self.db(True) as (_,c):
   initial=read(c);first=save(c,{'rules':['No broker names in narrative.'],'fingerprint':initial['fingerprint']})
   self.assertEqual(save(c,{'rules':first['rules'],'fingerprint':initial['fingerprint']}),first)
   with self.assertRaises(ValueError):save(c,{'rules':['Different'],'fingerprint':initial['fingerprint']})
  with self.db(True) as (_,c):
   self.assertEqual(read(c),first)
   cleared=save(c,{'rules':[],'fingerprint':first['fingerprint']});self.assertEqual(cleared,initial)
