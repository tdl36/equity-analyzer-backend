import hashlib
import json
from pathlib import Path
import unittest
from stock_visual_fundamentals import segment_facts
from research_infographic import render

FIXTURES=Path(__file__).parents[1]/'fixtures'/'visual-segments'

class RealSegmentTests(unittest.TestCase):
    def test_filing_fragments_have_stable_identity_and_expected_segments(self):
        manifest=json.loads((FIXTURES/'provenance.json').read_text())
        expected={'SYK':{'MedSurg and Neurotechnology':15647e6,'Orthopaedics':9469e6},
                  'DE':{'Production & Precision Agriculture':17753e6,'Small Agriculture & Turf':10432e6,'Construction & Forestry':11642e6,'Financial Services':6289e6}}
        for ticker,meta in manifest.items():
            raw=(FIXTURES/(ticker+'-2025.xhtml')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['fragmentSha256'])
            rows=segment_facts(raw,meta['start'],meta['end'],meta['cik'])
            self.assertEqual({r['name']:r['revenue'] for r in rows},expected[ticker])
            self.assertEqual(len(rows),len(expected[ticker]))
            self.assertEqual(segment_facts(raw,meta['start'],'2024-12-31',meta['cik']),[])
            self.assertEqual(segment_facts(raw,meta['start'],meta['end'],1),[])
            self.assertEqual(segment_facts(raw.replace(b'iso4217:USD',b'iso4217:EUR'),meta['start'],meta['end'],meta['cik']),[])

    def test_sector_copy_and_art_do_not_leak_between_issuers(self):
        for ticker in ['DE','SYK','OTHER']:
            h=render({'ticker':ticker,'sources':[],'state':{}},{},{'annuals':[]})
            for bad in ['Four businesses. One integrated model.','Optum Financial','Medicare Advantage','earnings recovery justify','Recovery argument']:
                self.assertNotIn(bad,h)
            if ticker!='OTHER':self.assertIn('Conceptual industry illustration',h)
            else:self.assertNotIn('data:image/png',h)

    def test_unh_keeps_its_own_business_context(self):
        h=render({'ticker':'UNH','sources':[],'state':{}},{},{'annuals':[]})
        self.assertIn('Four businesses. One integrated model.',h)
        self.assertIn('Optum Financial',h)

    def test_conflicting_aliases_are_omitted_instead_of_picking_a_number(self):
        meta=json.loads((FIXTURES/'provenance.json').read_text())['SYK']
        raw=(FIXTURES/'SYK-2025.xhtml').read_bytes().replace(b'>9,469<',b'>9,470<',1)
        rows=segment_facts(raw,meta['start'],meta['end'],meta['cik'])
        self.assertNotIn('Orthopaedics',[r['name'] for r in rows])
        self.assertIn('MedSurg and Neurotechnology',[r['name'] for r in rows])
