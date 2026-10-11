"""Frozen public SEC parser replays. Not expert investment-quality validation."""
import hashlib
import json
from pathlib import Path
import unittest
from stock_visual_fundamentals import annuals

ROOT=Path(__file__).parents[1]/'fixtures'/'visual-history'
class PublicHistoryTests(unittest.TestCase):
    def test_public_five_year_revenue_periods_and_fixture_integrity(self):
        expected={'UNH':[287597,324162,371622,400278,447567],
                  'SYK':[17108,18449,20498,22595,25116],
                  'DE':[44024,52577,61251,51716,45684]}
        for source in json.loads((ROOT/'provenance.json').read_text()):
            with self.subTest(ticker=source['ticker']):
                data=(ROOT/(source['ticker']+'.json')).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),source['fixtureSHA256'])
                rows=annuals(json.loads(data),'2026-10-11')
                self.assertEqual([r['year'] for r in rows],['2021','2022','2023','2024','2025'])
                self.assertEqual([r['revenue']['value']/1e6 for r in rows],expected[source['ticker']])
                for row in rows:
                    for derived,a,b in [('operatingMargin','operatingIncome','revenue'),('freeCashFlow','operatingCashFlow','capex')]:
                        if derived in row:self.assertEqual(row[a]['accession'],row[b]['accession'])
    def test_deere_mixed_vintage_margin_is_an_explicit_gap(self):
        issues=[];rows=annuals(json.loads((ROOT/'DE.json').read_text()),'2026-10-11',issues)
        self.assertEqual(len(issues),2)
        for row in rows:
            if row['year'] in ('2023','2024'):self.assertNotIn('operatingMargin',row)
        self.assertTrue(all('different filings' in issue for issue in issues))
