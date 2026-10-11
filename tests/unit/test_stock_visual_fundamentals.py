import unittest
from stock_visual_fundamentals import annuals, segment_facts, positive_number
from research_infographic import charts

class FundamentalsTests(unittest.TestCase):
    def fact(self,v,**kw):return dict(start='2025-01-01',end='2025-12-31',val=v,accn='0000000001-26-000001',filed='2026-02-01',form='10-K',**kw)
    def test_annual_periods_units_conflicts_and_arithmetic(self):
        raw={'facts':{'us-gaap':{k:{'units':{'USD':v}} for k,v in {
            'Revenues':[self.fact(100),{**self.fact(200),'filed':'2027-02-01'},{**self.fact(90),'start':'2025-10-01'}],
            'OperatingIncomeLoss':[self.fact(10)],'NetCashProvidedByUsedInOperatingActivities':[self.fact(12)],'PaymentsToAcquirePropertyPlantAndEquipment':[self.fact(4)]}.items()}}}
        r=annuals(raw,'2026-10-10')[0]
        self.assertEqual(r['revenue']['value'],100);self.assertEqual(r['operatingMargin']['value'],10);self.assertEqual(r['freeCashFlow']['value'],8)
        raw['facts']['us-gaap']['OperatingIncomeLoss']['units']['USD'].append(self.fact(20))
        self.assertNotIn('operatingMargin',annuals(raw,'2026-10-10')[0])
        raw['facts']['us-gaap']['PaymentsToAcquirePropertyPlantAndEquipment']['units']['USD'][0]['accn']='different'
        self.assertNotIn('freeCashFlow',annuals(raw,'2026-10-10')[0])
    def test_segment_identity_dimensions_sign_and_conflict(self):
        context='''<xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit><xbrli:context id="a"><xbrli:identifier>0000000001</xbrli:identifier><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate><xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">test:ServicesMember</xbrldi:explicitMember></xbrli:context>'''
        fact='<ix:nonFraction contextRef="a" name="us-gaap:Revenues" unitRef="usd" format="ixt:num-dot-decimal" scale="6">100</ix:nonFraction>'
        r=segment_facts(context+fact,'2025-01-01','2025-12-31',1);self.assertEqual(r[0]['revenue'],100e6)
        self.assertEqual(segment_facts(context+fact,'2025-01-01','2025-12-31',2),[])
        self.assertEqual(segment_facts(context+fact+fact.replace('>100<','>200<'),'2025-01-01','2025-12-31',1),[])
    def test_signed_chart_no_invented_points(self):
        r=[{'year':str(y),'eps':{'value':v}} for y,v in [(2023,-2),(2024,0),(2025,4)]]
        h=charts(r);self.assertIn('-2.0',h);self.assertIn('Comparable history unavailable',h)
        self.assertNotIn('nan',h.lower())

    def test_invalid_market_numbers_are_not_displayed(self):
        for value in [None,False,'bad',float('nan'),float('inf'),-1,0]:
            self.assertIsNone(positive_number(value))
        self.assertEqual(positive_number('24.5'),24.5)

    def test_quote_identity_freshness_and_optional_values(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from stock_visual_fundamentals import quote
        now=datetime(2026,10,10,tzinfo=timezone.utc)
        valid=dict(symbol='UNH',currency='USD',regularMarketTime=now.timestamp()-3600,regularMarketPrice=100,forwardPE=float('nan'))
        with patch('yfinance.Ticker') as ticker:
            ticker.return_value.get_info.return_value=valid
            self.assertIsNone(quote('UNH',now)['forwardPE'])
            for delta in [dict(symbol='OTHER'),dict(currency='EUR'),dict(regularMarketTime=now.timestamp()+60),dict(regularMarketTime=now.timestamp()-8*86400)]:
                ticker.return_value.get_info.return_value=valid|delta
                with self.assertRaises(ValueError):quote('UNH',now)
