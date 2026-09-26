import unittest
from datetime import date
from portfolio_heatmap import validate, period_return, create_blueprint
from flask import Flask

class HeatmapTests(unittest.TestCase):
    def test_holdings_require_real_weights(self):
        body={'name':'Fund','asOf':'2026-01-01','holdings':[{'ticker':'abt','weight':5}]}
        self.assertEqual(validate(body)['holdings'][0]['ticker'],'ABT')
        for weight in [None, '', 0, float('nan'), float('inf')]:
            body['holdings'][0]['weight']=weight
            with self.assertRaises(ValueError): validate(body)

    def test_duplicates_and_dates(self):
        row={'ticker':'ABT','weight':5}
        with self.assertRaises(ValueError):validate({'name':'F','asOf':'2026-01-01','holdings':[row,row]})
        with self.assertRaises(ValueError):validate({'name':'F','asOf':'2999-01-01','holdings':[row]})

    def test_period_baselines(self):
        points=[(date(2025,12,31),100),(date(2026,1,2),101),(date(2026,1,30),110),(date(2026,2,2),121)]
        self.assertAlmostEqual(period_return(points,'1d')['changePct'],10)
        self.assertEqual(period_return(points,'1m')['baselineDate'],'2026-01-02')
        self.assertAlmostEqual(period_return(points,'ytd')['changePct'],21)
        self.assertIsNone(period_return(points,'1y')['changePct'])

    def test_missing_and_stale_baseline(self):
        self.assertIsNone(period_return([(date(2026,1,1),100)],'1d')['changePct'])
        self.assertIsNone(period_return([(date(2025,1,1),100),(date(2026,2,2),110)],'1m')['changePct'])

    def test_bad_requests_do_not_fetch_market_data(self):
        app=Flask(__name__); app.register_blueprint(create_blueprint(None));client=app.test_client()
        for body in [{'tickers':['ABT'],'period':[]},{'tickers':['bad ticker']},{'tickers':[]},['bad']]:
            self.assertEqual(client.post('/api/portfolio/heatmap/returns',json=body).status_code,400)

if __name__=='__main__':unittest.main()
