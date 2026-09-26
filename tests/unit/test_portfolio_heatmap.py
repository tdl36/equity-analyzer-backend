import unittest
from unittest.mock import patch
import csv
import io
from datetime import date
from portfolio_heatmap import validate, period_return, create_blueprint, parse_index_holdings, index_holdings, _UNIVERSE_CACHE
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

class IndexUniverseTests(unittest.TestCase):
    def fixture(self):
        out=io.StringIO();writer=csv.writer(out)
        writer.writerow(['Example ETF'])
        writer.writerow(['Fund Holdings as of','Sep 24, 2026'])
        writer.writerow(['Ticker','Name','Sector','Asset Class','Market Value','Weight (%)'])
        for i in range(60):
            writer.writerow([f'X{i}','Example company','Health Care','Equity','100','1.50'])
        writer.writerow(['BRK B','Share class','Financials','Equity','0.1','0.00'])
        writer.writerow(['USD','Cash','Cash','Cash','100','1.5'])
        return out.getvalue()

    def test_dates_small_weights_and_cash_exclusion(self):
        result=parse_index_holdings(self.fixture())
        self.assertEqual(result['asOf'],'2026-09-24')
        self.assertEqual(len(result['holdings']),61)
        tiny=next(r for r in result['holdings'] if r['ticker']=='BRK-B')
        self.assertGreater(tiny['weight'],0)
        self.assertLess(sum(r['weight'] for r in result['holdings']),100)
        self.assertFalse(any(r['ticker']=='USD' for r in result['holdings']))

    def test_duplicate_share_rows_merge(self):
        result=parse_index_holdings(self.fixture().replace('X1,','X0,'))
        self.assertEqual(len(result['holdings']),60)
        weights={r['ticker']:r['weight'] for r in result['holdings']}
        self.assertAlmostEqual(weights['X0'],2*weights['X2'])

    def test_reject_missing_date_html_and_incomplete_file(self):
        for body in ['<html>unavailable</html>',self.fixture().replace('Sep 24, 2026','Sep 24, 2999'),self.fixture().replace('Fund Holdings as of','Unknown'),self.fixture()[:200]]:
            with self.assertRaises(ValueError):parse_index_holdings(body)

    def test_cache_preserves_issuer_provenance(self):
        _UNIVERSE_CACHE.clear()
        with patch('requests.get') as get:
            get.return_value.text=self.fixture();get.return_value.content=self.fixture().encode()
            first=index_holdings('rlv');second=index_holdings('rlv')
            self.assertEqual(get.call_count,1)
            self.assertEqual(first,second)
            self.assertEqual(first['proxy'],'IWD')
            self.assertIn('ishares.com',first['sourceUrl'])
        _UNIVERSE_CACHE.clear()

    def test_universe_routes_fail_visibly_without_touching_portfolio(self):
        app=Flask(__name__);app.register_blueprint(create_blueprint(None));client=app.test_client()
        self.assertEqual(client.get('/api/portfolio/heatmap/universe/unknown').status_code,400)
        with patch('portfolio_heatmap.index_holdings',side_effect=ValueError('unavailable')):
            self.assertEqual(client.get('/api/portfolio/heatmap/universe/spx').status_code,503)
        with patch('portfolio_heatmap.index_holdings',return_value={'name':'SPX','holdings':[]}) as load:
            self.assertEqual(client.get('/api/portfolio/heatmap/universe/spx').status_code,200)
            load.assert_called_once_with('spx')

class PriceCacheTests(unittest.TestCase):
    def test_overlapping_batches_reuse_stock_history_and_short_daily_fetch(self):
        import pandas as pd
        from portfolio_heatmap import market_returns, _HISTORY
        _HISTORY.clear()
        frame=pd.DataFrame({'Close':[100.,101.]},index=pd.to_datetime(['2026-09-24','2026-09-25']))
        with patch('yfinance.Ticker') as ticker:
            ticker.return_value.history.return_value=frame
            first=market_returns(['ABT','JNJ'],'1d')
            self.assertEqual(ticker.call_count,2)
            self.assertEqual(ticker.return_value.history.call_args.kwargs['period'],'5d')
            market_returns(['JNJ','MRK'],'1d')
            self.assertEqual(ticker.call_count,3)
            market_returns(['JNJ'],'1m')
            self.assertEqual(ticker.call_count,4)
            self.assertEqual(ticker.return_value.history.call_args.kwargs['period'],'2y')
            market_returns(['JNJ'],'1d')
            self.assertEqual(ticker.call_count,4)
            self.assertAlmostEqual(first['quotes']['ABT']['changePct'],1.)
        _HISTORY.clear()

    def test_provider_failure_is_missing_and_retryable(self):
        from portfolio_heatmap import market_returns, _HISTORY
        _HISTORY.clear()
        with patch('yfinance.Ticker') as ticker:
            ticker.return_value.history.side_effect=RuntimeError('unavailable')
            self.assertIsNone(market_returns(['ABT'],'1d')['quotes']['ABT']['changePct'])
            market_returns(['ABT'],'1d')
            self.assertEqual(ticker.call_count,2)
        _HISTORY.clear()

if __name__=='__main__':unittest.main()
