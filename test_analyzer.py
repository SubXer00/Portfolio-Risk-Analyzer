"""
test_analyzer.py - Unit test suite for risk_metrics.py and clustering.py.
"""

import unittest
import numpy as np
import pandas as pd

from risk_metrics import (
    calculate_daily_returns,
    calculate_portfolio_returns,
    calculate_annualized_volatility,
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_historical_var,
    calculate_correlation_matrix,
    calculate_cumulative_returns,
)
from clustering import cluster_stocks


class TestPortfolioRiskAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create deterministic synthetic price data
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
        
        # Stock A: Low volatility, steady growth
        ret_a = np.random.normal(0.0005, 0.01, size=252)
        # Stock B: High volatility, high growth
        ret_b = np.random.normal(0.0010, 0.02, size=252)
        # Stock C: Moderate volatility, low return
        ret_c = np.random.normal(0.0002, 0.015, size=252)
        
        price_a = 100 * np.cumprod(1 + ret_a)
        price_b = 100 * np.cumprod(1 + ret_b)
        price_c = 100 * np.cumprod(1 + ret_c)
        
        self.prices_df = pd.DataFrame(
            {"AAPL": price_a, "MSFT": price_b, "JPM": price_c},
            index=dates,
        )
        self.returns_df = calculate_daily_returns(self.prices_df)
        self.weights = [0.4, 0.4, 0.2]

    def test_daily_returns_shape(self):
        self.assertEqual(len(self.returns_df), 251)
        self.assertEqual(list(self.returns_df.columns), ["AAPL", "MSFT", "JPM"])

    def test_portfolio_returns(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        self.assertEqual(len(port_ret), 251)
        # Verify manual dot product for first row
        expected_first = (
            self.returns_df.iloc[0]["AAPL"] * 0.4
            + self.returns_df.iloc[0]["MSFT"] * 0.4
            + self.returns_df.iloc[0]["JPM"] * 0.2
        )
        self.assertAlmostEqual(port_ret.iloc[0], expected_first, places=7)

    def test_annualized_volatility(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        ann_vol = calculate_annualized_volatility(port_ret, trading_days=252)
        expected_vol = port_ret.std() * np.sqrt(252)
        self.assertAlmostEqual(ann_vol, expected_vol, places=7)
        self.assertGreater(ann_vol, 0.0)

    def test_annualized_return(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        ann_ret = calculate_annualized_return(port_ret, trading_days=252)
        expected_ret = port_ret.mean() * 252
        self.assertAlmostEqual(ann_ret, expected_ret, places=7)

    def test_sharpe_ratio(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        sharpe = calculate_sharpe_ratio(port_ret, risk_free_rate=0.04, trading_days=252)
        ann_ret = calculate_annualized_return(port_ret, 252)
        ann_vol = calculate_annualized_volatility(port_ret, 252)
        expected_sharpe = (ann_ret - 0.04) / ann_vol
        self.assertAlmostEqual(sharpe, expected_sharpe, places=7)

    def test_historical_var(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        var_pct, var_dollars = calculate_historical_var(
            port_ret, confidence_level=0.95, portfolio_value=100000.0
        )
        # 5th percentile
        p5 = np.percentile(port_ret, 5.0)
        expected_var_pct = max(0.0, -p5)
        self.assertAlmostEqual(var_pct, expected_var_pct, places=7)
        self.assertAlmostEqual(var_dollars, expected_var_pct * 100000.0, places=4)
        self.assertGreater(var_pct, 0.0)

    def test_correlation_matrix(self):
        corr = calculate_correlation_matrix(self.returns_df)
        self.assertEqual(corr.shape, (3, 3))
        # Diagonal must be 1.0
        for col in corr.columns:
            self.assertAlmostEqual(corr.loc[col, col], 1.0, places=7)

    def test_cumulative_returns(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        cum_ret = calculate_cumulative_returns(port_ret)
        self.assertEqual(len(cum_ret), len(port_ret))
        # First cumulative return = first daily return
        self.assertAlmostEqual(cum_ret.iloc[0], port_ret.iloc[0], places=7)

    def test_clustering_stocks(self):
        cluster_df = cluster_stocks(self.returns_df, k=2)
        self.assertEqual(len(cluster_df), 3)
        self.assertIn("Ticker", cluster_df.columns)
        self.assertIn("Annualized Volatility", cluster_df.columns)
        self.assertIn("Annualized Return", cluster_df.columns)
        self.assertIn("Cluster", cluster_df.columns)
        # Verify cluster labels are present
        clusters = cluster_df["Cluster"].unique()
        self.assertTrue(len(clusters) <= 2)


if __name__ == "__main__":
    unittest.main()
