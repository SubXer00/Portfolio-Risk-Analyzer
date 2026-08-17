"""
test_analyzer.py - Unit test suite for risk_metrics.py, clustering.py, and breach_classifier.py.
"""

import unittest
import numpy as np
import pandas as pd

from breach_classifier import (
    build_breach_features,
    evaluate_breach_model,
    extract_feature_importance,
    generate_breach_labels,
    time_series_train_test_split,
    train_breach_model,
)
from clustering import cluster_stocks
from risk_metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_correlation_matrix,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_expected_shortfall,
    calculate_historical_var,
    calculate_portfolio_returns,
    calculate_sharpe_ratio,
)


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

        # Synthetic volume data
        vol_a = np.random.uniform(1000000, 2000000, size=len(self.returns_df))
        vol_b = np.random.uniform(2000000, 3000000, size=len(self.returns_df))
        vol_c = np.random.uniform(500000, 1500000, size=len(self.returns_df))
        self.volume_df = pd.DataFrame(
            {"AAPL": vol_a, "MSFT": vol_b, "JPM": vol_c},
            index=self.returns_df.index,
        )

    def test_daily_returns_shape(self):
        self.assertEqual(len(self.returns_df), 251)
        self.assertEqual(list(self.returns_df.columns), ["AAPL", "MSFT", "JPM"])

    def test_portfolio_returns(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        self.assertEqual(len(port_ret), 251)
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
        p5 = np.percentile(port_ret, 5.0)
        expected_var_pct = max(0.0, -p5)
        self.assertAlmostEqual(var_pct, expected_var_pct, places=7)
        self.assertAlmostEqual(var_dollars, expected_var_pct * 100000.0, places=4)
        self.assertGreater(var_pct, 0.0)

    def test_expected_shortfall(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        var_pct, var_dollars = calculate_historical_var(
            port_ret, confidence_level=0.95, portfolio_value=100000.0
        )
        cvar_pct, cvar_dollars = calculate_expected_shortfall(
            port_ret, confidence_level=0.95, portfolio_value=100000.0
        )
        # Mathematically, Expected Shortfall loss magnitude must be >= VaR loss magnitude
        self.assertGreaterEqual(cvar_pct, var_pct)
        self.assertGreaterEqual(cvar_dollars, var_dollars)
        self.assertAlmostEqual(cvar_dollars, cvar_pct * 100000.0, places=4)

    def test_correlation_matrix(self):
        corr = calculate_correlation_matrix(self.returns_df)
        self.assertEqual(corr.shape, (3, 3))
        for col in corr.columns:
            self.assertAlmostEqual(corr.loc[col, col], 1.0, places=7)

    def test_cumulative_returns(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        cum_ret = calculate_cumulative_returns(port_ret)
        self.assertEqual(len(cum_ret), len(port_ret))
        self.assertAlmostEqual(cum_ret.iloc[0], port_ret.iloc[0], places=7)

    def test_clustering_stocks(self):
        cluster_df = cluster_stocks(self.returns_df, k=2)
        self.assertEqual(len(cluster_df), 3)
        self.assertIn("Ticker", cluster_df.columns)
        self.assertIn("Annualized Volatility", cluster_df.columns)
        self.assertIn("Annualized Return", cluster_df.columns)
        self.assertIn("Cluster", cluster_df.columns)
        clusters = cluster_df["Cluster"].unique()
        self.assertTrue(len(clusters) <= 2)

    def test_breach_features_no_lookahead(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        features = build_breach_features(port_ret, self.volume_df, self.weights)
        
        self.assertEqual(
            list(features.columns),
            ["Rolling 20D Volatility", "5D Momentum", "20D Volume Trend"],
        )
        self.assertEqual(len(features), len(port_ret))
        
        # Verify no lookahead: day t feature cannot use day t return
        # First 20 rows of rolling vol must be NaN due to shift(1) + window=20
        self.assertTrue(np.isnan(features["Rolling 20D Volatility"].iloc[0]))
        self.assertTrue(np.isnan(features["Rolling 20D Volatility"].iloc[19]))
        self.assertFalse(np.isnan(features["Rolling 20D Volatility"].iloc[20]))

    def test_breach_labels(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        labels = generate_breach_labels(port_ret, confidence_level=0.95)
        
        self.assertEqual(len(labels), len(port_ret))
        # Labels must be binary (0 or 1)
        self.assertTrue(set(labels.unique()).issubset({0, 1}))
        # Breaches should be approximately 5% of days
        breach_count = labels.sum()
        self.assertGreater(breach_count, 0)
        self.assertLess(breach_count, len(port_ret) * 0.15)

    def test_time_series_split_no_leakage(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        features = build_breach_features(port_ret, self.volume_df, self.weights)
        labels = generate_breach_labels(port_ret, confidence_level=0.95)
        
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            features, labels, train_ratio=0.8
        )
        
        # Verify chronological ordering: all train dates < test dates (no leakage)
        self.assertLess(X_train.index.max(), X_test.index.min())
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), len(X_test))

    def test_breach_models_and_evaluation(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        features = build_breach_features(port_ret, self.volume_df, self.weights)
        labels = generate_breach_labels(port_ret, confidence_level=0.95)
        
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            features, labels, train_ratio=0.8
        )
        
        # Test Random Forest
        rf_model = train_breach_model(X_train, y_train, model_type="Random Forest")
        rf_metrics = evaluate_breach_model(rf_model, X_test, y_test)
        self.assertIn("precision", rf_metrics)
        self.assertIn("recall", rf_metrics)
        self.assertIn("f1", rf_metrics)
        self.assertIn("accuracy", rf_metrics)
        self.assertEqual(rf_metrics["confusion_matrix"].shape, (2, 2))
        
        # Test Logistic Regression
        lr_model = train_breach_model(X_train, y_train, model_type="Logistic Regression")
        lr_metrics = evaluate_breach_model(lr_model, X_test, y_test)
        self.assertIn("precision", lr_metrics)
        self.assertIn("recall", lr_metrics)
        self.assertIn("f1", lr_metrics)

    def test_feature_importance_extraction(self):
        port_ret = calculate_portfolio_returns(self.returns_df, self.weights)
        features = build_breach_features(port_ret, self.volume_df, self.weights)
        labels = generate_breach_labels(port_ret, confidence_level=0.95)
        
        X_train, _, y_train, _ = time_series_train_test_split(features, labels)
        
        # Random Forest importance
        rf_model = train_breach_model(X_train, y_train, model_type="Random Forest")
        rf_imp = extract_feature_importance(rf_model, list(X_train.columns))
        self.assertEqual(len(rf_imp), 3)
        self.assertIn("Feature", rf_imp.columns)
        self.assertIn("Importance", rf_imp.columns)
        self.assertEqual(rf_imp["Type"].iloc[0], "Feature Importance (MDI)")
        
        # Logistic Regression coefficients
        lr_model = train_breach_model(X_train, y_train, model_type="Logistic Regression")
        lr_imp = extract_feature_importance(lr_model, list(X_train.columns))
        self.assertEqual(len(lr_imp), 3)
        self.assertEqual(lr_imp["Type"].iloc[0], "Logistic Coefficient")


if __name__ == "__main__":
    unittest.main()
