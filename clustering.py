"""
clustering.py - Unsupervised K-Means clustering for asset risk/return profiling.

Extracts annualized return and annualized volatility features for each stock
and groups assets into risk/return clusters using scikit-learn's KMeans.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from risk_metrics import calculate_annualized_return, calculate_annualized_volatility


def cluster_stocks(
    returns_df: pd.DataFrame,
    k: int = 2,
    random_state: int = 42,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Groups stocks into k risk/return clusters using K-Means on annualized volatility and annualized return features."""
    num_stocks = len(returns_df.columns)
    if num_stocks < 2:
        raise ValueError("At least 2 stock tickers are required for clustering.")
    
    # Ensure k does not exceed number of stocks
    k_actual = min(max(2, k), num_stocks)

    # Compute features: Annualized Volatility (Risk) and Annualized Return
    features = []
    for ticker in returns_df.columns:
        stock_returns = returns_df[ticker]
        ann_vol = calculate_annualized_volatility(stock_returns, trading_days)
        ann_ret = calculate_annualized_return(stock_returns, trading_days)
        features.append({
            "Ticker": ticker,
            "Annualized Volatility": ann_vol,
            "Annualized Return": ann_ret,
        })

    features_df = pd.DataFrame(features)

    # Feature matrix (X) with 2 dimensions: Volatility (x-axis) and Return (y-axis)
    X = features_df[["Annualized Volatility", "Annualized Return"]].values

    # Run unsupervised K-Means clustering
    kmeans = KMeans(n_clusters=k_actual, random_state=random_state, n_init="auto")
    features_df["Cluster_ID"] = kmeans.fit_predict(X)
    features_df["Cluster"] = features_df["Cluster_ID"].apply(lambda c: f"Cluster {c + 1}")

    return features_df
