"""
risk_metrics.py - Pure financial calculation functions for portfolio risk analysis.

Contains standalone, easily testable functions for returns, volatility, Sharpe ratio,
Historical Value at Risk (VaR), correlation, and cumulative growth.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily percentage returns from adjusted close price series: (P_t - P_{t-1}) / P_{t-1}."""
    return prices_df.pct_change().dropna()


def calculate_portfolio_returns(
    returns_df: pd.DataFrame, weights: Union[list, np.ndarray]
) -> pd.Series:
    """Computes weighted portfolio daily return series: R_p = sum(w_i * R_i)."""
    weights_arr = np.array(weights, dtype=float)
    return returns_df.dot(weights_arr)


def calculate_annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame], trading_days: int = 252
) -> float:
    """Computes annualized volatility: sample standard deviation scaled by sqrt(trading_days)."""
    return float(returns.std() * np.sqrt(trading_days))


def calculate_annualized_return(
    returns: Union[pd.Series, pd.DataFrame], trading_days: int = 252
) -> float:
    """Computes annualized expected return: mean daily return scaled by trading_days."""
    return float(returns.mean() * trading_days)


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252
) -> float:
    """Computes annualized Sharpe Ratio: (annualized return - risk_free_rate) / annualized volatility."""
    ann_return = calculate_annualized_return(returns, trading_days)
    ann_vol = calculate_annualized_volatility(returns, trading_days)
    if ann_vol == 0 or np.isnan(ann_vol):
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def calculate_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100000.0,
) -> Tuple[float, float]:
    """Computes 1-day Historical Value at Risk (VaR) at given confidence level as (loss percentage, loss dollar amount)."""
    # 5th percentile for 95% confidence represents the cutoff for worst 5% daily returns
    percentile_cutoff = (1.0 - confidence_level) * 100.0
    var_cutoff = float(np.percentile(returns, percentile_cutoff))
    # Express VaR as a positive loss magnitude (e.g., 0.02 = 2.0% potential 1-day loss)
    var_pct_loss = max(0.0, -var_cutoff) if var_cutoff < 0 else 0.0
    var_dollar_loss = var_pct_loss * portfolio_value
    return var_pct_loss, var_dollar_loss


def calculate_expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 100000.0,
) -> Tuple[float, float]:
    """Computes 1-day Expected Shortfall (CVaR) as the average of returns at or below the historical VaR threshold."""
    percentile_cutoff = (1.0 - confidence_level) * 100.0
    var_cutoff = float(np.percentile(returns, percentile_cutoff))
    # Filter returns at or below the VaR cutoff
    tail_returns = returns[returns <= var_cutoff]
    if len(tail_returns) == 0:
        cvar_cutoff = var_cutoff
    else:
        cvar_cutoff = float(tail_returns.mean())
    # Express CVaR as a positive loss magnitude
    cvar_pct_loss = max(0.0, -cvar_cutoff) if cvar_cutoff < 0 else 0.0
    cvar_dollar_loss = cvar_pct_loss * portfolio_value
    return cvar_pct_loss, cvar_dollar_loss


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Pearson correlation coefficient matrix between individual asset daily returns."""
    return returns_df.corr()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Computes compounded cumulative portfolio return series: cumprod(1 + R) - 1."""
    return (1.0 + returns).cumprod() - 1.0

