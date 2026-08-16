"""
app.py - Streamlit User Interface for Portfolio Risk Analyzer.

Provides an interactive dashboard for portfolio analysis:
- Stock ticker & portfolio weight management
- Key risk metrics (Annualized Volatility, Sharpe Ratio, 1-Day 95% Historical VaR)
- Cumulative performance, return distribution, and correlation heatmap
- Unsupervised K-Means clustering of assets by Risk/Return profile
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf

from clustering import cluster_stocks
from risk_metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_correlation_matrix,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_historical_var,
    calculate_portfolio_returns,
    calculate_sharpe_ratio,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished card UI and readable layout
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888888;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 6px;
    }
    .metric-desc {
        font-size: 0.82rem;
        line-height: 1.35;
        color: #aaaaaa;
    }
    .section-header {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA FETCHING HELPER WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock_prices(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Downloads historical daily adjusted close prices for given tickers using yfinance."""
    if not tickers:
        return pd.DataFrame()

    # Format dates as YYYY-MM-DD
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    # Handle multi-ticker vs single-ticker DataFrame structure from yfinance
    if "Close" in data.columns:
        close_data = data["Close"]
    else:
        close_data = data

    # Ensure result is a DataFrame with tickers as columns
    if isinstance(close_data, pd.Series):
        close_df = close_data.to_frame(name=tickers[0])
    else:
        close_df = close_data.copy()

    # Drop any tickers that are entirely NaN
    close_df = close_df.dropna(how="all", axis=1)
    # Forward-fill and drop remaining NaNs
    close_df = close_df.ffill().dropna()

    return close_df


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & USER INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Portfolio Setup")
st.sidebar.markdown("Configure your portfolio assets, weights, and parameters.")

# 1. Tickers Input
default_tickers_str = "AAPL, MSFT, JPM, XOM"
tickers_input = st.sidebar.text_input(
    "Stock Tickers (2–6 comma-separated)",
    value=default_tickers_str,
    help="Enter 2 to 6 valid stock ticker symbols, separated by commas (e.g. AAPL, MSFT, JPM, XOM).",
)

# Parse and clean tickers
parsed_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Validate ticker count
ticker_count_valid = 2 <= len(parsed_tickers) <= 6
if not ticker_count_valid:
    st.sidebar.error(f"⚠️ Please enter between 2 and 6 tickers. (Current count: {len(parsed_tickers)})")

# 2. Portfolio Weights Input
st.sidebar.markdown("### Portfolio Weights")
equal_weight = 1.0 / len(parsed_tickers) if parsed_tickers else 0.0

# Store weights in session state or build dynamic inputs
weights = []
for i, ticker in enumerate(parsed_tickers):
    col1, col2 = st.sidebar.columns([2, 3])
    with col1:
        st.write(f"**{ticker}**")
    with col2:
        # Default to equal weight
        val = st.number_input(
            f"Weight for {ticker}",
            min_value=0.0,
            max_value=1.0,
            value=float(round(equal_weight, 4)),
            step=0.05,
            key=f"weight_{ticker}_{i}",
            label_visibility="collapsed",
        )
        weights.append(val)

weights_sum = sum(weights)
weights_valid = np.isclose(weights_sum, 1.0, atol=0.002)

if not weights_valid:
    st.sidebar.warning(f"⚠️ Total Weight: **{weights_sum * 100:.1f}%** (Must sum to **100%**)")
else:
    st.sidebar.success(f"✅ Total Weight: **{weights_sum * 100:.1f}%**")

# 3. Date Range Picker
st.sidebar.markdown("### Historical Period")
today = datetime.today()
two_years_ago = today - timedelta(days=730)

start_date = st.sidebar.date_input("Start Date", value=two_years_ago)
end_date = st.sidebar.date_input("End Date", value=today)

if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")

# 4. Financial & Model Parameters
st.sidebar.markdown("### Parameters")
risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate ($R_f$)",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
    help="Default is 0.04 (4.0%), representing the risk-free benchmark (e.g. US Treasury yield).",
)

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=1000.0,
    max_value=100000000.0,
    value=100000.0,
    step=10000.0,
    format="%.2f",
    help="Initial capital used to compute dollar Value at Risk (VaR).",
)

# 5. K-Means Cluster Count
max_k = min(3, len(parsed_tickers)) if len(parsed_tickers) >= 2 else 2
k_clusters = st.sidebar.radio(
    "K-Means Clusters ($k$)",
    options=[2, 3] if len(parsed_tickers) >= 3 else [2],
    index=0,
    horizontal=True,
    help="Number of risk/return groups for unsupervised K-Means clustering.",
)


# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("📊 Portfolio Risk Analyzer")
st.markdown(
    "Analyze key risk and return metrics for a multi-asset equity portfolio, "
    "simulate historical Value at Risk (VaR), and discover asset profiles with unsupervised K-Means clustering."
)
st.divider()

# Validation Gate
if not ticker_count_valid:
    st.info("👈 Please enter between 2 and 6 stock tickers in the sidebar to begin analysis.")
    st.stop()

if start_date >= end_date:
    st.warning("👈 Please select a valid date range in the sidebar.")
    st.stop()

if not weights_valid:
    st.error(
        f"🚨 **Invalid Weights:** The sum of portfolio weights is **{weights_sum * 100:.2f}%**. "
        "Please adjust the weights in the sidebar to sum exactly to **100.0%** (1.0)."
    )
    st.stop()

# -----------------------------------------------------------------------------
# DATA FETCHING & PROCESSING
# -----------------------------------------------------------------------------
with st.spinner("Fetching historical market data from Yahoo Finance..."):
    prices_df = fetch_stock_prices(parsed_tickers, start_date, end_date)

if prices_df.empty or len(prices_df) < 20:
    st.error(
        "❌ Unable to retrieve sufficient price history for the selected tickers and date range. "
        "Please verify that the ticker symbols are valid and that the date range spans at least one month."
    )
    st.stop()

# Check if any tickers were omitted due to lack of data
missing_tickers = [t for t in parsed_tickers if t not in prices_df.columns]
if missing_tickers:
    st.warning(f"⚠️ No price data found for: {', '.join(missing_tickers)}. Proceeding with available assets.")

active_tickers = [t for t in parsed_tickers if t in prices_df.columns]
if len(active_tickers) < 2:
    st.error("❌ At least 2 valid stock tickers with price data are required.")
    st.stop()

# Filter and re-normalize weights for active tickers if needed
active_weights = [weights[parsed_tickers.index(t)] for t in active_tickers]
active_weights_sum = sum(active_weights)
if not np.isclose(active_weights_sum, 1.0, atol=0.002):
    active_weights = [w / active_weights_sum for w in active_weights]

# Compute Daily Returns & Portfolio Returns
daily_returns_df = calculate_daily_returns(prices_df[active_tickers])
portfolio_daily_returns = calculate_portfolio_returns(daily_returns_df, active_weights)

# -----------------------------------------------------------------------------
# KEY RISK METRICS CALCULATION
# -----------------------------------------------------------------------------
ann_volatility = calculate_annualized_volatility(portfolio_daily_returns)
ann_return = calculate_annualized_return(portfolio_daily_returns)
sharpe_ratio = calculate_sharpe_ratio(portfolio_daily_returns, risk_free_rate=risk_free_rate)
var_pct, var_dollars = calculate_historical_var(
    portfolio_daily_returns,
    confidence_level=0.95,
    portfolio_value=portfolio_value,
)

# -----------------------------------------------------------------------------
# 1. TOP METRIC CARDS
# -----------------------------------------------------------------------------
st.markdown("### 📈 Key Portfolio Risk Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Annualized Volatility</div>
            <div class="metric-value">{ann_volatility * 100:.2f}%</div>
            <div class="metric-desc">
                Measures annual return dispersion ($\sigma \cdot \sqrt{{252}}$). Higher volatility signifies greater price fluctuation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    sharpe_color = "#2E7D32" if sharpe_ratio >= 1.0 else ("#F57C00" if sharpe_ratio >= 0 else "#D32F2F")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Sharpe Ratio (Ann.)</div>
            <div class="metric-value" style="color: {sharpe_color};">{sharpe_ratio:.2f}</div>
            <div class="metric-desc">
                Excess return per unit of volatility above the {risk_free_rate*100:.1f}% risk-free rate. Values &gt; 1.0 indicate attractive risk-adjusted returns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">1-Day 95% Historical VaR (%)</div>
            <div class="metric-value" style="color: #E65100;">{var_pct * 100:.2f}%</div>
            <div class="metric-desc">
                Maximum expected 1-day percentage loss with 95% statistical confidence based on past return distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">1-Day 95% Historical VaR ($)</div>
            <div class="metric-value" style="color: #E65100;">${var_dollars:,.2f}</div>
            <div class="metric-desc">
                Estimated maximum 1-day dollar loss on your ${portfolio_value:,.0f} portfolio under normal market conditions (95% confidence).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VISUALIZATIONS SECTION
# -----------------------------------------------------------------------------
st.markdown("### 📊 Performance & Risk Visualizations")

chart_tab1, chart_tab2 = st.columns(2)

with chart_tab1:
    # --- CHART 1: Cumulative Portfolio Growth ---
    st.markdown("#### Cumulative Portfolio Return")
    cum_returns = calculate_cumulative_returns(portfolio_daily_returns)
    
    fig_cum, ax_cum = plt.subplots(figsize=(7, 4.2), dpi=100)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    
    ax_cum.plot(cum_returns.index, cum_returns * 100, color="#1976D2", linewidth=2.0, label="Portfolio")
    ax_cum.axhline(0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Fill positive/negative areas
    ax_cum.fill_between(
        cum_returns.index,
        cum_returns * 100,
        0,
        where=(cum_returns >= 0),
        color="#1976D2",
        alpha=0.15,
    )
    ax_cum.fill_between(
        cum_returns.index,
        cum_returns * 100,
        0,
        where=(cum_returns < 0),
        color="#D32F2F",
        alpha=0.15,
    )
    
    ax_cum.set_ylabel("Cumulative Return (%)", fontsize=10, fontweight="bold")
    ax_cum.set_xlabel("Date", fontsize=10)
    ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}%"))
    ax_cum.set_title("Compounded Growth Over Time", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig_cum)
    plt.close(fig_cum)

    st.caption(
        "💡 **Cumulative Return**: Demonstrates the compounded growth trajectory of the weighted portfolio "
        f"over the selected period (Total Return: **{cum_returns.iloc[-1] * 100:+.2f}%**)."
    )

with chart_tab2:
    # --- CHART 2: Return Distribution & VaR Threshold ---
    st.markdown("#### Daily Return Distribution & 95% VaR")
    
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4.2), dpi=100)
    
    # 5th percentile return value (negative number)
    var_cutoff_ret = -var_pct
    
    # Plot histogram with KDE
    sns.histplot(
        portfolio_daily_returns * 100,
        kde=True,
        bins=35,
        color="#37474F",
        ax=ax_hist,
        stat="density",
        edgecolor="white",
        linewidth=0.5,
    )
    
    # Add VaR cutoff vertical line
    ax_hist.axvline(
        var_cutoff_ret * 100,
        color="#D32F2F",
        linestyle="--",
        linewidth=2.0,
        label=f"95% 1-Day VaR Threshold ({var_cutoff_ret * 100:.2f}%)",
    )
    
    # Highlight the tail (left 5% region)
    kde_x = ax_hist.get_lines()[0].get_xdata() if ax_hist.get_lines() else []
    kde_y = ax_hist.get_lines()[0].get_ydata() if ax_hist.get_lines() else []
    if len(kde_x) > 0:
        tail_mask = kde_x <= (var_cutoff_ret * 100)
        ax_hist.fill_between(
            kde_x[tail_mask],
            kde_y[tail_mask],
            color="#D32F2F",
            alpha=0.35,
            label="Worst 5% Tail Loss Region",
        )
    
    ax_hist.set_xlabel("Daily Return (%)", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_hist.set_title("Historical Daily Return Histogram", fontsize=11, fontweight="bold")
    ax_hist.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.caption(
        "💡 **Value at Risk (VaR)**: The red dashed line marks the 5th percentile cutoff. "
        f"Only 5% of historical trading days experienced a daily loss worse than **{var_pct * 100:.2f}%**."
    )

st.markdown("<br>", unsafe_allow_html=True)

chart_tab3, chart_tab4 = st.columns(2)

with chart_tab3:
    # --- CHART 3: Stock Correlation Heatmap ---
    st.markdown("#### Asset Return Correlation Heatmap")
    corr_matrix = calculate_correlation_matrix(daily_returns_df)
    
    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5), dpi=100)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax_corr,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax_corr.set_title("Pairwise Daily Return Correlations", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.caption(
        "💡 **Correlation Matrix**: Coefficients range from -1 (perfect inverse movement) to +1 (identical movement). "
        "Lower or negative pairwise correlations provide superior risk diversification."
    )

with chart_tab4:
    # --- CHART 4: K-Means Clustering Scatter Plot ---
    st.markdown(f"#### K-Means Asset Risk/Return Clusters ($k={k_clusters}$)")
    
    try:
        cluster_df = cluster_stocks(daily_returns_df, k=k_clusters)
        
        fig_cluster, ax_cluster = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        
        cluster_colors = {
            "Cluster 1": "#1E88E5",
            "Cluster 2": "#E53935",
            "Cluster 3": "#43A047",
        }
        
        # Scatter plot of assets
        for cluster_name, group in cluster_df.groupby("Cluster"):
            color = cluster_colors.get(cluster_name, "#8E24AA")
            ax_cluster.scatter(
                group["Annualized Volatility"] * 100,
                group["Annualized Return"] * 100,
                s=160,
                color=color,
                label=cluster_name,
                alpha=0.85,
                edgecolors="black",
                linewidth=1.2,
                zorder=4,
            )
            
            # Annotate ticker labels
            for _, row in group.iterrows():
                ax_cluster.annotate(
                    row["Ticker"],
                    (row["Annualized Volatility"] * 100, row["Annualized Return"] * 100),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color="#212121",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
                )
        
        ax_cluster.set_xlabel("Annualized Volatility (%) [Risk]", fontsize=10, fontweight="bold")
        ax_cluster.set_ylabel("Annualized Return (%) [Reward]", fontsize=10, fontweight="bold")
        ax_cluster.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax_cluster.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}%"))
        ax_cluster.set_title("Stock Clustering: Volatility vs. Return", fontsize=11, fontweight="bold")
        ax_cluster.legend(loc="best", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cluster)
        plt.close(fig_cluster)
        
        st.caption(
            "💡 **K-Means Clustering**: Unsupervised algorithm grouping individual stocks into distinct "
            "risk/reward profiles based on their annualized return and annualized volatility features."
        )
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")

# -----------------------------------------------------------------------------
# 3. ASSET BREAKDOWN TABLE
# -----------------------------------------------------------------------------
st.markdown("### 📋 Asset Risk & Return Breakdown")

summary_table = cluster_df.copy()
summary_table["Weight"] = [
    f"{active_weights[active_tickers.index(t)] * 100:.1f}%"
    if t in active_tickers else "0.0%"
    for t in summary_table["Ticker"]
]
summary_table["Annualized Return"] = summary_table["Annualized Return"].apply(lambda r: f"{r * 100:+.2f}%")
summary_table["Annualized Volatility"] = summary_table["Annualized Volatility"].apply(lambda v: f"{v * 100:.2f}%")

st.dataframe(
    summary_table[["Ticker", "Weight", "Annualized Return", "Annualized Volatility", "Cluster"]],
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# FOOTER & METHODOLOGY NOTES
# -----------------------------------------------------------------------------
st.divider()
with st.expander("📖 Financial Formulas & Methodology Guide"):
    st.markdown(
        """
        - **Daily Return ($R_t$):** $R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}$
        - **Portfolio Weighted Return ($R_{p,t}$):** $R_{p,t} = \\sum_{i=1}^{N} w_i \\cdot R_{i,t}$
        - **Annualized Volatility ($\sigma_{\\text{ann}}$):** $\\sigma_{\\text{daily}} \\times \\sqrt{252}$
        - **Annualized Return ($\mu_{\\text{ann}}$):** $\\mu_{\\text{daily}} \\times 252$
        - **Sharpe Ratio:** $\\frac{\mu_{\\text{ann}} - R_f}{\sigma_{\\text{ann}}}$, where $R_f$ is the risk-free rate.
        - **1-Day 95% Historical VaR:** The 5th percentile of the empirical daily portfolio return distribution.
        - **K-Means Clustering:** An unsupervised clustering algorithm that minimizes within-cluster sum-of-squares (inertia) across the 2-dimensional feature space $(\sigma_{\\text{ann}}, \mu_{\\text{ann}})$.
        """
    )
