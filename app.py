"""
app.py - Streamlit Dashboard for Portfolio Risk Analyzer.

Features:
- Persistent sidebar setup for stock tickers, weights, date range, and financial parameters
- 5 Structured Tabs: Overview, Risk Metrics, Visualizations, Clustering, Breach Prediction
- Key metrics including Annualized Volatility, Sharpe Ratio, 1-Day 95% Historical VaR, and Expected Shortfall (CVaR)
- Compounded performance charts, VaR/CVaR return distribution histograms, and correlation heatmaps
- Unsupervised K-Means clustering of asset risk/return regimes
- Supervised Machine Learning for VaR breach classification with Feature Importance analysis
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf

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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-themed cards, metric alignment, and clean tab typography
st.markdown(
    """
    <style>
    .metric-container-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    /* Hide the blank grey placeholder box Streamlit renders above st.metric() labels */
    .metric-container-card [data-testid="stMetricLabel"] > div:empty,
    .metric-container-card [data-testid="metric-container"] > div:first-child:empty {
        display: none !important;
    }
    .metric-caption {
        font-size: 0.82rem;
        line-height: 1.35;
        color: #8B949E;
        margin-top: 6px;
    }
    .section-banner {
        padding: 10px 14px;
        background: rgba(0, 180, 216, 0.08);
        border-left: 4px solid #00B4D8;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #E6EDF3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA FETCHING HELPER (CACHED WITH SPINNER)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_portfolio_market_data(
    tickers: List[str], start_date: datetime, end_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads adjusted close prices and trading volume for selected tickers using yfinance."""
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    raw_data = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        auto_adjust=True,
        progress=False,
    )

    if raw_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Extract Close Prices
    if "Close" in raw_data.columns:
        close_raw = raw_data["Close"]
    else:
        close_raw = raw_data

    # Extract Trading Volume
    if "Volume" in raw_data.columns:
        volume_raw = raw_data["Volume"]
    else:
        volume_raw = pd.DataFrame()

    # Normalize single-ticker vs multi-ticker dataframes
    if isinstance(close_raw, pd.Series):
        close_df = close_raw.to_frame(name=tickers[0])
    else:
        close_df = close_raw.copy()

    if isinstance(volume_raw, pd.Series):
        volume_df = volume_raw.to_frame(name=tickers[0])
    else:
        volume_df = volume_raw.copy() if not volume_raw.empty else pd.DataFrame()

    # Clean data
    close_df = close_df.dropna(how="all", axis=1).ffill().dropna()
    if not volume_df.empty:
        volume_df = volume_df[close_df.columns].ffill().fillna(0.0)

    return close_df, volume_df


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & PERSISTENT CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Portfolio Configuration")
st.sidebar.markdown("Configure your portfolio assets, weights, and parameters.")

# 1. Tickers Input
default_tickers_str = "AAPL, MSFT, JPM, XOM"
tickers_input = st.sidebar.text_input(
    "Stock Tickers (2–6 comma-separated)",
    value=default_tickers_str,
    help="Enter 2 to 6 valid stock ticker symbols (e.g. AAPL, MSFT, JPM, XOM).",
)
parsed_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Validate ticker count
ticker_count_valid = 2 <= len(parsed_tickers) <= 6
if not ticker_count_valid:
    st.sidebar.error(f"⚠️ Please enter between 2 and 6 tickers. (Current: {len(parsed_tickers)})")

# 2. Portfolio Weights Input
st.sidebar.markdown("### Portfolio Weights")
equal_weight = 1.0 / len(parsed_tickers) if parsed_tickers else 0.0

weights = []
for i, ticker in enumerate(parsed_tickers):
    col_sym, col_wt = st.sidebar.columns([2, 3])
    with col_sym:
        st.write(f"**{ticker}**")
    with col_wt:
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
st.sidebar.markdown("### Historical Date Range")
today = datetime.today()
two_years_ago = today - timedelta(days=730)
start_date = st.sidebar.date_input("Start Date", value=two_years_ago)
end_date = st.sidebar.date_input("End Date", value=today)

if start_date >= end_date:
    st.sidebar.error("Start Date must be earlier than End Date.")

# 4. Financial Parameters
st.sidebar.markdown("### Financial Parameters")
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate ($R_f$)",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
    help="Annual risk-free benchmark rate (e.g. 0.04 for 4.0% Treasury yield).",
)

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=1000.0,
    max_value=100000000.0,
    value=100000.0,
    step=10000.0,
    format="%.2f",
    help="Total portfolio capital for dollar Value at Risk and Expected Shortfall calculations.",
)

# 5. ML & Clustering Parameters
st.sidebar.markdown("### Machine Learning Setup")
k_clusters = st.sidebar.radio(
    "K-Means Clusters ($k$)",
    options=[2, 3] if len(parsed_tickers) >= 3 else [2],
    index=0,
    horizontal=True,
    help="Number of risk/return regimes to group individual stocks into.",
)

breach_model_type = st.sidebar.selectbox(
    "VaR Breach Classifier Model",
    options=["Random Forest", "Logistic Regression"],
    index=0,
    help="Supervised classification algorithm used to predict high-risk breach days.",
)


# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("📈 Portfolio Risk Analyzer")
st.markdown(
    "A multi-asset risk intelligence platform calculating parametric & historical risk metrics, "
    "Expected Shortfall (CVaR), asset clustering regimes, and supervised VaR breach classification."
)
st.divider()

# Input Validation Gates
if not ticker_count_valid:
    st.info("👈 Please enter between 2 and 6 stock tickers in the sidebar to run analysis.")
    st.stop()

if start_date >= end_date:
    st.warning("👈 Please select a valid date range in the sidebar.")
    st.stop()

if not weights_valid:
    st.error(
        f"🚨 **Invalid Weights:** Current sum is **{weights_sum * 100:.2f}%**. "
        "Please adjust weights in the sidebar to sum to **100.0%**."
    )
    st.stop()


# -----------------------------------------------------------------------------
# DATA ENGINE & CALCULATIONS
# -----------------------------------------------------------------------------
with st.spinner("Fetching market data from Yahoo Finance..."):
    prices_df, volume_df = fetch_portfolio_market_data(parsed_tickers, start_date, end_date)

if prices_df.empty or len(prices_df) < 25:
    st.error(
        "❌ Insufficient price history returned. Please verify ticker symbols and ensure the date range spans at least 2 months."
    )
    st.stop()

# Ensure active tickers
active_tickers = [t for t in parsed_tickers if t in prices_df.columns]
if len(active_tickers) < 2:
    st.error("❌ At least 2 valid stock tickers with price data are required.")
    st.stop()

# Re-normalize active weights if necessary
active_weights = [weights[parsed_tickers.index(t)] for t in active_tickers]
active_weights_sum = sum(active_weights)
if not np.isclose(active_weights_sum, 1.0, atol=0.002):
    active_weights = [w / active_weights_sum for w in active_weights]

# 1. Compute Return Series
daily_returns_df = calculate_daily_returns(prices_df[active_tickers])
portfolio_daily_returns = calculate_portfolio_returns(daily_returns_df, active_weights)

# 2. Compute Core Financial Risk Metrics
ann_volatility = calculate_annualized_volatility(portfolio_daily_returns)
ann_return = calculate_annualized_return(portfolio_daily_returns)
sharpe_ratio = calculate_sharpe_ratio(portfolio_daily_returns, risk_free_rate=risk_free_rate)
var_pct, var_dollars = calculate_historical_var(
    portfolio_daily_returns, confidence_level=0.95, portfolio_value=portfolio_value
)
cvar_pct, cvar_dollars = calculate_expected_shortfall(
    portfolio_daily_returns, confidence_level=0.95, portfolio_value=portfolio_value
)

# 3. K-Means Clustering
cluster_df = cluster_stocks(daily_returns_df, k=k_clusters)

# 4. Supervised VaR Breach Classifier Engine
with st.spinner("Training VaR breach prediction model..."):
    breach_features = build_breach_features(
        portfolio_returns=portfolio_daily_returns,
        volume_df=volume_df[active_tickers] if not volume_df.empty else None,
        weights=active_weights,
    )
    breach_labels = generate_breach_labels(portfolio_daily_returns, confidence_level=0.95)
    
    try:
        X_train, X_test, y_train, y_test = time_series_train_test_split(
            breach_features, breach_labels, train_ratio=0.8
        )
        classifier_model = train_breach_model(
            X_train, y_train, model_type=breach_model_type, random_state=42
        )
        model_metrics = evaluate_breach_model(classifier_model, X_test, y_test)
        feature_importance_df = extract_feature_importance(
            classifier_model, list(X_train.columns)
        )
        classification_success = True
    except Exception as exc:
        classification_success = False
        classification_error = str(exc)


# -----------------------------------------------------------------------------
# TABBED USER INTERFACE
# -----------------------------------------------------------------------------
tab_overview, tab_metrics, tab_viz, tab_cluster, tab_breach = st.tabs(
    [
        "📋 Overview",
        "📈 Risk Metrics",
        "📊 Visualizations",
        "🎯 Clustering",
        "🤖 Breach Prediction",
    ]
)

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("📋 Portfolio Overview & Allocation")
    
    st.markdown(
        f"""
        <div class="section-banner">
            Analyzing <b>{len(active_tickers)} assets</b> over <b>{len(prices_df)} trading days</b> 
            ({prices_df.index.min().strftime('%b %d, %Y')} to {prices_df.index.max().strftime('%b %d, %Y')}). 
            Portfolio Value: <b>${portfolio_value:,.2f}</b> | Benchmark Risk-Free Rate: <b>{risk_free_rate*100:.1f}%</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ov_col1, ov_col2 = st.columns([2, 3])

    with ov_col1:
        st.markdown("#### 🏷️ Asset Weights")
        alloc_data = []
        for ticker, wt in zip(active_tickers, active_weights):
            alloc_data.append({"Ticker": ticker, "Allocation Weight": f"{wt * 100:.1f}%"})
        st.dataframe(pd.DataFrame(alloc_data), use_container_width=True, hide_index=True)

        st.markdown("#### ⚡ Quick Snapshot")
        snap_col1, snap_col2 = st.columns(2)
        snap_col1.metric("Ann. Volatility", f"{ann_volatility * 100:.2f}%")
        snap_col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    with ov_col2:
        st.markdown("#### 📈 Compounded Growth Preview")
        cum_returns = calculate_cumulative_returns(portfolio_daily_returns)
        fig_ov, ax_ov = plt.subplots(figsize=(7, 3.8), dpi=100)
        ax_ov.set_facecolor("#161B22")
        fig_ov.patch.set_facecolor("#0D1117")
        ax_ov.plot(cum_returns.index, cum_returns * 100, color="#00B4D8", linewidth=2.0)
        ax_ov.axhline(0, color="#8B949E", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_ov.fill_between(cum_returns.index, cum_returns * 100, 0, where=(cum_returns >= 0), color="#00B4D8", alpha=0.2)
        ax_ov.fill_between(cum_returns.index, cum_returns * 100, 0, where=(cum_returns < 0), color="#EF476F", alpha=0.2)
        ax_ov.set_ylabel("Cumulative Return (%)", color="#E6EDF3", fontsize=9)
        ax_ov.tick_params(colors="#8B949E", labelsize=8)
        ax_ov.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}%"))
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig_ov)
        plt.close(fig_ov)


# =============================================================================
# TAB 2: RISK METRICS
# =============================================================================
with tab_metrics:
    st.subheader("📈 Key Risk & Downside Metrics")
    st.markdown("Comprehensive risk measures evaluated from the empirical portfolio return distribution.")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        st.metric(label="Annualized Volatility", value=f"{ann_volatility * 100:.2f}%")
        st.markdown(
            '<div class="metric-caption">Dispersion of daily returns scaled by $\\sqrt{252}$. Higher volatility signifies greater price fluctuation.</div></div>',
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        st.metric(label="Sharpe Ratio (Ann.)", value=f"{sharpe_ratio:.2f}")
        st.markdown(
            f'<div class="metric-caption">Excess return generated per unit of total risk above the {risk_free_rate*100:.1f}% risk-free rate. Values &gt; 1.0 indicate attractive risk-adjusted return.</div></div>',
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        st.metric(label="1D 95% Historical VaR (%)", value=f"{var_pct * 100:.2f}%")
        st.markdown(
            '<div class="metric-caption">The 5th percentile cutoff: on 19 out of 20 trading days, daily loss will not exceed this threshold.</div></div>',
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        st.metric(label="1D 95% Historical VaR ($)", value=f"${var_dollars:,.2f}")
        st.markdown(
            f'<div class="metric-caption">Maximum estimated 1-day dollar loss on your ${portfolio_value:,.0f} portfolio under normal market conditions (95% confidence).</div></div>',
            unsafe_allow_html=True,
        )

    with m_col5:
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        st.metric(
            label="Expected Shortfall (CVaR)",
            value=f"{cvar_pct * 100:.2f}%",
            delta=f"${cvar_dollars:,.2f}",
            delta_color="inverse",
        )
        st.markdown(
            '<div class="metric-caption"><b>Conditional VaR:</b> Answers "if a bad day happens, how bad on average is it," complementing VaR which only marks the boundary.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Key Takeaway & Risk Interpretation")
    # Use \\$ to prevent Streamlit from treating dollar amounts as LaTeX math delimiters
    st.info(
        f"\u2022 **VaR vs. CVaR:** While your 1-day 95% VaR is **{var_pct * 100:.2f}%** "
        f"(\\${var_dollars:,.2f}), if the market breaks below this threshold into the worst "
        f"5% tail, the expected average loss increases to **{cvar_pct * 100:.2f}%** "
        f"(\\${cvar_dollars:,.2f})."
    )


# =============================================================================
# TAB 3: VISUALIZATIONS
# =============================================================================
with tab_viz:
    st.subheader("📊 Performance & Risk Visualizations")

    v_row1_col1, v_row1_col2 = st.columns(2)

    with v_row1_col1:
        st.markdown("#### 📈 Compounded Cumulative Return")
        cum_ret_series = calculate_cumulative_returns(portfolio_daily_returns)
        fig_c, ax_c = plt.subplots(figsize=(6.8, 4.2), dpi=100)
        ax_c.set_facecolor("#161B22")
        fig_c.patch.set_facecolor("#0D1117")
        ax_c.plot(cum_ret_series.index, cum_ret_series * 100, color="#00B4D8", linewidth=2.0, label="Portfolio")
        ax_c.axhline(0, color="#8B949E", linestyle="--", linewidth=0.8)
        ax_c.fill_between(cum_ret_series.index, cum_ret_series * 100, 0, where=(cum_ret_series >= 0), color="#00B4D8", alpha=0.18)
        ax_c.fill_between(cum_ret_series.index, cum_ret_series * 100, 0, where=(cum_ret_series < 0), color="#EF476F", alpha=0.18)
        ax_c.set_ylabel("Cumulative Return (%)", color="#E6EDF3", fontsize=9, fontweight="bold")
        ax_c.tick_params(colors="#8B949E", labelsize=8)
        ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}%"))
        ax_c.grid(True, linestyle=":", alpha=0.3, color="#30363D")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)
        st.caption(f"💡 Total cumulative growth across selected period: **{cum_ret_series.iloc[-1] * 100:+.2f}%**.")

    with v_row1_col2:
        st.markdown("#### 📉 Daily Return Histogram with VaR & CVaR")
        fig_h, ax_h = plt.subplots(figsize=(6.8, 4.2), dpi=100)
        ax_h.set_facecolor("#161B22")
        fig_h.patch.set_facecolor("#0D1117")
        
        var_cutoff_val = -var_pct * 100
        cvar_cutoff_val = -cvar_pct * 100

        sns.histplot(
            portfolio_daily_returns * 100,
            kde=True,
            bins=35,
            color="#00B4D8",
            ax=ax_h,
            stat="density",
            edgecolor="#30363D",
            linewidth=0.5,
        )

        # VaR threshold line
        ax_h.axvline(
            var_cutoff_val,
            color="#FFB703",
            linestyle="--",
            linewidth=2.0,
            label=f"95% VaR Cutoff ({var_cutoff_val:.2f}%)",
        )

        # CVaR threshold line
        ax_h.axvline(
            cvar_cutoff_val,
            color="#EF476F",
            linestyle="-.",
            linewidth=2.0,
            label=f"95% CVaR Avg ({cvar_cutoff_val:.2f}%)",
        )

        # Highlight CVaR tail region
        if ax_h.get_lines():
            kde_x = ax_h.get_lines()[0].get_xdata()
            kde_y = ax_h.get_lines()[0].get_ydata()
            tail_mask = kde_x <= var_cutoff_val
            ax_h.fill_between(
                kde_x[tail_mask],
                kde_y[tail_mask],
                color="#EF476F",
                alpha=0.35,
                label="Worst 5% Tail Region (CVaR)",
            )

        ax_h.set_xlabel("Daily Return (%)", color="#E6EDF3", fontsize=9, fontweight="bold")
        ax_h.set_ylabel("Density", color="#E6EDF3", fontsize=9, fontweight="bold")
        ax_h.tick_params(colors="#8B949E", labelsize=8)
        ax_h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax_h.grid(True, linestyle=":", alpha=0.3, color="#30363D")
        ax_h.legend(loc="upper right", fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)
        st.caption(
            "💡 **Histogram & Tail Risk:** The yellow dashed line marks the 95% VaR cutoff. "
            f"The red line marks the Expected Shortfall (CVaR) average of **{cvar_pct * 100:.2f}%** inside the tail."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔗 Asset Return Correlation Heatmap")
    corr_matrix = calculate_correlation_matrix(daily_returns_df)
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax_corr.set_facecolor("#161B22")
    fig_corr.patch.set_facecolor("#0D1117")
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        linecolor="#30363D",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax_corr,
        annot_kws={"size": 10, "weight": "bold", "color": "#E6EDF3"},
    )
    ax_corr.tick_params(colors="#E6EDF3", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    st.caption("💡 Lower or negative pairwise correlations provide stronger diversification benefits to reduce total portfolio variance.")


# =============================================================================
# TAB 4: CLUSTERING
# =============================================================================
with tab_cluster:
    st.subheader(f"🎯 Unsupervised Asset Risk/Return Clustering (k={k_clusters})")
    st.markdown(
        "K-Means segments your assets into distinct risk/reward profiles using annualized return and annualized volatility features."
    )

    cl_col1, cl_col2 = st.columns([3, 2])

    with cl_col1:
        fig_cl, ax_cl = plt.subplots(figsize=(6.8, 4.5), dpi=100)
        ax_cl.set_facecolor("#161B22")
        fig_cl.patch.set_facecolor("#0D1117")

        cluster_palette = {
            "Cluster 1": "#00B4D8",
            "Cluster 2": "#EF476F",
            "Cluster 3": "#06D6A0",
        }

        for cluster_name, group in cluster_df.groupby("Cluster"):
            color = cluster_palette.get(cluster_name, "#FFD166")
            ax_cl.scatter(
                group["Annualized Volatility"] * 100,
                group["Annualized Return"] * 100,
                s=180,
                color=color,
                label=cluster_name,
                alpha=0.9,
                edgecolors="white",
                linewidth=1.2,
                zorder=4,
            )

            for _, row in group.iterrows():
                ax_cl.annotate(
                    row["Ticker"],
                    (row["Annualized Volatility"] * 100, row["Annualized Return"] * 100),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color="#E6EDF3",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#21262D", alpha=0.85, edgecolor="#30363D"),
                )

        ax_cl.set_xlabel("Annualized Volatility (%) [Risk]", color="#E6EDF3", fontsize=9, fontweight="bold")
        ax_cl.set_ylabel("Annualized Return (%) [Reward]", color="#E6EDF3", fontsize=9, fontweight="bold")
        ax_cl.tick_params(colors="#8B949E", labelsize=8)
        ax_cl.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax_cl.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}%"))
        ax_cl.grid(True, linestyle=":", alpha=0.3, color="#30363D")
        ax_cl.legend(loc="best", fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
        plt.tight_layout()
        st.pyplot(fig_cl)
        plt.close(fig_cl)

        st.caption(
            "💡 **K-Means Scatter Plot:** Groups holdings into distinct risk/return regimes "
            "(e.g., lower risk/stable return vs. higher risk/growth assets)."
        )

    with cl_col2:
        st.markdown("#### 📋 Asset Regime Breakdown")
        table_display = cluster_df.copy()
        table_display["Weight"] = [
            f"{active_weights[active_tickers.index(t)] * 100:.1f}%" if t in active_tickers else "0.0%"
            for t in table_display["Ticker"]
        ]
        table_display["Ann. Return"] = table_display["Annualized Return"].apply(lambda r: f"{r * 100:+.2f}%")
        table_display["Ann. Volatility"] = table_display["Annualized Volatility"].apply(lambda v: f"{v * 100:.2f}%")

        st.dataframe(
            table_display[["Ticker", "Weight", "Ann. Return", "Ann. Volatility", "Cluster"]],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# TAB 5: BREACH PREDICTION (SUPERVISED ML)
# =============================================================================
with tab_breach:
    st.subheader(f"🤖 Supervised VaR Breach Classifier ({breach_model_type})")
    st.markdown(
        "Predicts whether a given trading day will experience a **high-risk breach** (daily portfolio return falling below the 95% Historical VaR threshold) "
        "using backward-looking features (Rolling 20D Volatility, 5D Momentum, and 20D Volume Trend)."
    )

    if not classification_success:
        st.error(f"⚠️ Classification error: {classification_error}")
    else:
        st.markdown(
            f"""
            <div class="section-banner">
                Model: <b>{breach_model_type}</b> | Time-Series Split: <b>80% Train ({len(X_train)} days) / 20% Test ({len(X_test)} days)</b> 
                | Actual Test Breaches: <b>{model_metrics['test_breaches']} days</b> out of {model_metrics['total_test_samples']} test days.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics row
        ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
        ev_col1.metric("Precision", f"{model_metrics['precision'] * 100:.1f}%")
        ev_col2.metric("Recall (Sensitivity)", f"{model_metrics['recall'] * 100:.1f}%")
        ev_col3.metric("F1-Score", f"{model_metrics['f1']:.2f}")
        ev_col4.metric("Test Accuracy", f"{model_metrics['accuracy'] * 100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        br_col1, br_col2 = st.columns(2)

        with br_col1:
            st.markdown("#### 🔲 Test Set Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.8), dpi=100)
            ax_cm.set_facecolor("#161B22")
            fig_cm.patch.set_facecolor("#0D1117")

            cm_labels = ["Normal (0)", "VaR Breach (1)"]
            sns.heatmap(
                model_metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=cm_labels,
                yticklabels=cm_labels,
                ax=ax_cm,
                cbar=False,
                annot_kws={"size": 11, "weight": "bold"},
            )
            ax_cm.set_xlabel("Predicted Label", color="#E6EDF3", fontsize=9, fontweight="bold")
            ax_cm.set_ylabel("Actual Label", color="#E6EDF3", fontsize=9, fontweight="bold")
            ax_cm.tick_params(colors="#E6EDF3", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with br_col2:
            st.markdown(f"#### 📊 {feature_importance_df['Type'].iloc[0]}")
            fig_fi, ax_fi = plt.subplots(figsize=(5.5, 3.8), dpi=100)
            ax_fi.set_facecolor("#161B22")
            fig_fi.patch.set_facecolor("#0D1117")

            bars = ax_fi.barh(
                feature_importance_df["Feature"],
                feature_importance_df["Importance"],
                color="#00B4D8",
                edgecolor="white",
                linewidth=0.8,
            )
            ax_fi.set_xlabel("Relative Importance / Weight", color="#E6EDF3", fontsize=9, fontweight="bold")
            ax_fi.tick_params(colors="#E6EDF3", labelsize=8)
            ax_fi.grid(True, linestyle=":", alpha=0.3, color="#30363D")
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

            if breach_model_type == "Random Forest":
                st.caption(
                    "💡 **Feature Importance (MDI):** Measures how much each backward-looking feature contributed "
                    "to the tree splits when anticipating tail risk. *Note: Reflects historical empirical association, not direct causality.*"
                )
            else:
                st.caption(
                    "💡 **Logistic Regression Coefficients:** Positive coefficients increase the log-odds of a tail breach, "
                    "while negative coefficients reduce the likelihood."
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "⚠️ **Class Imbalance Consideration:** Because VaR breaches are naturally rare tail events (~5% of days), "
            "the test set contains very few positive examples. In risk management, **Recall** is prioritized to avoid "
            "failing to detect catastrophic downturn days (minimizing False Negatives)."
        )


# -----------------------------------------------------------------------------
# CONSOLIDATED METHODOLOGY EXPANDER
# -----------------------------------------------------------------------------
st.divider()
with st.expander("📖 Complete Methodology & Financial Mathematics Guide"):
    st.markdown(
        """
        ### 1. Return Calculations
        - **Daily Return:** $R_{i,t} = \\frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$
        - **Portfolio Weighted Return:** $R_{p,t} = \\sum_{i=1}^{N} w_i \\cdot R_{i,t}$
        - **Annualized Return:** $\\mu_{\\text{ann}} = \\mu_{\\text{daily}} \\times 252$
        - **Annualized Volatility:** $\\sigma_{\\text{ann}} = \\sigma_{\\text{daily}} \\times \\sqrt{252}$

        ---

        ### 2. Risk Metrics
        - **Sharpe Ratio:** $\\text{Sharpe} = \\frac{\\mu_{\\text{ann}} - R_f}{\\sigma_{\\text{ann}}}$, where $R_f$ is the annual risk-free rate.
        - **1-Day 95% Historical Value at Risk (VaR):** The 5th percentile cutoff of the historical daily return distribution.
        - **1-Day 95% Expected Shortfall (CVaR):** The expected value (mean) of all daily returns that fall at or below the VaR cutoff:
          $$\\text{CVaR} = \\mathbb{E}[R_p \\mid R_p \\le \\text{VaR}_{95\\%}]$$

        ---

        ### 3. Machine Learning Components
        - **Unsupervised K-Means Clustering:** Groups assets in 2D space $(\\sigma_{\\text{ann}}, \\mu_{\\text{ann}})$ by minimizing within-cluster variance without labels.
        - **Supervised VaR Breach Classifier:** Predicts binary breach events ($y_t = 1$ if $R_{p,t} \\le \\text{VaR}_{95\\%}$) using three trailing features:
          1. *Rolling 20-Day Volatility* (trailing standard deviation scaled to annual).
          2. *5-Day Compounded Momentum* (trailing 5-day return).
          3. *20-Day Volume Trend* (% change vs. 20-day moving average volume).
          *Strictly trained with chronological time-series splitting (80/20) to eliminate look-ahead bias.*
        """
    )
