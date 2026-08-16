# 📊 Portfolio Risk Analyzer

A clean, production-ready Python web application built with **Streamlit** that allows investors to analyze stock portfolios, calculate key risk metrics, evaluate downside risk via Historical Value at Risk (VaR), and discover asset risk/return regimes using unsupervised K-Means clustering.

---

## 🚀 Live Demo & Quickstart

### Prerequisites
- Python 3.11+
- Git

### 1. Local Setup
Clone the repository (or navigate to the project directory) and install dependencies:

```bash
# Clone repository
git clone https://github.com/yourusername/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Running the App Locally
Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Community Cloud

This project is configured for one-click deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub (ensure `app.py`, `risk_metrics.py`, `clustering.py`, and `requirements.txt` are at the repository root).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your GitHub repository, branch (`main`), and set the main file path to `app.py`.
5. Click **"Deploy"**! Streamlit will automatically install dependencies from `requirements.txt` and serve the app.

---

## 🏗️ Architecture & Codebase Structure

The project strictly decouples financial mathematics and machine learning from UI rendering to ensure clean unit-testability and maintainability:

```text
├── app.py              # Streamlit web interface, inputs, charts, and layout
├── risk_metrics.py     # Pure financial mathematical functions (no UI dependencies)
├── clustering.py       # Unsupervised K-Means clustering module (scikit-learn)
├── test_analyzer.py    # Automated unit test suite
├── requirements.txt    # Pinned dependencies for local and cloud environments
└── README.md           # Project documentation and financial theory guide
```

---

## 📐 Mathematical Foundations & Financial Methodology

### 1. Daily Asset Returns & Portfolio Return Series
For each asset $i$ at day $t$, daily percentage return is calculated from adjusted close prices $P$:
$$R_{i, t} = \frac{P_{i, t} - P_{i, t-1}}{P_{i, t-1}}$$

For a portfolio with asset weights $\mathbf{w} = [w_1, w_2, \dots, w_N]^T$ where $\sum_{i=1}^N w_i = 1$:
$$R_{p, t} = \sum_{i=1}^{N} w_i \cdot R_{i, t}$$

---

### 2. Annualized Volatility ($\sigma_{\text{ann}}$)
Volatility measures the dispersion and uncertainty of returns.
1. Daily standard deviation $\sigma_{\text{daily}}$ is calculated with $N-1$ degrees of freedom.
2. Under the standard assumption of independent and identically distributed (i.i.d.) returns across 252 annual US trading days, variance scales linearly with time: $\sigma^2_{\text{ann}} = 252 \times \sigma^2_{\text{daily}}$.
3. Taking the square root gives the **Square Root of Time Rule**:
$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$$

---

### 3. Annualized Sharpe Ratio
The Sharpe ratio quantifies the excess return generated per unit of total risk compared to a risk-free benchmark $R_f$ (default: 4.0% / 0.04):
$$\mu_{\text{ann}} = \mu_{\text{daily}} \times 252$$
$$\text{Sharpe Ratio} = \frac{\mu_{\text{ann}} - R_f}{\sigma_{\text{ann}}}$$

- **Interpretation**:
  - $< 1.0$: Sub-optimal risk-adjusted return.
  - $1.0 - 1.99$: Good risk-adjusted return.
  - $\ge 2.0$: Very good to excellent risk-adjusted performance.

---

### 4. 1-Day 95% Historical Value at Risk (VaR)
**Historical Value at Risk** is a non-parametric risk measure that estimates downside loss without assuming a normal (Gaussian) distribution, capturing real market fat tails and skewness.

- **Method**: The empirical daily portfolio returns are sorted in ascending order.
- At 95% confidence, the 1-day VaR corresponds to the **5th percentile** ($\alpha = 0.05$) of the historical distribution:
$$\text{VaR}_{\text{cutoff}} = \text{Percentile}_{5\%}(R_p)$$
$$\text{VaR}_{\%} = |\text{VaR}_{\text{cutoff}}| \quad (\text{if } \text{VaR}_{\text{cutoff}} < 0)$$
$$\text{VaR}_{\$} = \text{VaR}_{\%} \times \text{Portfolio Value}$$

- **Plain-English Meaning**: On 95% of trading days (19 out of 20 days), your 1-day portfolio loss is not expected to exceed $\text{VaR}_{\%}$ ($\text{VaR}_{\$}$).

---

### 5. Asset Return Correlation Matrix
Measures the linear co-movement between pairs of assets $X$ and $Y$:
$$\rho_{X, Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, +1]$$

- **Diversification Insight**: Assets with lower or negative correlations mitigate overall portfolio variance according to Markowitz Modern Portfolio Theory.

---

### 6. Unsupervised K-Means Clustering

#### Why K-Means?
K-Means partitions $N$ assets into $k$ distinct risk/reward clusters by minimizing within-cluster variance (sum of squared Euclidean distances to cluster centroids):
$$\min_{\mathbf{S}} \sum_{j=1}^{k} \sum_{\mathbf{x} \in S_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|^2$$

#### Why These Two Features? (Annualized Return & Annualized Volatility)
In financial economics, assets are fundamentally evaluated on the two-dimensional tradeoff between **expected return** (reward) and **volatility** (risk). Clustering in this $(\sigma_{\text{ann}}, \mu_{\text{ann}})$ feature space segments holdings into intuitive categories:
- **Low Risk / Stable Return**: Defensive assets, utilities, dividend-yielding equities (e.g., JPM, XOM in specific market cycles).
- **High Risk / High Return**: High-beta growth and tech assets (e.g., NVDA, TSLA, AAPL).
- **High Risk / Low Return**: Underperforming high-volatility holdings that drag down the Sharpe ratio.

#### Why Unsupervised?
Because risk regimes are intrinsic structures in asset dynamics rather than labeled targets, unsupervised clustering allows the data to speak for itself without arbitrary classification thresholds or overfitting risks.

---

## 🧪 Running Automated Unit Tests

A comprehensive unit test suite is included in `test_analyzer.py`:

```bash
python -m unittest test_analyzer.py -v
```

This verifies:
- Daily return calculations and shape matching.
- Portfolio return dot-product weighting.
- Mathematical precision of annualized volatility, return, and Sharpe ratio.
- 5th percentile empirical VaR calculations (percentage and dollar amounts).
- Pearson correlation matrix properties (identity diagonal, symmetry).
- Compounded cumulative return series.
- Scikit-learn K-Means feature extraction and cluster assignment.

---

## 🛠️ Tech Stack
- **UI Framework**: [Streamlit](https://streamlit.io/)
- **Financial Data**: [yfinance](https://github.com/ranaroussi/yfinance)
- **Data Analysis**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (KMeans)
- **Visualizations**: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
