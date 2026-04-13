# 📈 Thai Equity Price Prediction Model

An academic dashboard for the Thai stock market (SET Index) that compares four statistical and machine-learning forecast models — Random Walk, ARIMA, GARCH, and XGBoost — side by side against the benchmark.

Built with [Streamlit](https://streamlit.io/) · Data via [Yahoo Finance](https://finance.yahoo.com/)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Deploying to Streamlit Cloud

Push the following structure to a GitHub repo and connect it to [share.streamlit.io](https://share.streamlit.io):

```
your-repo/
├── .streamlit/
│   └── config.toml      # Forces light theme with #0070FF blue accent
├── app.py
├── requirements.txt
└── README.md
```

---

## Dashboard Sections

### Section 1 — SET Index Overview

Displays the SET Index (^SET.BK) with a selectable timespan (1M, 6M, YTD, 1Y, 2Y, 5Y, 10Y, 20Y). The four metric cards — Last Close, Annualised Return, Annualised Volatility, and Data Points — all recompute dynamically based on the chosen window.

### Section 2 — Asset Overview

A comprehensive metrics table covering all five listed equities. Includes:

- **Last Price**
- **Annualised Return** (logarithmic)
- **Annualised Volatility** (standard deviation)
- **Sharpe Ratio** (risk-free rate = 2%)
- **RSI (14)** — Relative Strength Index with a 14-day rolling window
- **5-Day Moving Average**
- **20-Day Moving Average**

All values adjust automatically to the timespan selected in Section 3.

### Section 3 — Price Predictions

Four separate dual-axis charts showing the selected stock against the SET Index, each with an independent timespan selector. Every model reports its parameters and **RMSFE** (Root Mean Square Forecast Error) on a held-out test split. A correlation panel shows the Pearson correlation, Beta, and R² between the stock and the SET Index.

| Model | Description |
|-------|-------------|
| **Random Walk** | Geometric Brownian Motion with historical drift (μ) and volatility (σ) estimated from log-returns. |
| **ARIMA(p, 0, q)** | Grid search over p ∈ {0–3}, q ∈ {0–3} with d = 0 (log-returns are already stationary). Selects by lowest AIC. Reports AIC, BIC, and log-likelihood. |
| **GARCH(p, q)** | Grid search over p ∈ {1–3}, q ∈ {1–3}. Selects by lowest BIC. Captures volatility clustering. Reports ω, α, β coefficients and log-likelihood. |
| **XGBoost** | Gradient-boosted trees trained on 20 lagged log-return features. 500 estimators, depth 5, early stopping on a 20% validation split. Reports validation RMSE and MAE on returns. |

---

## Listed Assets

| Company | Ticker |
|---------|--------|
| PTT Public Company Limited | `PTT-R.BK` |
| Airports of Thailand | `AOT.BK` |
| Advanced Info Service | `ADVANC.BK` |
| CP ALL Public Company Limited | `CPALL.BK` |
| SCB X Public Company Limited | `SCB.BK` |

Benchmark: **SET Index** (`^SET.BK`)

---

## Methodology

### Data

- **Source**: Yahoo Finance via `yfinance` (adjusted close prices)
- **Period**: January 1, 2000 — present
- **Returns**: Logarithmic returns: `r_t = ln(P_t / P_{t-1})`
- **Risk-free rate**: 2% annualised (proxy)
- **Trading days**: 252 per year

### Model Selection

- ARIMA order is chosen by AIC across a (p, 0, q) grid. The differencing parameter d is fixed at 0 because log-returns are already approximately stationary (the price series is I(1), so its log-return is I(0)).
- GARCH order is chosen by BIC across a (p, q) grid, fitting volatility dynamics on scaled log-returns.
- XGBoost uses early stopping on a validation set to prevent overfitting.

### Evaluation

All models report **RMSFE** (Root Mean Square Forecast Error) computed on a single train/test split where the last *h* observations (equal to the forecast horizon) are held out.

---

## Performance & Caching

All model-fitting functions are wrapped in `@st.cache_data` with a 1-hour TTL. Once a model is fitted for a given asset and horizon, switching timespans or revisiting the page does not re-trigger computation. A JavaScript snippet also runs client-side to enforce the blue slider colour across Streamlit re-renders.

---

## Tech Stack

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `yfinance` | Market data download |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | ARIMA modelling |
| `arch` | GARCH modelling |
| `xgboost` | Gradient-boosted tree forecasting |
| `scikit-learn` | Validation metrics (RMSE, MAE) |

---

## Disclaimer

> ⚠️ **This dashboard is for educational and academic purposes only.** It does not constitute financial advice. The statistical models shown have well-known limitations (see academic literature on the Efficient Market Hypothesis). Past performance does not guarantee future results. Always consult a licensed financial professional before making investment decisions.

---

© 2026 · Academic Project · Built with Streamlit
