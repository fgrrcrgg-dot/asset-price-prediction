# 📈 Thai Equity Price Prediction Model

An academic dashboard for the Thai stock market (SET Index) that compares four forecast models — Random Walk, ARIMA, GARCH, and XGBoost — with backtested predictions and forward forecasts.

Built with [Streamlit](https://streamlit.io/) · Data via [Yahoo Finance](https://finance.yahoo.com/)

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. For Streamlit Cloud, push this structure:

```
your-repo/
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
└── README.md
```

---

## Dashboard Sections

### Section 1 — SET Index Overview

Interactive chart of ^SET.BK with a selectable timespan (1M → 20Y). Five metric cards — Last Close, CAGR, Annualised Return, Annualised Volatility, Data Points — all recompute based on the chosen window.

### Section 2 — Asset Overview

Comprehensive metrics table for all five equities, synced to the Section 4 timespan:

- **Last Price**, **CAGR**, **Annualised Return** (log), **Annualised Volatility**, **Sharpe Ratio** (rf = 2%), **RSI (14)**, **5-Day MA**, **20-Day MA**

### Section 3 — SET Index Forecast

Four models forecasting the SET Index itself. Each chart shows:

- **Historical line** — actual SET prices over the selected window
- **Backtest line** (purple dashed) — model predictions over the held-out test period
- **Forecast line** (green/orange dotted) — future predictions beyond the last observation

The stats panel on the right shows model parameters, RMSFE on the backtest, and the backtest date range — all updating with the chosen timespan.

### Section 4 — Stock Forecast vs SET Index

Same four models applied to the selected stock, plotted on dual axes against the SET benchmark. Includes correlation, beta, and R² computed on the selected window. Backtest and forecast lines work identically to Section 3.

---

## Models

| Model | Description |
|-------|-------------|
| **Random Walk** | Geometric Brownian Motion with drift (μ) and volatility (σ) from log-returns. |
| **ARIMA(p, 0, q)** | Grid search p ∈ {0–3}, q ∈ {0–3}, d = 0 (log-returns are I(0)). Selects by AIC. |
| **GARCH(p, q)** | Grid search p ∈ {1–3}, q ∈ {1–3}. Selects by BIC. Captures volatility clustering. |
| **XGBoost** | 500 gradient-boosted trees on 20 lagged log-return features. Early stopping on 20% validation. |

### Backtest Methodology

Each model is evaluated via a train/test split: the last *h* observations (equal to the forecast horizon) are held out. The model is trained on everything before that, then predicts the held-out period. **RMSFE** (Root Mean Square Forecast Error) measures price-level accuracy. The backtest line is plotted in purple alongside the actual prices so you can visually assess fit.

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

- **Source**: Yahoo Finance (adjusted close), January 2000 – present
- **Returns**: Logarithmic: `r_t = ln(P_t / P_{t-1})`
- **CAGR**: `(P_end / P_start)^(1/years) − 1`
- **Risk-free rate**: 2% annualised (proxy)
- **Trading days**: 252/year
- **Correlation & Beta**: Computed on the trimmed window selected by the user, not the full dataset

---

## Performance & Caching

All model functions use `@st.cache_data` (1-hour TTL). Switching timespans or assets reuses cached fits. The backtest split retrains once per asset+horizon combination.

---

## Tech Stack

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `yfinance` | Market data |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data wrangling |
| `statsmodels` | ARIMA |
| `arch` | GARCH |
| `xgboost` | Gradient-boosted trees |
| `scikit-learn` | Validation metrics |

---

## Disclaimer

> ⚠️ **Educational and academic purposes only.** Not financial advice. Models have well-known limitations. Past performance ≠ future results. Consult a licensed professional.

---

© 2026 · Academic Project · Built with Streamlit
