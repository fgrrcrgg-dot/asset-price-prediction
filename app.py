"""
Thai Equity Price Prediction Dashboard
Academic project — NOT financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings, itertools

# ── Optional heavy imports (graceful fallback) ──────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
TICKERS = {
    "PTT (PTT-R.BK)": "PTT-R.BK",
    "Airports of Thailand (AOT.BK)": "AOT.BK",
    "Advanced Info Service (ADVANC.BK)": "ADVANC.BK",
    "CP ALL (CPALL.BK)": "CPALL.BK",
    "SCB X (SCB.BK)": "SCB.BK",
}
BENCHMARK = "^SET.BK"
START_DATE = "2020-01-01"
TIMESPAN_MAP = {
    "1M": 21, "6M": 126, "YTD": None, "1Y": 252,
    "2Y": 504, "5Y": 1260, "10Y": 2520, "20Y": 5040,
}
TRADING_DAYS = 252
RF_RATE = 0.02  # risk-free proxy

# ════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Thai Equity Prediction Model", layout="wide", page_icon="📈")

# ── Custom CSS to mimic the Black Swan dashboard ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Source Sans Pro', sans-serif;
}
h1 { font-weight: 700; letter-spacing: -0.5px; }
h2 { font-weight: 600; color: #31333F; }
h3 { font-weight: 600; color: #31333F; }

div[data-testid="stMetric"] {
    background: #f8f9fb;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label {
    font-size: 0.82rem !important;
    color: #808495 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

div.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
div.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: #f8f9fb;
    border-right: 1px solid #e6e9ef;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.15rem;
}

.disclaimer-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #664d03;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Fetching market data …", ttl=3600)
def load_data(tickers: list[str], start: str) -> pd.DataFrame:
    """Download adjusted-close prices for all tickers from yfinance."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all")
    return prices


def trim_to_span(df: pd.DataFrame, span_key: str) -> pd.DataFrame:
    """Trim dataframe to a given timespan."""
    if span_key == "YTD":
        start = pd.Timestamp(datetime.now().year, 1, 1)
        return df.loc[df.index >= start]
    n = TIMESPAN_MAP[span_key]
    if n is None:
        return df
    return df.iloc[-min(n, len(df)):]


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def compute_metrics(prices: pd.DataFrame, ticker: str) -> dict:
    """Compute annualised return, vol, Sharpe, RSI, MAs for a single ticker."""
    s = prices[ticker].dropna()
    lr = log_returns(s)
    ann_ret = lr.mean() * TRADING_DAYS
    ann_vol = lr.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol != 0 else np.nan

    # RSI-14
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if len(rsi) > 0 else np.nan

    ma5 = s.rolling(5).mean().iloc[-1]
    ma20 = s.rolling(20).mean().iloc[-1]
    last_price = s.iloc[-1]

    return {
        "Last Price": f"{last_price:,.2f}",
        "Ann. Return": f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "RSI (14)": f"{current_rsi:.1f}",
        "5-Day MA": f"{ma5:,.2f}",
        "20-Day MA": f"{ma20:,.2f}",
    }


# ════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════
def random_walk_forecast(log_ret: pd.Series, last_price: float, horizon: int = 60):
    """Geometric random walk: drift + noise."""
    mu = log_ret.mean()
    sigma = log_ret.std()
    np.random.seed(42)
    shocks = np.random.normal(mu, sigma, horizon)
    path = last_price * np.exp(np.cumsum(shocks))
    return path, {"Drift (μ daily)": f"{mu:.6f}", "Volatility (σ daily)": f"{sigma:.6f}"}


def arima_forecast(series: pd.Series, horizon: int = 60):
    """ARIMA on log-returns, select (p,d,q) by AIC grid search."""
    lr = log_returns(series).dropna()
    best_aic, best_order = np.inf, (1, 0, 1)
    for p, q in itertools.product(range(0, 4), range(0, 4)):
        try:
            m = ARIMA(lr.values, order=(p, 0, q))
            res = m.fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, 0, q)
        except Exception:
            continue

    model = ARIMA(lr.values, order=best_order).fit()
    fc = model.forecast(steps=horizon)
    pred_prices = series.iloc[-1] * np.exp(np.cumsum(fc))
    stats = {
        "Best Order (p,d,q)": str(best_order),
        "AIC": f"{best_aic:.2f}",
        "Log-likelihood": f"{model.llf:.2f}",
    }
    return pred_prices, stats


def garch_forecast(series: pd.Series, horizon: int = 60):
    """GARCH(1,1) volatility forecast on log-returns."""
    lr = log_returns(series).dropna() * 100  # scale for arch lib
    am = arch_model(lr, vol="Garch", p=1, q=1, mean="AR", lags=1, rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon)

    mean_fc = fc.mean.iloc[-1].values / 100  # back to decimal
    var_fc = fc.variance.iloc[-1].values / 10000
    np.random.seed(42)
    shocks = np.random.normal(mean_fc, np.sqrt(var_fc))
    pred_prices = series.iloc[-1] * np.exp(np.cumsum(shocks))

    stats = {
        "ω": f"{res.params.get('omega', 0):.6f}",
        "α₁": f"{res.params.get('alpha[1]', 0):.6f}",
        "β₁": f"{res.params.get('beta[1]', 0):.6f}",
        "Log-likelihood": f"{res.loglikelihood:.2f}",
    }
    return pred_prices, stats


def correlation_stats(prices: pd.DataFrame, ticker: str) -> dict:
    """Correlation and beta between ticker and SET index."""
    both = prices[[ticker, BENCHMARK]].dropna()
    lr = np.log(both / both.shift(1)).dropna()
    corr = lr[ticker].corr(lr[BENCHMARK])
    beta = lr[ticker].cov(lr[BENCHMARK]) / lr[BENCHMARK].var()
    r2 = corr ** 2
    return {
        "Correlation": f"{corr:.4f}",
        "Beta": f"{beta:.4f}",
        "R²": f"{r2:.4f}",
    }


# ════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Source Sans Pro, sans-serif", size=13),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def price_chart(df: pd.Series, title: str, color: str = "#1f77b4") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df.values, mode="lines",
        line=dict(color=color, width=2), name=title,
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title="Price (THB)", **CHART_LAYOUT)
    return fig


def dual_chart(bench: pd.Series, stock: pd.Series, bench_name: str, stock_name: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=bench.index, y=bench.values, mode="lines",
        line=dict(color="#636EFA", width=2), name=bench_name,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=stock.index, y=stock.values, mode="lines",
        line=dict(color="#EF553B", width=2), name=stock_name,
    ), secondary_y=True)
    fig.update_layout(
        title=f"{stock_name} vs {bench_name}",
        **CHART_LAYOUT,
    )
    fig.update_yaxes(title_text=bench_name, secondary_y=False)
    fig.update_yaxes(title_text=stock_name, secondary_y=True)
    return fig


def prediction_chart(
    hist_bench, hist_stock, pred_prices, future_dates,
    bench_name, stock_name, model_label
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # historical benchmark
    fig.add_trace(go.Scatter(
        x=hist_bench.index, y=hist_bench.values, mode="lines",
        line=dict(color="#636EFA", width=1.5), name=f"{bench_name} (hist)",
        opacity=0.6,
    ), secondary_y=False)
    # historical stock
    fig.add_trace(go.Scatter(
        x=hist_stock.index, y=hist_stock.values, mode="lines",
        line=dict(color="#EF553B", width=1.5), name=f"{stock_name} (hist)",
        opacity=0.6,
    ), secondary_y=True)
    # predicted stock
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_prices, mode="lines",
        line=dict(color="#00CC96", width=2.5, dash="dot"), name=f"Forecast ({model_label})",
    ), secondary_y=True)
    fig.update_layout(
        title=f"{model_label} — {stock_name} vs {bench_name}",
        **CHART_LAYOUT,
    )
    fig.update_yaxes(title_text=bench_name, secondary_y=False)
    fig.update_yaxes(title_text=stock_name, secondary_y=True)
    return fig


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    selected_label = st.selectbox("**Select Asset**", list(TICKERS.keys()))
    selected_ticker = TICKERS[selected_label]
    forecast_horizon = st.slider("Forecast horizon (trading days)", 10, 120, 60, step=5)

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Built for an **academic project** on equity price prediction. "
        "The models do *not* predict exact future prices — they illustrate "
        "different statistical approaches (Random Walk, ARIMA, GARCH) "
        "applied to the Thai stock market (SET Index)."
    )
    st.markdown(
        '<div class="disclaimer-box">'
        "⚠️ <b>Disclaimer</b> — This dashboard is for <b>educational and academic purposes only</b>. "
        "It does not constitute financial advice. Past performance does not guarantee future results. "
        "The creators accept no liability for any trading decisions based on this tool."
        "</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════
all_tickers = [BENCHMARK] + list(TICKERS.values())
prices = load_data(all_tickers, START_DATE)

# ════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════
st.markdown("# 📈 Thai Equity Price Prediction Model")
st.markdown(
    "**An academic dashboard for the Thai stock market (SET Index)** — "
    "comparing Random Walk, ARIMA, and GARCH forecasts against the benchmark."
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — SET INDEX OVERVIEW
# ════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · SET Index Overview")

set_prices = prices[BENCHMARK].dropna()
set_lr = log_returns(set_prices)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close", f"{set_prices.iloc[-1]:,.2f}")
col2.metric("Ann. Return", f"{set_lr.mean() * TRADING_DAYS:.2%}")
col3.metric("Ann. Volatility", f"{set_lr.std() * np.sqrt(TRADING_DAYS):.2%}")
col4.metric("Data Points", f"{len(set_prices):,}")

span_keys_1 = list(TIMESPAN_MAP.keys())
tabs_1 = st.tabs(span_keys_1)
for tab, sk in zip(tabs_1, span_keys_1):
    with tab:
        trimmed = trim_to_span(set_prices.to_frame(), sk)[BENCHMARK]
        st.plotly_chart(price_chart(trimmed, f"SET Index ({sk})"), use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — ASSET METRICS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 2 · Asset Overview — {selected_label}")

stock_prices = prices[selected_ticker].dropna()

# Metric cards
s_lr = log_returns(stock_prices)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Last Close", f"{stock_prices.iloc[-1]:,.2f}")
ann_r = s_lr.mean() * TRADING_DAYS
mc2.metric("Ann. Return", f"{ann_r:.2%}")
ann_v = s_lr.std() * np.sqrt(TRADING_DAYS)
mc3.metric("Ann. Volatility", f"{ann_v:.2%}")
sharpe = (ann_r - RF_RATE) / ann_v if ann_v else 0
mc4.metric("Sharpe Ratio", f"{sharpe:.3f}")

# Full table for all listed assets
rows = {}
for label, tkr in TICKERS.items():
    if tkr in prices.columns:
        rows[label] = compute_metrics(prices, tkr)
metrics_df = pd.DataFrame(rows).T
metrics_df.index.name = "Asset"
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 3 · Price Predictions — {selected_label} vs SET Index")

span_keys_3 = list(TIMESPAN_MAP.keys())
pred_span = st.radio("Historical window for prediction charts", span_keys_3, horizontal=True, index=3)

hist_bench = trim_to_span(set_prices.to_frame(), pred_span)[BENCHMARK]
hist_stock = trim_to_span(stock_prices.to_frame(), pred_span)[selected_ticker]

# Shared future dates
last_date = hist_stock.index[-1]
future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_horizon)

# Correlation stats (shared across models)
corr_info = correlation_stats(prices, selected_ticker)

if not HAS_MODELS:
    st.error(
        "⚠️ `statsmodels` and/or `arch` are not installed. "
        "Install them (`pip install statsmodels arch`) to enable ARIMA & GARCH forecasts."
    )

# ── 3a  Random Walk ────────────────────────────────────────────────────
st.markdown("### 3.1 · Random Walk (Geometric Brownian Motion)")
rw_pred, rw_stats = random_walk_forecast(log_returns(stock_prices), stock_prices.iloc[-1], forecast_horizon)

c_chart, c_stats = st.columns([3, 1])
with c_chart:
    st.plotly_chart(
        prediction_chart(hist_bench, hist_stock, rw_pred, future_dates,
                         "SET Index", selected_ticker, "Random Walk"),
        use_container_width=True,
    )
with c_stats:
    st.markdown("**Model Statistics**")
    for k, v in rw_stats.items():
        st.markdown(f"- **{k}:** `{v}`")
    st.markdown("**Correlation with SET**")
    for k, v in corr_info.items():
        st.markdown(f"- **{k}:** `{v}`")

# ── 3b  ARIMA ──────────────────────────────────────────────────────────
st.markdown("### 3.2 · ARIMA (Best AIC Order)")
if HAS_MODELS:
    with st.spinner("Fitting ARIMA — searching (p,d,q) grid …"):
        arima_pred, arima_stats = arima_forecast(stock_prices, forecast_horizon)
    c_chart2, c_stats2 = st.columns([3, 1])
    with c_chart2:
        st.plotly_chart(
            prediction_chart(hist_bench, hist_stock, arima_pred, future_dates,
                             "SET Index", selected_ticker, "ARIMA"),
            use_container_width=True,
        )
    with c_stats2:
        st.markdown("**Model Statistics**")
        for k, v in arima_stats.items():
            st.markdown(f"- **{k}:** `{v}`")
        st.markdown("**Correlation with SET**")
        for k, v in corr_info.items():
            st.markdown(f"- **{k}:** `{v}`")
else:
    st.info("ARIMA model unavailable — install `statsmodels`.")

# ── 3c  GARCH ──────────────────────────────────────────────────────────
st.markdown("### 3.3 · GARCH(1,1) — Volatility Clustering")
if HAS_MODELS:
    with st.spinner("Fitting GARCH(1,1) …"):
        garch_pred, garch_stats = garch_forecast(stock_prices, forecast_horizon)
    c_chart3, c_stats3 = st.columns([3, 1])
    with c_chart3:
        st.plotly_chart(
            prediction_chart(hist_bench, hist_stock, garch_pred, future_dates,
                             "SET Index", selected_ticker, "GARCH(1,1)"),
            use_container_width=True,
        )
    with c_stats3:
        st.markdown("**Model Statistics**")
        for k, v in garch_stats.items():
            st.markdown(f"- **{k}:** `{v}`")
        st.markdown("**Correlation with SET**")
        for k, v in corr_info.items():
            st.markdown(f"- **{k}:** `{v}`")
else:
    st.info("GARCH model unavailable — install `arch`.")

# ════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <b>Academic Disclaimer</b> — This tool is created solely for educational and academic purposes. "
    "None of the outputs constitute investment advice. The statistical models shown have well-known "
    "limitations (see academic literature on the Efficient Market Hypothesis). Always consult a licensed "
    "financial professional before making investment decisions. Data sourced from Yahoo Finance."
    "</div>",
    unsafe_allow_html=True,
)
st.caption("© 2026 · Academic Project · Data via Yahoo Finance · Built with Streamlit")
