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
START_DATE = "2000-01-01"
TIMESPAN_MAP = {
    "1M": 21, "6M": 126, "YTD": None, "1Y": 252,
    "2Y": 504, "5Y": 1260, "10Y": 2520, "20Y": 5040,
}
TRADING_DAYS = 252
RF_RATE = 0.02  # risk-free proxy

# Blue accent palette
BLUE_PRIMARY = "#2E86DE"
BLUE_DARK = "#1B5FAA"
BLUE_LIGHT = "#D6EAFF"
ACCENT_RED = "#E74C3C"
ACCENT_GREEN = "#27AE60"

# ════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Thai Equity Prediction Model", layout="wide", page_icon="📈")

# ── Custom CSS — light theme with blue accents ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Source Sans Pro', sans-serif;
}

/* ── Force light background ── */
.stApp {
    background-color: #FFFFFF !important;
    color: #31333F !important;
}

h1 { font-weight: 700; letter-spacing: -0.5px; color: #1a1a2e !important; }
h2 { font-weight: 600; color: #31333F !important; }
h3 { font-weight: 600; color: #31333F !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #f8f9fb;
    border: 1px solid #e0e5ec;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    font-size: 0.8rem !important;
    color: #808495 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* ── Tabs — blue active state ── */
div.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e0e5ec;
}
div.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    color: #808495;
}
div.stTabs [aria-selected="true"] {
    color: #2E86DE !important;
    border-bottom: 3px solid #2E86DE !important;
    background: transparent !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f8f9fb !important;
    border-right: 1px solid #e0e5ec;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1a1a2e !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #4a4a5a !important;
}

/* ── Selectbox / dropdown — blue border ── */
div[data-baseweb="select"] > div {
    border-color: #2E86DE !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
}

/* ── Slider — blue track & thumb ── */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #2E86DE !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #2E86DE !important;
}

/* ── Radio buttons — blue when selected ── */
div.stRadio > div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    color: #31333F !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #e0e5ec;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Dividers ── */
hr {
    border-color: #e0e5ec !important;
}

/* ── Disclaimer box ── */
.disclaimer-box {
    background: #FFF8E1;
    border-left: 4px solid #FFB300;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #5D4037;
    margin-top: 24px;
    line-height: 1.55;
}

/* ── Stats card in prediction section ── */
.stats-card {
    background: #f8f9fb;
    border: 1px solid #e0e5ec;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.stats-card h4 {
    color: #2E86DE !important;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    border-bottom: 2px solid #D6EAFF;
    padding-bottom: 6px;
}
.stats-card p {
    margin: 4px 0;
    font-size: 0.88rem;
    color: #31333F;
}
.stats-card .label {
    color: #808495;
    font-weight: 400;
}
.stats-card .value {
    font-weight: 700;
    color: #1a1a2e;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Fetching market data …", ttl=3600)
def load_data(tickers: list, start: str) -> pd.DataFrame:
    """Download adjusted-close prices for all tickers from yfinance.
    Handles multiple yfinance column formats robustly."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    if raw.empty:
        st.error("No data returned from Yahoo Finance. Check your internet connection.")
        st.stop()

    # yfinance >= 0.2.31 may return MultiIndex (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        # Try to extract Close prices
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            # Flatten and take first level
            prices = raw.iloc[
                :,
                raw.columns.get_level_values(0) == raw.columns.get_level_values(0)[0],
            ].copy()
            prices.columns = prices.columns.droplevel(0)
    else:
        # Single ticker fallback
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            if len(tickers) == 1:
                prices.columns = tickers
        else:
            prices = raw.copy()

    # Ensure all requested tickers are present (fill missing with NaN)
    for t in tickers:
        if t not in prices.columns:
            prices[t] = np.nan

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


def safe_last(series: pd.Series, default=np.nan):
    """Safely get the last value of a series, skipping NaN."""
    if series is None or len(series) == 0:
        return default
    clean = series.dropna()
    if len(clean) == 0:
        return default
    return clean.iloc[-1]


def compute_metrics(prices: pd.DataFrame, ticker: str) -> dict:
    """Compute annualised return, vol, Sharpe, RSI, MAs for a single ticker.
    Returns 'N/A' for any metric that can't be computed."""
    na_row = {
        "Last Price": "N/A",
        "Ann. Return": "N/A",
        "Ann. Volatility": "N/A",
        "Sharpe Ratio": "N/A",
        "RSI (14)": "N/A",
        "5-Day MA": "N/A",
        "20-Day MA": "N/A",
    }

    if ticker not in prices.columns:
        return na_row

    s = prices[ticker].dropna()
    if len(s) < 25:  # Need at least 25 data points for meaningful metrics
        return na_row

    lr = log_returns(s)
    if len(lr) < 2:
        return na_row

    ann_ret = lr.mean() * TRADING_DAYS
    ann_vol = lr.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol != 0 else np.nan

    # RSI-14
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = safe_last(rsi)

    ma5 = safe_last(s.rolling(5).mean())
    ma20 = safe_last(s.rolling(20).mean())
    last_price = s.iloc[-1]

    def fmt(v, spec):
        try:
            if pd.isna(v):
                return "N/A"
            return f"{v:{spec}}"
        except (ValueError, TypeError):
            return "N/A"

    return {
        "Last Price": fmt(last_price, ",.2f"),
        "Ann. Return": fmt(ann_ret, ".2%"),
        "Ann. Volatility": fmt(ann_vol, ".2%"),
        "Sharpe Ratio": fmt(sharpe, ".3f"),
        "RSI (14)": fmt(current_rsi, ".1f"),
        "5-Day MA": fmt(ma5, ",.2f"),
        "20-Day MA": fmt(ma20, ",.2f"),
    }


# ════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════
def random_walk_forecast(log_ret: pd.Series, last_price: float, horizon: int = 60):
    """Geometric random walk: drift + noise."""
    if len(log_ret) < 2:
        return np.full(horizon, last_price), {
            "Drift (μ daily)": "N/A",
            "Volatility (σ daily)": "N/A",
        }
    mu = log_ret.mean()
    sigma = log_ret.std()
    np.random.seed(42)
    shocks = np.random.normal(mu, sigma, horizon)
    path = last_price * np.exp(np.cumsum(shocks))
    return path, {
        "Drift (μ daily)": f"{mu:.6f}",
        "Volatility (σ daily)": f"{sigma:.6f}",
    }


def arima_forecast(series: pd.Series, horizon: int = 60):
    """ARIMA on log-returns, select (p,d,q) by AIC grid search."""
    lr = log_returns(series).dropna()
    if len(lr) < 30:
        return np.full(horizon, series.iloc[-1]), {"Error": "Insufficient data"}

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
    if len(lr) < 30:
        return np.full(horizon, series.iloc[-1]), {"Error": "Insufficient data"}

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
    if ticker not in prices.columns or BENCHMARK not in prices.columns:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}

    both = prices[[ticker, BENCHMARK]].dropna()
    if len(both) < 10:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}

    lr = np.log(both / both.shift(1)).dropna()
    if len(lr) < 10:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}

    corr = lr[ticker].corr(lr[BENCHMARK])
    cov_val = lr[ticker].cov(lr[BENCHMARK])
    var_val = lr[BENCHMARK].var()
    beta = cov_val / var_val if var_val != 0 else np.nan
    r2 = corr**2 if not pd.isna(corr) else np.nan

    return {
        "Correlation": f"{corr:.4f}" if not pd.isna(corr) else "N/A",
        "Beta": f"{beta:.4f}" if not pd.isna(beta) else "N/A",
        "R²": f"{r2:.4f}" if not pd.isna(r2) else "N/A",
    }


# ════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Source Sans Pro, sans-serif", size=13, color="#31333F"),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
)


def price_chart(df: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.values,
            mode="lines",
            line=dict(color=BLUE_PRIMARY, width=2),
            name=title,
            fill="tozeroy",
            fillcolor="rgba(46,134,222,0.08)",
            hovertemplate="%{x|%d %b %Y}<br>Price: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis_title="Price (THB)", **CHART_LAYOUT)
    fig.update_xaxes(gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=False)
    return fig


def prediction_chart(
    hist_bench,
    hist_stock,
    pred_prices,
    future_dates,
    bench_name,
    stock_name,
    model_label,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # historical benchmark
    fig.add_trace(
        go.Scatter(
            x=hist_bench.index,
            y=hist_bench.values,
            mode="lines",
            line=dict(color=BLUE_PRIMARY, width=1.5),
            name=f"{bench_name} (hist)",
            opacity=0.6,
        ),
        secondary_y=False,
    )
    # historical stock
    fig.add_trace(
        go.Scatter(
            x=hist_stock.index,
            y=hist_stock.values,
            mode="lines",
            line=dict(color=ACCENT_RED, width=1.5),
            name=f"{stock_name} (hist)",
            opacity=0.6,
        ),
        secondary_y=True,
    )
    # predicted stock
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=pred_prices,
            mode="lines",
            line=dict(color=ACCENT_GREEN, width=2.5, dash="dot"),
            name=f"Forecast ({model_label})",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{model_label} — {stock_name} vs {bench_name}", **CHART_LAYOUT
    )
    fig.update_yaxes(
        title_text=bench_name, secondary_y=False, gridcolor="#f0f0f0"
    )
    fig.update_yaxes(
        title_text=stock_name, secondary_y=True, gridcolor="#f0f0f0"
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    return fig


def render_stats_card(title: str, stats: dict):
    """Render a styled stats card using HTML."""
    rows = "".join(
        f'<p><span class="label">{k}:</span> <span class="value">{v}</span></p>'
        for k, v in stats.items()
    )
    st.markdown(
        f'<div class="stats-card"><h4>{title}</h4>{rows}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    selected_label = st.selectbox("**Select Asset**", list(TICKERS.keys()))
    selected_ticker = TICKERS[selected_label]
    forecast_horizon = st.slider(
        "Forecast horizon (trading days)", 10, 120, 60, step=5
    )

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
        "⚠️ <b>Disclaimer</b> — This dashboard is for <b>educational and "
        "academic purposes only</b>. It does not constitute financial advice. "
        "Past performance does not guarantee future results. The creators "
        "accept no liability for any trading decisions based on this tool."
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

set_series = (
    prices[BENCHMARK].dropna()
    if BENCHMARK in prices.columns
    else pd.Series(dtype=float)
)

if len(set_series) < 2:
    st.warning(
        "⚠️ SET Index data unavailable. Check your data connection."
    )
else:
    set_lr = log_returns(set_series)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Close", f"{set_series.iloc[-1]:,.2f}")
    col2.metric("Ann. Return", f"{set_lr.mean() * TRADING_DAYS:.2%}")
    col3.metric(
        "Ann. Volatility", f"{set_lr.std() * np.sqrt(TRADING_DAYS):.2%}"
    )
    col4.metric("Data Points", f"{len(set_series):,}")

    span_keys_1 = list(TIMESPAN_MAP.keys())
    tabs_1 = st.tabs(span_keys_1)
    for tab, sk in zip(tabs_1, span_keys_1):
        with tab:
            trimmed = trim_to_span(set_series.to_frame(), sk)[
                BENCHMARK
            ].dropna()
            if len(trimmed) > 0:
                st.plotly_chart(
                    price_chart(trimmed, f"SET Index ({sk})"),
                    use_container_width=True,
                )
            else:
                st.info(f"Not enough data for {sk} window.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — ASSET METRICS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 2 · Asset Overview — {selected_label}")

stock_series = (
    prices[selected_ticker].dropna()
    if selected_ticker in prices.columns
    else pd.Series(dtype=float)
)

if len(stock_series) >= 25:
    s_lr = log_returns(stock_series)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Last Close", f"{stock_series.iloc[-1]:,.2f}")
    ann_r = s_lr.mean() * TRADING_DAYS
    mc2.metric("Ann. Return", f"{ann_r:.2%}")
    ann_v = s_lr.std() * np.sqrt(TRADING_DAYS)
    mc3.metric("Ann. Volatility", f"{ann_v:.2%}")
    sharpe_val = (ann_r - RF_RATE) / ann_v if ann_v else 0
    mc4.metric("Sharpe Ratio", f"{sharpe_val:.3f}")
else:
    st.warning(
        f"⚠️ Insufficient price data for **{selected_label}**. "
        "The ticker may be delisted or renamed on Yahoo Finance."
    )

# Full table for all listed assets
rows = {}
for label, tkr in TICKERS.items():
    rows[label] = compute_metrics(prices, tkr)
metrics_df = pd.DataFrame(rows).T
metrics_df.index.name = "Asset"
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 3 · Price Predictions — {selected_label} vs SET Index")

if len(stock_series) < 30 or len(set_series) < 30:
    st.error(
        "⚠️ Not enough data for the selected asset to run prediction models."
    )
else:
    span_keys_3 = list(TIMESPAN_MAP.keys())
    pred_span = st.radio(
        "Historical window for prediction charts",
        span_keys_3,
        horizontal=True,
        index=3,
    )

    hist_bench = trim_to_span(set_series.to_frame(), pred_span)[
        BENCHMARK
    ].dropna()
    hist_stock = trim_to_span(stock_series.to_frame(), pred_span)[
        selected_ticker
    ].dropna()

    if len(hist_stock) < 2 or len(hist_bench) < 2:
        st.warning("Not enough data in the selected window.")
    else:
        # Shared future dates
        last_date = hist_stock.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + timedelta(days=1), periods=forecast_horizon
        )

        # Correlation stats (shared across models)
        corr_info = correlation_stats(prices, selected_ticker)

        if not HAS_MODELS:
            st.error(
                "⚠️ `statsmodels` and/or `arch` are not installed. "
                "Install them (`pip install statsmodels arch`) to enable "
                "ARIMA & GARCH forecasts."
            )

        # ── 3a  Random Walk ────────────────────────────────────────
        st.markdown("### 3.1 · Random Walk (Geometric Brownian Motion)")
        rw_pred, rw_stats = random_walk_forecast(
            log_returns(stock_series),
            stock_series.iloc[-1],
            forecast_horizon,
        )

        c_chart, c_stats = st.columns([3, 1])
        with c_chart:
            st.plotly_chart(
                prediction_chart(
                    hist_bench,
                    hist_stock,
                    rw_pred,
                    future_dates,
                    "SET Index",
                    selected_ticker,
                    "Random Walk",
                ),
                use_container_width=True,
            )
        with c_stats:
            render_stats_card("Model Parameters", rw_stats)
            render_stats_card("Correlation with SET", corr_info)

        # ── 3b  ARIMA ──────────────────────────────────────────────
        st.markdown("### 3.2 · ARIMA (Best AIC Order)")
        if HAS_MODELS:
            with st.spinner("Fitting ARIMA — searching (p,d,q) grid …"):
                try:
                    arima_pred, arima_stats = arima_forecast(
                        stock_series, forecast_horizon
                    )
                except Exception as e:
                    arima_pred = np.full(
                        forecast_horizon, stock_series.iloc[-1]
                    )
                    arima_stats = {"Error": str(e)[:80]}

            c_chart2, c_stats2 = st.columns([3, 1])
            with c_chart2:
                st.plotly_chart(
                    prediction_chart(
                        hist_bench,
                        hist_stock,
                        arima_pred,
                        future_dates,
                        "SET Index",
                        selected_ticker,
                        "ARIMA",
                    ),
                    use_container_width=True,
                )
            with c_stats2:
                render_stats_card("Model Parameters", arima_stats)
                render_stats_card("Correlation with SET", corr_info)
        else:
            st.info("ARIMA model unavailable — install `statsmodels`.")

        # ── 3c  GARCH ──────────────────────────────────────────────
        st.markdown("### 3.3 · GARCH(1,1) — Volatility Clustering")
        if HAS_MODELS:
            with st.spinner("Fitting GARCH(1,1) …"):
                try:
                    garch_pred, garch_stats = garch_forecast(
                        stock_series, forecast_horizon
                    )
                except Exception as e:
                    garch_pred = np.full(
                        forecast_horizon, stock_series.iloc[-1]
                    )
                    garch_stats = {"Error": str(e)[:80]}

            c_chart3, c_stats3 = st.columns([3, 1])
            with c_chart3:
                st.plotly_chart(
                    prediction_chart(
                        hist_bench,
                        hist_stock,
                        garch_pred,
                        future_dates,
                        "SET Index",
                        selected_ticker,
                        "GARCH(1,1)",
                    ),
                    use_container_width=True,
                )
            with c_stats3:
                render_stats_card("Model Parameters", garch_stats)
                render_stats_card("Correlation with SET", corr_info)
        else:
            st.info("GARCH model unavailable — install `arch`.")

# ════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div class="disclaimer-box">'
    "⚠️ <b>Academic Disclaimer</b> — This tool is created solely for "
    "educational and academic purposes. None of the outputs constitute "
    "investment advice. The statistical models shown have well-known "
    "limitations (see academic literature on the Efficient Market "
    "Hypothesis). Always consult a licensed financial professional before "
    "making investment decisions. Data sourced from Yahoo Finance."
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "© 2026 · Academic Project · Data via Yahoo Finance · Built with Streamlit"
)
