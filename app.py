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
    HAS_TIMESERIES = True
except ImportError:
    HAS_TIMESERIES = False

try:
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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

# ── Colour palette ─────────────────────────────────────────────────────
BLUE = "#0070FF"
BLUE_LIGHT = "#D6EAFF"
ACCENT_RED = "#E74C3C"
ACCENT_GREEN = "#27AE60"
ACCENT_ORANGE = "#F39C12"
TEXT_BLACK = "#1a1a2e"

# ════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Thai Equity Prediction Model", layout="wide", page_icon="📈"
)

# ── Custom CSS — light theme, #0070FF blue everywhere ──────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Source Sans Pro', sans-serif;
}

/* ── Force light background everywhere ── */
.stApp, .main, .block-container {
    background-color: #FFFFFF !important;
    color: #1a1a2e !important;
}

h1 { font-weight: 700; letter-spacing: -0.5px; color: #1a1a2e !important; }
h2 { font-weight: 600; color: #1a1a2e !important; }
h3 { font-weight: 600; color: #1a1a2e !important; }
p, li, span { color: #1a1a2e; }

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
    color: #606475 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* ── Tabs — ALL blue (active tab highlight) ── */
div.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e0e5ec;
}
div.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    color: #808495 !important;
    background: transparent !important;
}
div.stTabs [aria-selected="true"] {
    color: #0070FF !important;
    border-bottom: 3px solid #0070FF !important;
    background: transparent !important;
}
/* Override any built-in red/pink highlight Streamlit adds */
div.stTabs [data-baseweb="tab-highlight"] {
    background-color: #0070FF !important;
}
div.stTabs [data-baseweb="tab-border"] {
    background-color: #e0e5ec !important;
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
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #3a3a4a !important;
}

/* ── Selectbox / dropdown — blue border ── */
div[data-baseweb="select"] > div {
    border-color: #0070FF !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
}

/* ── Slider — ALL blue: track, active track, thumb ── */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #0070FF !important;
    border-color: #0070FF !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #0070FF !important;
}
/* Track (filled portion) */
div[data-testid="stSlider"] div[role="progressbar"],
div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
    background-color: #0070FF !important;
}

/* ── Radio buttons ── */
div.stRadio label {
    color: #1a1a2e !important;
}

/* ── Dataframe / table — white bg, black text ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #e0e5ec;
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stDataFrame"] * {
    color: #1a1a2e !important;
}
div[data-testid="stDataFrame"] table {
    background: #FFFFFF !important;
}
div[data-testid="stDataFrame"] th {
    background: #f0f2f6 !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
}
div[data-testid="stDataFrame"] td {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
}
/* Glide (the dataframe renderer Streamlit uses) */
div[data-testid="stDataFrame"] .glideDataEditor,
div[data-testid="stDataFrame"] canvas + div {
    background: #FFFFFF !important;
}

/* ── Dividers ── */
hr { border-color: #e0e5ec !important; }

/* ── Disclaimer box ── */
.disclaimer-box {
    background: #FFF8E1;
    border-left: 4px solid #FFB300;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #5D4037 !important;
    margin-top: 24px;
    line-height: 1.55;
}

/* ── Stats card ── */
.stats-card {
    background: #f8f9fb;
    border: 1px solid #e0e5ec;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.stats-card h4 {
    color: #0070FF !important;
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
    color: #1a1a2e !important;
}
.stats-card .label { color: #606475 !important; font-weight: 400; }
.stats-card .value {
    font-weight: 700;
    color: #1a1a2e !important;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Fetching market data …", ttl=3600)
def load_data(tickers: list, start: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.iloc[
                :,
                raw.columns.get_level_values(0)
                == raw.columns.get_level_values(0)[0],
            ].copy()
            prices.columns = prices.columns.droplevel(0)
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            if len(tickers) == 1:
                prices.columns = tickers
        else:
            prices = raw.copy()
    for t in tickers:
        if t not in prices.columns:
            prices[t] = np.nan
    return prices.dropna(how="all")


def trim_to_span(df: pd.DataFrame, span_key: str) -> pd.DataFrame:
    if span_key == "YTD":
        return df.loc[df.index >= pd.Timestamp(datetime.now().year, 1, 1)]
    n = TIMESPAN_MAP[span_key]
    if n is None:
        return df
    return df.iloc[-min(n, len(df)) :]


def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1)).dropna()


def safe_last(s: pd.Series, default=np.nan):
    if s is None or len(s) == 0:
        return default
    c = s.dropna()
    return c.iloc[-1] if len(c) else default


def compute_metrics(prices: pd.DataFrame, ticker: str) -> dict:
    na = {
        "Last Price": "N/A", "Ann. Return": "N/A", "Ann. Volatility": "N/A",
        "Sharpe Ratio": "N/A", "RSI (14)": "N/A", "5-Day MA": "N/A",
        "20-Day MA": "N/A",
    }
    if ticker not in prices.columns:
        return na
    s = prices[ticker].dropna()
    if len(s) < 25:
        return na
    lr = log_returns(s)
    if len(lr) < 2:
        return na
    ann_ret = lr.mean() * TRADING_DAYS
    ann_vol = lr.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol else np.nan
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    def f(v, sp):
        try:
            return "N/A" if pd.isna(v) else f"{v:{sp}}"
        except (ValueError, TypeError):
            return "N/A"

    return {
        "Last Price": f(s.iloc[-1], ",.2f"),
        "Ann. Return": f(ann_ret, ".2%"),
        "Ann. Volatility": f(ann_vol, ".2%"),
        "Sharpe Ratio": f(sharpe, ".3f"),
        "RSI (14)": f(safe_last(rsi), ".1f"),
        "5-Day MA": f(safe_last(s.rolling(5).mean()), ",.2f"),
        "20-Day MA": f(safe_last(s.rolling(20).mean()), ",.2f"),
    }


# ════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ════════════════════════════════════════════════════════════════════════
def random_walk_forecast(lr: pd.Series, last: float, h: int = 60):
    if len(lr) < 2:
        return np.full(h, last), {"Drift (μ daily)": "N/A", "Volatility (σ daily)": "N/A"}
    mu, sigma = lr.mean(), lr.std()
    np.random.seed(42)
    path = last * np.exp(np.cumsum(np.random.normal(mu, sigma, h)))
    return path, {"Drift (μ daily)": f"{mu:.6f}", "Volatility (σ daily)": f"{sigma:.6f}"}


def arima_forecast(series: pd.Series, h: int = 60):
    lr = log_returns(series).dropna()
    if len(lr) < 30:
        return np.full(h, series.iloc[-1]), {"Error": "Insufficient data"}
    best_aic, best_order = np.inf, (1, 0, 1)
    for p, q in itertools.product(range(4), range(4)):
        try:
            res = ARIMA(lr.values, order=(p, 0, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, 0, q)
        except Exception:
            continue
    model = ARIMA(lr.values, order=best_order).fit()
    fc = model.forecast(steps=h)
    return series.iloc[-1] * np.exp(np.cumsum(fc)), {
        "Best Order (p,d,q)": str(best_order),
        "AIC": f"{best_aic:.2f}",
        "Log-likelihood": f"{model.llf:.2f}",
    }


def garch_forecast(series: pd.Series, h: int = 60):
    lr = log_returns(series).dropna() * 100
    if len(lr) < 30:
        return np.full(h, series.iloc[-1]), {"Error": "Insufficient data"}
    am = arch_model(lr, vol="Garch", p=1, q=1, mean="AR", lags=1, rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=h)
    mean_fc = fc.mean.iloc[-1].values / 100
    var_fc = fc.variance.iloc[-1].values / 10000
    np.random.seed(42)
    shocks = np.random.normal(mean_fc, np.sqrt(var_fc))
    return series.iloc[-1] * np.exp(np.cumsum(shocks)), {
        "ω": f"{res.params.get('omega', 0):.6f}",
        "α₁": f"{res.params.get('alpha[1]', 0):.6f}",
        "β₁": f"{res.params.get('beta[1]', 0):.6f}",
        "Log-likelihood": f"{res.loglikelihood:.2f}",
    }


def xgboost_forecast(series: pd.Series, h: int = 60, n_lags: int = 20):
    """XGBoost regression on lagged log-return features."""
    lr = log_returns(series).dropna()
    if len(lr) < n_lags + 50:
        return np.full(h, series.iloc[-1]), {"Error": "Insufficient data"}

    # Build feature matrix: each row = [ret_{t-1}, ret_{t-2}, …, ret_{t-n_lags}]
    data = pd.DataFrame({"ret": lr.values})
    for lag in range(1, n_lags + 1):
        data[f"lag_{lag}"] = data["ret"].shift(lag)
    data = data.dropna().reset_index(drop=True)

    X = data.drop(columns=["ret"]).values
    y = data["ret"].values

    # Train/test split (last 20% for validation stats)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))

    # Recursive multi-step forecast
    recent = list(lr.values[-n_lags:])
    preds = []
    for _ in range(h):
        feat = np.array(recent[-n_lags:][::-1]).reshape(1, -1)  # most recent first
        pred_ret = float(model.predict(feat)[0])
        preds.append(pred_ret)
        recent.append(pred_ret)

    pred_prices = series.iloc[-1] * np.exp(np.cumsum(preds))
    stats = {
        "Lags": str(n_lags),
        "n_estimators": "300",
        "max_depth": "4",
        "Val RMSE": f"{rmse:.6f}",
        "Val MAE": f"{mae:.6f}",
    }
    return pred_prices, stats


def correlation_stats(prices: pd.DataFrame, ticker: str) -> dict:
    if ticker not in prices.columns or BENCHMARK not in prices.columns:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}
    both = prices[[ticker, BENCHMARK]].dropna()
    if len(both) < 10:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}
    lr = np.log(both / both.shift(1)).dropna()
    if len(lr) < 10:
        return {"Correlation": "N/A", "Beta": "N/A", "R²": "N/A"}
    corr = lr[ticker].corr(lr[BENCHMARK])
    beta = lr[ticker].cov(lr[BENCHMARK]) / lr[BENCHMARK].var()
    r2 = corr**2 if not pd.isna(corr) else np.nan
    return {
        "Correlation": f"{corr:.4f}" if not pd.isna(corr) else "N/A",
        "Beta": f"{beta:.4f}" if not pd.isna(beta) else "N/A",
        "R²": f"{r2:.4f}" if not pd.isna(r2) else "N/A",
    }


# ════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (all axis / title labels → black)
# ════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Source Sans Pro, sans-serif", size=13, color="#1a1a2e"),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color="#1a1a2e"),
    ),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    title_font=dict(color="#1a1a2e"),
)

AXIS_STYLE = dict(
    gridcolor="#f0f0f0", zeroline=False,
    title_font=dict(color="#1a1a2e"),
    tickfont=dict(color="#1a1a2e"),
)


def price_chart(df: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df.values, mode="lines",
        line=dict(color=BLUE, width=2), name=title,
        fill="tozeroy", fillcolor="rgba(0,112,255,0.07)",
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title="Price (THB)", **CHART_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def prediction_chart(hist_bench, hist_stock, pred_prices, future_dates,
                     bench_name, stock_name, model_label, line_color=ACCENT_GREEN):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=hist_bench.index, y=hist_bench.values, mode="lines",
        line=dict(color=BLUE, width=1.5), name=f"{bench_name} (hist)", opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=hist_stock.index, y=hist_stock.values, mode="lines",
        line=dict(color=ACCENT_RED, width=1.5), name=f"{stock_name} (hist)", opacity=0.6,
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_prices, mode="lines",
        line=dict(color=line_color, width=2.5, dash="dot"),
        name=f"Forecast ({model_label})",
    ), secondary_y=True)
    fig.update_layout(
        title=f"{model_label} — {stock_name} vs {bench_name}", **CHART_LAYOUT
    )
    fig.update_yaxes(title_text=bench_name, secondary_y=False, **AXIS_STYLE)
    fig.update_yaxes(title_text=stock_name, secondary_y=True, **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)
    return fig


def render_stats_card(title: str, stats: dict):
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
    forecast_horizon = st.slider("Forecast horizon (trading days)", 10, 120, 60, step=5)

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Built for an **academic project** on equity price prediction. "
        "The models do *not* predict exact future prices — they illustrate "
        "different statistical approaches (Random Walk, ARIMA, GARCH, XGBoost) "
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
    "comparing Random Walk, ARIMA, GARCH, and XGBoost forecasts against the benchmark."
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — SET INDEX OVERVIEW
# ════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · SET Index Overview")

set_series = (
    prices[BENCHMARK].dropna() if BENCHMARK in prices.columns
    else pd.Series(dtype=float)
)

if len(set_series) < 2:
    st.warning("⚠️ SET Index data unavailable.")
else:
    set_lr = log_returns(set_series)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{set_series.iloc[-1]:,.2f}")
    c2.metric("Ann. Return", f"{set_lr.mean() * TRADING_DAYS:.2%}")
    c3.metric("Ann. Volatility", f"{set_lr.std() * np.sqrt(TRADING_DAYS):.2%}")
    c4.metric("Data Points", f"{len(set_series):,}")

    for tab, sk in zip(st.tabs(list(TIMESPAN_MAP)), TIMESPAN_MAP):
        with tab:
            trimmed = trim_to_span(set_series.to_frame(), sk)[BENCHMARK].dropna()
            if len(trimmed) > 0:
                st.plotly_chart(price_chart(trimmed, f"SET Index ({sk})"), use_container_width=True)
            else:
                st.info(f"Not enough data for {sk} window.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — ASSET METRICS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 2 · Asset Overview — {selected_label}")

stock_series = (
    prices[selected_ticker].dropna() if selected_ticker in prices.columns
    else pd.Series(dtype=float)
)

if len(stock_series) >= 25:
    s_lr = log_returns(stock_series)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close", f"{stock_series.iloc[-1]:,.2f}")
    ar = s_lr.mean() * TRADING_DAYS
    m2.metric("Ann. Return", f"{ar:.2%}")
    av = s_lr.std() * np.sqrt(TRADING_DAYS)
    m3.metric("Ann. Volatility", f"{av:.2%}")
    m4.metric("Sharpe Ratio", f"{(ar - RF_RATE) / av if av else 0:.3f}")
else:
    st.warning(f"⚠️ Insufficient price data for **{selected_label}**.")

rows = {lbl: compute_metrics(prices, t) for lbl, t in TICKERS.items()}
mdf = pd.DataFrame(rows).T
mdf.index.name = "Asset"
st.dataframe(mdf, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"## 3 · Price Predictions — {selected_label} vs SET Index")

if len(stock_series) < 30 or len(set_series) < 30:
    st.error("⚠️ Not enough data for prediction models.")
else:
    pred_span = st.radio(
        "Historical window for prediction charts",
        list(TIMESPAN_MAP), horizontal=True, index=3,
    )
    hist_bench = trim_to_span(set_series.to_frame(), pred_span)[BENCHMARK].dropna()
    hist_stock = trim_to_span(stock_series.to_frame(), pred_span)[selected_ticker].dropna()

    if len(hist_stock) < 2 or len(hist_bench) < 2:
        st.warning("Not enough data in the selected window.")
    else:
        last_date = hist_stock.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
        corr_info = correlation_stats(prices, selected_ticker)

        if not HAS_TIMESERIES:
            st.error("⚠️ `statsmodels` / `arch` not installed for ARIMA & GARCH.")

        # ── 3.1  Random Walk ───────────────────────────────────────
        st.markdown("### 3.1 · Random Walk (Geometric Brownian Motion)")
        rw_pred, rw_stats = random_walk_forecast(
            log_returns(stock_series), stock_series.iloc[-1], forecast_horizon
        )
        cc, cs = st.columns([3, 1])
        with cc:
            st.plotly_chart(prediction_chart(
                hist_bench, hist_stock, rw_pred, future_dates,
                "SET Index", selected_ticker, "Random Walk",
            ), use_container_width=True)
        with cs:
            render_stats_card("Model Parameters", rw_stats)
            render_stats_card("Correlation with SET", corr_info)

        # ── 3.2  ARIMA ─────────────────────────────────────────────
        st.markdown("### 3.2 · ARIMA (Best AIC Order)")
        if HAS_TIMESERIES:
            with st.spinner("Fitting ARIMA — grid search …"):
                try:
                    arima_pred, arima_stats = arima_forecast(stock_series, forecast_horizon)
                except Exception as e:
                    arima_pred = np.full(forecast_horizon, stock_series.iloc[-1])
                    arima_stats = {"Error": str(e)[:80]}
            cc2, cs2 = st.columns([3, 1])
            with cc2:
                st.plotly_chart(prediction_chart(
                    hist_bench, hist_stock, arima_pred, future_dates,
                    "SET Index", selected_ticker, "ARIMA",
                ), use_container_width=True)
            with cs2:
                render_stats_card("Model Parameters", arima_stats)
                render_stats_card("Correlation with SET", corr_info)
        else:
            st.info("ARIMA unavailable — install `statsmodels`.")

        # ── 3.3  GARCH ─────────────────────────────────────────────
        st.markdown("### 3.3 · GARCH(1,1) — Volatility Clustering")
        if HAS_TIMESERIES:
            with st.spinner("Fitting GARCH(1,1) …"):
                try:
                    garch_pred, garch_stats = garch_forecast(stock_series, forecast_horizon)
                except Exception as e:
                    garch_pred = np.full(forecast_horizon, stock_series.iloc[-1])
                    garch_stats = {"Error": str(e)[:80]}
            cc3, cs3 = st.columns([3, 1])
            with cc3:
                st.plotly_chart(prediction_chart(
                    hist_bench, hist_stock, garch_pred, future_dates,
                    "SET Index", selected_ticker, "GARCH(1,1)",
                ), use_container_width=True)
            with cs3:
                render_stats_card("Model Parameters", garch_stats)
                render_stats_card("Correlation with SET", corr_info)
        else:
            st.info("GARCH unavailable — install `arch`.")

        # ── 3.4  XGBoost ───────────────────────────────────────────
        st.markdown("### 3.4 · XGBoost (Gradient-Boosted Trees)")
        if HAS_XGBOOST:
            with st.spinner("Training XGBoost on lagged returns …"):
                try:
                    xgb_pred, xgb_stats = xgboost_forecast(stock_series, forecast_horizon)
                except Exception as e:
                    xgb_pred = np.full(forecast_horizon, stock_series.iloc[-1])
                    xgb_stats = {"Error": str(e)[:80]}
            cc4, cs4 = st.columns([3, 1])
            with cc4:
                st.plotly_chart(prediction_chart(
                    hist_bench, hist_stock, xgb_pred, future_dates,
                    "SET Index", selected_ticker, "XGBoost",
                    line_color=ACCENT_ORANGE,
                ), use_container_width=True)
            with cs4:
                render_stats_card("Model Parameters", xgb_stats)
                render_stats_card("Correlation with SET", corr_info)
        else:
            st.info("XGBoost unavailable — install `xgboost` and `scikit-learn`.")

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
st.caption("© 2026 · Academic Project · Data via Yahoo Finance · Built with Streamlit")
