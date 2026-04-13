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

# ── Optional heavy imports ──────────────────────────────────────────────
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
RF_RATE = 0.02

# ── Colours ────────────────────────────────────────────────────────────
BLUE = "#0070FF"
BLUE_LIGHT = "#D6EAFF"
ACCENT_RED = "#E74C3C"
ACCENT_GREEN = "#27AE60"
ACCENT_ORANGE = "#F39C12"

# ════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Thai Equity Prediction Model", layout="wide", page_icon="📈"
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Source Sans Pro', sans-serif;
}

/* ═══ Force light background ═══ */
.stApp, .main, .block-container {
    background-color: #FFFFFF !important;
    color: #1a1a2e !important;
}
h1 { font-weight: 700; letter-spacing: -0.5px; color: #1a1a2e !important; }
h2 { font-weight: 600; color: #1a1a2e !important; }
h3 { font-weight: 600; color: #1a1a2e !important; }
p, li, span, label { color: #1a1a2e; }

/* ═══ Metric cards ═══ */
div[data-testid="stMetric"] {
    background: #f8f9fb; border: 1px solid #e0e5ec;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    font-size: 0.8rem !important; color: #606475 !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* ═══ Tabs — all blue ═══ */
div.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #e0e5ec; }
div.stTabs [data-baseweb="tab"] {
    padding: 8px 20px; border-radius: 6px 6px 0 0;
    font-weight: 600; color: #808495 !important; background: transparent !important;
}
div.stTabs [aria-selected="true"] {
    color: #0070FF !important; border-bottom: 3px solid #0070FF !important;
    background: transparent !important;
}
div.stTabs [data-baseweb="tab-highlight"] { background-color: #0070FF !important; }

/* ═══ Sidebar ═══ */
section[data-testid="stSidebar"] { background: #f8f9fb !important; border-right: 1px solid #e0e5ec; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #1a1a2e !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #3a3a4a !important; }

/* ═══ Selectbox — blue border ═══ */
div[data-baseweb="select"] > div {
    border-color: #0070FF !important; background: #ffffff !important; color: #1a1a2e !important;
}
/* ═══ Dropdown menu / popover — white bg, black text ═══ */
div[data-baseweb="popover"] {
    background: #FFFFFF !important;
    border: 1px solid #e0e5ec !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
ul[data-baseweb="menu"],
ul[role="listbox"],
ul[role="listbox"] li {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
}
ul[role="listbox"] li:hover,
ul[role="listbox"] li[aria-selected="true"],
ul[data-baseweb="menu"] li:hover {
    background: #EBF3FF !important;
    color: #1a1a2e !important;
}
/* Option text inside the menu */
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [data-baseweb="menu"] li div {
    color: #1a1a2e !important;
}

/* ═══ SLIDER — ALL BLUE (nuclear override) ═══ */
/* Thumb circle */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #0070FF !important;
    border-color: #0070FF !important;
}
/* Thumb value label — blue text, TRANSPARENT background */
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #0070FF !important;
    background-color: transparent !important;
    background: transparent !important;
}
/* Active (filled) track */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] {
    background-color: #0070FF !important;
}
/* Target inner track fill via nested divs */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div:first-child {
    background-color: #0070FF !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div:nth-child(2) {
    background-color: #0070FF !important;
}
/* Blanket override: any element with inline red-ish bg inside the slider */
div[data-testid="stSlider"] div[style*="background-color: rgb(255"],
div[data-testid="stSlider"] div[style*="background-color: rgb(246"],
div[data-testid="stSlider"] div[style*="background-color: rgb(240"],
div[data-testid="stSlider"] div[style*="background-color: rgb(255, 75"],
div[data-testid="stSlider"] div[style*="background-color: rgb(255,75"],
div[data-testid="stSlider"] div[style*="background: rgb(255"] {
    background-color: #0070FF !important;
    background: #0070FF !important;
}
/* Streamlit primary-color track override */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[aria-hidden="true"] div {
    background-color: #0070FF !important;
}

/* ═══ Radio buttons ═══ */
div.stRadio label { color: #1a1a2e !important; }

/* ═══ Table (st.table) — white bg, black text ═══ */
div[data-testid="stTable"] table {
    background: #FFFFFF !important; color: #1a1a2e !important;
    border-collapse: collapse; width: 100%;
}
div[data-testid="stTable"] th {
    background: #f0f2f6 !important; color: #1a1a2e !important;
    font-weight: 700 !important; padding: 10px 14px !important;
    border-bottom: 2px solid #d0d5dd !important; text-align: left !important;
    font-size: 0.85rem;
}
div[data-testid="stTable"] td {
    background: #FFFFFF !important; color: #1a1a2e !important;
    padding: 9px 14px !important; border-bottom: 1px solid #e8eaed !important;
    font-size: 0.88rem;
}
div[data-testid="stTable"] tr:hover td { background: #f8f9fb !important; }

/* Dataframe fallback */
div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] * {
    background: #FFFFFF !important; color: #1a1a2e !important;
}

hr { border-color: #e0e5ec !important; }

.disclaimer-box {
    background: #FFF8E1; border-left: 4px solid #FFB300;
    padding: 14px 18px; border-radius: 6px; font-size: 0.85rem;
    color: #5D4037 !important; margin-top: 24px; line-height: 1.55;
}

.stats-card {
    background: #f8f9fb; border: 1px solid #e0e5ec;
    border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;
}
.stats-card h4 {
    color: #0070FF !important; font-size: 0.9rem; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 10px;
    border-bottom: 2px solid #D6EAFF; padding-bottom: 6px;
}
.stats-card p { margin: 4px 0; font-size: 0.88rem; color: #1a1a2e !important; }
.stats-card .label { color: #606475 !important; font-weight: 400; }
.stats-card .value {
    font-weight: 700; color: #1a1a2e !important;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
</style>

<script>
// Force slider track to blue — overrides Streamlit's inline primaryColor
(function() {
    function fixSliders() {
        document.querySelectorAll('[data-testid="stSlider"] div[role="progressbar"]').forEach(el => {
            el.style.setProperty('background-color', '#0070FF', 'important');
        });
        // Also catch the inner track div that Streamlit colours with primaryColor
        document.querySelectorAll('[data-testid="stSlider"] [data-baseweb="slider"] div').forEach(el => {
            const bg = el.style.backgroundColor;
            if (bg && (bg.includes('255') || bg.includes('ff4') || bg.includes('rgb(255'))) {
                el.style.setProperty('background-color', '#0070FF', 'important');
            }
        });
        // Thumb value background transparent
        document.querySelectorAll('[data-testid="stThumbValue"]').forEach(el => {
            el.style.setProperty('background-color', 'transparent', 'important');
            el.style.setProperty('background', 'transparent', 'important');
        });
    }
    // Run immediately + on interval to catch rerenders
    fixSliders();
    setInterval(fixSliders, 500);
})();
</script>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Fetching market data …", ttl=3600)
def load_data(tickers: list, start: str, max_retries: int = 3) -> pd.DataFrame:
    import time

    raw = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
            if not raw.empty:
                break
        except Exception:
            pass
        if attempt < max_retries:
            time.sleep(2 * attempt)  # back-off: 2s, 4s

    if raw.empty:
        st.error("No data returned from Yahoo Finance after multiple attempts.")
        st.stop()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.iloc[
                :, raw.columns.get_level_values(0) == raw.columns.get_level_values(0)[0]
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

    # ── Retry individual tickers that came back all-NaN ────────────
    for t in tickers:
        if t in prices.columns and prices[t].dropna().empty:
            for attempt in range(1, max_retries + 1):
                try:
                    single = yf.download(t, start=start, auto_adjust=True, progress=False)
                    if not single.empty:
                        if isinstance(single.columns, pd.MultiIndex):
                            col = single["Close"].iloc[:, 0] if "Close" in single.columns.get_level_values(0) else single.iloc[:, 0]
                        elif "Close" in single.columns:
                            col = single["Close"]
                        else:
                            col = single.iloc[:, 0]
                        prices[t] = col.reindex(prices.index)
                        if prices[t].dropna().any():
                            break
                except Exception:
                    pass
                if attempt < max_retries:
                    time.sleep(2 * attempt)

    return prices.dropna(how="all")


def trim_to_span(df: pd.DataFrame, span_key: str) -> pd.DataFrame:
    if span_key == "YTD":
        return df.loc[df.index >= pd.Timestamp(datetime.now().year, 1, 1)]
    n = TIMESPAN_MAP[span_key]
    return df if n is None else df.iloc[-min(n, len(df)):]


def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1)).dropna()


def safe_last(s: pd.Series, default=np.nan):
    if s is None or len(s) == 0:
        return default
    c = s.dropna()
    return c.iloc[-1] if len(c) else default


def compute_metrics(prices: pd.DataFrame, ticker: str) -> dict:
    """Compute metrics for a single ticker. Prices should already be trimmed."""
    na = {
        "Last Price": "N/A", "CAGR": "N/A", "Ann. Return": "N/A",
        "Ann. Volatility": "N/A", "Sharpe Ratio": "N/A", "RSI (14)": "N/A",
        "5-Day MA": "N/A", "20-Day MA": "N/A",
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

    # CAGR
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 and s.iloc[0] > 0 else np.nan

    def f(v, sp):
        try:
            return "N/A" if pd.isna(v) else f"{v:{sp}}"
        except (ValueError, TypeError):
            return "N/A"

    return {
        "Last Price": f(s.iloc[-1], ",.2f"),
        "CAGR": f(cagr, ".2%"),
        "Ann. Return": f(ann_ret, ".2%"),
        "Ann. Volatility": f(ann_vol, ".2%"),
        "Sharpe Ratio": f(sharpe, ".3f"),
        "RSI (14)": f(safe_last(rsi), ".1f"),
        "5-Day MA": f(safe_last(s.rolling(5).mean()), ",.2f"),
        "20-Day MA": f(safe_last(s.rolling(20).mean()), ",.2f"),
    }


# ════════════════════════════════════════════════════════════════════════
# RMSFE  (simple train/test split — fast)
# ════════════════════════════════════════════════════════════════════════
def compute_rmsfe_split(series: pd.Series, forecast_fn, h: int = 60):
    """Compute RMSFE on the last h observations using a single train/test split.
    Much faster than walk-forward CV — avoids refitting models multiple times."""
    if len(series) < h + 100:
        return np.nan
    train = series.iloc[:-h]
    actual = series.iloc[-h:].values
    try:
        pred, _ = forecast_fn(train, h)
        pred = np.array(pred)
        n = min(len(actual), len(pred))
        return float(np.sqrt(np.mean((actual[:n] - pred[:n]) ** 2)))
    except Exception:
        return np.nan


# ════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS  (cached for speed)
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_random_walk(values_tuple, last_price, h):
    """Cache-friendly wrapper: takes tuple of values."""
    lr = pd.Series(values_tuple)
    if len(lr) < 2:
        return list(np.full(h, last_price)), {
            "Drift (μ daily)": "N/A", "Volatility (σ daily)": "N/A",
        }
    mu, sigma = lr.mean(), lr.std()
    np.random.seed(42)
    path = last_price * np.exp(np.cumsum(np.random.normal(mu, sigma, h)))
    return list(path), {
        "Drift (μ daily)": f"{mu:.6f}",
        "Volatility (σ daily)": f"{sigma:.6f}",
    }


def random_walk_forecast(series: pd.Series, h: int = 60):
    lr = log_returns(series)
    path, stats = _cached_random_walk(tuple(lr.values), float(series.iloc[-1]), h)
    return np.array(path), stats


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_arima(values_tuple, last_price, h):
    """ARIMA on log-returns — grid search (p, 0, q) since data is I(1)."""
    series = pd.Series(values_tuple)
    lr = log_returns(series).dropna()
    if len(lr) < 50:
        return list(np.full(h, last_price)), {"Error": "Insufficient data"}

    best_aic, best_order = np.inf, (1, 0, 1)
    total = 4 * 4  # p=0..3, q=0..3
    tested = 0
    for p, q in itertools.product(range(4), range(4)):
        if p == 0 and q == 0:
            tested += 1
            continue
        try:
            res = ARIMA(lr.values, order=(p, 0, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, 0, q)
        except Exception:
            pass
        tested += 1

    model = ARIMA(lr.values, order=best_order).fit()
    fc = model.forecast(steps=h)
    pred = last_price * np.exp(np.cumsum(fc))
    return list(pred), {
        "Best Order (p,d,q)": str(best_order),
        "AIC": f"{best_aic:.2f}",
        "BIC": f"{model.bic:.2f}",
        "Log-likelihood": f"{model.llf:.2f}",
    }


def arima_forecast(series: pd.Series, h: int = 60):
    pred, stats = _cached_arima(tuple(series.values), float(series.iloc[-1]), h)
    return np.array(pred), stats


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_garch(values_tuple, last_price, h):
    """GARCH on log-returns — grid search (p, q)."""
    series = pd.Series(values_tuple)
    lr = log_returns(series).dropna() * 100
    if len(lr) < 50:
        return list(np.full(h, last_price)), {"Error": "Insufficient data"}

    best_bic, best_pq = np.inf, (1, 1)
    for gp, gq in itertools.product(range(1, 4), range(1, 4)):
        try:
            am = arch_model(lr, vol="Garch", p=gp, q=gq, mean="AR", lags=1, rescale=False)
            res = am.fit(disp="off")
            if res.bic < best_bic:
                best_bic, best_pq = res.bic, (gp, gq)
        except Exception:
            continue

    am = arch_model(lr, vol="Garch", p=best_pq[0], q=best_pq[1], mean="AR", lags=1, rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=h)
    mean_fc = fc.mean.iloc[-1].values / 100
    var_fc = fc.variance.iloc[-1].values / 10000
    np.random.seed(42)
    shocks = np.random.normal(mean_fc, np.sqrt(var_fc))
    pred = last_price * np.exp(np.cumsum(shocks))

    return list(pred), {
        "Best Order (p,q)": str(best_pq),
        "BIC": f"{best_bic:.2f}",
        "ω": f"{res.params.get('omega', 0):.6f}",
        f"α (1‥{best_pq[0]})": ", ".join(
            f"{res.params.get(f'alpha[{i}]', 0):.5f}" for i in range(1, best_pq[0] + 1)
        ),
        f"β (1‥{best_pq[1]})": ", ".join(
            f"{res.params.get(f'beta[{i}]', 0):.5f}" for i in range(1, best_pq[1] + 1)
        ),
        "Log-likelihood": f"{res.loglikelihood:.2f}",
    }


def garch_forecast(series: pd.Series, h: int = 60):
    pred, stats = _cached_garch(tuple(series.values), float(series.iloc[-1]), h)
    return np.array(pred), stats


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_xgboost(values_tuple, last_price, h, n_lags):
    """XGBoost on lagged log-return features."""
    series = pd.Series(values_tuple)
    lr = log_returns(series).dropna()
    if len(lr) < n_lags + 50:
        return list(np.full(h, last_price)), {"Error": "Insufficient data"}

    data = pd.DataFrame({"ret": lr.values})
    for lag in range(1, n_lags + 1):
        data[f"lag_{lag}"] = data["ret"].shift(lag)
    data = data.dropna().reset_index(drop=True)

    X = data.drop(columns=["ret"]).values
    y = data["ret"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_test = model.predict(X_test)
    rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_val = float(mean_absolute_error(y_test, y_pred_test))

    recent = list(lr.values[-n_lags:])
    preds = []
    for _ in range(h):
        feat = np.array(recent[-n_lags:][::-1]).reshape(1, -1)
        pred_ret = float(model.predict(feat)[0])
        preds.append(pred_ret)
        recent.append(pred_ret)

    pred_prices = last_price * np.exp(np.cumsum(preds))
    best_iter = model.best_iteration + 1 if hasattr(model, "best_iteration") and model.best_iteration else 500
    return list(pred_prices), {
        "Lags": str(n_lags),
        "n_estimators (best)": str(best_iter),
        "max_depth": "5",
        "learning_rate": "0.03",
        "Val RMSE (returns)": f"{rmse_val:.6f}",
        "Val MAE (returns)": f"{mae_val:.6f}",
    }


def xgboost_forecast(series: pd.Series, h: int = 60, n_lags: int = 20):
    pred, stats = _cached_xgboost(tuple(series.values), float(series.iloc[-1]), h, n_lags)
    return np.array(pred), stats


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
    r2 = corr ** 2 if not pd.isna(corr) else np.nan
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
    font=dict(family="Source Sans Pro, sans-serif", size=13, color="#1a1a2e"),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#1a1a2e")),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    title_font=dict(color="#1a1a2e"),
)
AX = dict(gridcolor="#f0f0f0", zeroline=False,
          title_font=dict(color="#1a1a2e"), tickfont=dict(color="#1a1a2e"))


def price_chart(df: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df.values, mode="lines",
        line=dict(color=BLUE, width=2), name=title,
        fill="tozeroy", fillcolor="rgba(0,112,255,0.07)",
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title="Price (THB)", **CHART_LAYOUT)
    fig.update_xaxes(**AX)
    fig.update_yaxes(**AX)
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
    fig.update_layout(title=f"{model_label} — {stock_name} vs {bench_name}", **CHART_LAYOUT)
    fig.update_yaxes(title_text=bench_name, secondary_y=False, **AX)
    fig.update_yaxes(title_text=stock_name, secondary_y=True, **AX)
    fig.update_xaxes(**AX)
    return fig


def render_stats_card(title: str, stats: dict):
    rows = "".join(
        f'<p><span class="label">{k}:</span> <span class="value">{v}</span></p>'
        for k, v in stats.items()
    )
    st.markdown(f'<div class="stats-card"><h4>{title}</h4>{rows}</div>', unsafe_allow_html=True)


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
    set_span = st.radio(
        "Timespan", list(TIMESPAN_MAP), horizontal=True, index=3,
        key="set_span",
    )
    set_trimmed = trim_to_span(set_series.to_frame(), set_span)[BENCHMARK].dropna()

    if len(set_trimmed) >= 2:
        set_lr_t = log_returns(set_trimmed)
        set_years = (set_trimmed.index[-1] - set_trimmed.index[0]).days / 365.25
        set_cagr = (set_trimmed.iloc[-1] / set_trimmed.iloc[0]) ** (1 / set_years) - 1 if set_years > 0 and set_trimmed.iloc[0] > 0 else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Close", f"{set_trimmed.iloc[-1]:,.2f}")
        c2.metric("CAGR", f"{set_cagr:.2%}")
        c3.metric("Ann. Return", f"{set_lr_t.mean() * TRADING_DAYS:.2%}")
        c4.metric("Ann. Volatility", f"{set_lr_t.std() * np.sqrt(TRADING_DAYS):.2%}")
        c5.metric("Data Points", f"{len(set_trimmed):,}")
        st.plotly_chart(price_chart(set_trimmed, f"SET Index ({set_span})"), use_container_width=True)
    else:
        st.info(f"Not enough data for {set_span} window.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTIONS  (placed BEFORE section 2 in code so we get
#   the timespan selection first, then section 2 uses it)
# ════════════════════════════════════════════════════════════════════════
# We need the timespan choice early so Section 2 can use it.
# We'll render Section 2 first visually via st.container ordering.

stock_series_full = (
    prices[selected_ticker].dropna() if selected_ticker in prices.columns
    else pd.Series(dtype=float)
)

# Create containers so Section 2 appears above Section 3 visually
section2_container = st.container()
st.markdown("---")
section3_container = st.container()

# ── Section 3: get timespan choice ─────────────────────────────────────
with section3_container:
    st.markdown(f"## 3 · Price Predictions — {selected_label} vs SET Index")

    if len(stock_series_full) < 50 or len(set_series) < 50:
        st.error("⚠️ Not enough data for prediction models.")
        pred_span = "1Y"  # default fallback
    else:
        pred_span = st.radio(
            "Historical window for prediction charts",
            list(TIMESPAN_MAP), horizontal=True, index=3,
        )

# ── Section 2: uses the timespan from Section 3 ───────────────────────
with section2_container:
    st.markdown(f"## 2 · Asset Overview — {selected_label}")
    st.caption(f"📅 Metrics computed over **{pred_span}** window (synced with Section 3)")

    # Trim ALL prices to the chosen timespan
    prices_trimmed = trim_to_span(prices, pred_span)
    stock_series_trimmed = (
        prices_trimmed[selected_ticker].dropna()
        if selected_ticker in prices_trimmed.columns
        else pd.Series(dtype=float)
    )

    if len(stock_series_trimmed) >= 25:
        s_lr = log_returns(stock_series_trimmed)
        s_years = (stock_series_trimmed.index[-1] - stock_series_trimmed.index[0]).days / 365.25
        s_cagr = (stock_series_trimmed.iloc[-1] / stock_series_trimmed.iloc[0]) ** (1 / s_years) - 1 if s_years > 0 and stock_series_trimmed.iloc[0] > 0 else 0
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Last Close", f"{stock_series_trimmed.iloc[-1]:,.2f}")
        m2.metric("CAGR", f"{s_cagr:.2%}")
        ar = s_lr.mean() * TRADING_DAYS
        m3.metric("Ann. Return", f"{ar:.2%}")
        av = s_lr.std() * np.sqrt(TRADING_DAYS)
        m4.metric("Ann. Volatility", f"{av:.2%}")
        m5.metric("Sharpe Ratio", f"{(ar - RF_RATE) / av if av else 0:.3f}")
    else:
        st.warning(f"⚠️ Insufficient price data for **{selected_label}** in {pred_span} window.")

    rows = {lbl: compute_metrics(prices_trimmed, t) for lbl, t in TICKERS.items()}
    mdf = pd.DataFrame(rows).T
    mdf.index.name = "Asset"
    st.table(mdf)

# ── Section 3: prediction models ──────────────────────────────────────
with section3_container:
    if len(stock_series_full) >= 50 and len(set_series) >= 50:
        hist_bench = trim_to_span(set_series.to_frame(), pred_span)[BENCHMARK].dropna()
        hist_stock = trim_to_span(stock_series_full.to_frame(), pred_span)[selected_ticker].dropna()

        if len(hist_stock) < 2 or len(hist_bench) < 2:
            st.warning("Not enough data in the selected window.")
        else:
            last_date = hist_stock.index[-1]
            future_dates = pd.bdate_range(
                start=last_date + timedelta(days=1), periods=forecast_horizon
            )
            corr_info = correlation_stats(prices, selected_ticker)

            if not HAS_TIMESERIES:
                st.error("⚠️ `statsmodels` / `arch` not installed.")

            # ── 3.1  Random Walk ───────────────────────────────────
            st.markdown("### 3.1 · Random Walk (Geometric Brownian Motion)")
            with st.spinner("🎲 Simulating random walk paths …"):
                rw_pred, rw_stats = random_walk_forecast(stock_series_full, forecast_horizon)
                rmsfe_rw = compute_rmsfe_split(stock_series_full, random_walk_forecast, forecast_horizon)
                rw_stats["RMSFE"] = f"{rmsfe_rw:.2f}" if not np.isnan(rmsfe_rw) else "N/A"

            cc, cs = st.columns([3, 1])
            with cc:
                st.plotly_chart(prediction_chart(
                    hist_bench, hist_stock, rw_pred, future_dates,
                    "SET Index", selected_ticker, "Random Walk",
                ), use_container_width=True)
            with cs:
                render_stats_card("Model Parameters", rw_stats)
                render_stats_card("Correlation with SET", corr_info)

            # ── 3.2  ARIMA (p, 0, q) ──────────────────────────────
            st.markdown("### 3.2 · ARIMA (Grid Search p, 0, q)")
            if HAS_TIMESERIES:
                with st.spinner("🔍 Rolling through ARIMA(p, 0, q) grid — testing 15 combinations …"):
                    try:
                        arima_pred, arima_stats = arima_forecast(stock_series_full, forecast_horizon)
                        rmsfe_a = compute_rmsfe_split(stock_series_full, arima_forecast, forecast_horizon)
                        arima_stats["RMSFE"] = f"{rmsfe_a:.2f}" if not np.isnan(rmsfe_a) else "N/A"
                    except Exception as e:
                        arima_pred = np.full(forecast_horizon, stock_series_full.iloc[-1])
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

            # ── 3.3  GARCH (p, q) ─────────────────────────────────
            st.markdown("### 3.3 · GARCH (Grid Search p, q)")
            if HAS_TIMESERIES:
                with st.spinner("📈 Fitting GARCH(p, q) — estimating volatility clusters across 9 combinations …"):
                    try:
                        garch_pred, garch_stats = garch_forecast(stock_series_full, forecast_horizon)
                        rmsfe_g = compute_rmsfe_split(stock_series_full, garch_forecast, forecast_horizon)
                        garch_stats["RMSFE"] = f"{rmsfe_g:.2f}" if not np.isnan(rmsfe_g) else "N/A"
                    except Exception as e:
                        garch_pred = np.full(forecast_horizon, stock_series_full.iloc[-1])
                        garch_stats = {"Error": str(e)[:80]}
                cc3, cs3 = st.columns([3, 1])
                with cc3:
                    st.plotly_chart(prediction_chart(
                        hist_bench, hist_stock, garch_pred, future_dates,
                        "SET Index", selected_ticker, "GARCH",
                    ), use_container_width=True)
                with cs3:
                    render_stats_card("Model Parameters", garch_stats)
                    render_stats_card("Correlation with SET", corr_info)
            else:
                st.info("GARCH unavailable — install `arch`.")

            # ── 3.4  XGBoost ──────────────────────────────────────
            st.markdown("### 3.4 · XGBoost (Gradient-Boosted Trees)")
            if HAS_XGBOOST:
                with st.spinner("🌲 Training XGBoost — building 500 trees on lagged return features …"):
                    try:
                        xgb_pred, xgb_stats = xgboost_forecast(stock_series_full, forecast_horizon)
                        rmsfe_x = compute_rmsfe_split(stock_series_full, xgboost_forecast, forecast_horizon)
                        xgb_stats["RMSFE"] = f"{rmsfe_x:.2f}" if not np.isnan(rmsfe_x) else "N/A"
                    except Exception as e:
                        xgb_pred = np.full(forecast_horizon, stock_series_full.iloc[-1])
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
