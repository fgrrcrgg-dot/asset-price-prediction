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
import warnings, itertools, time

try:
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    HAS_TS = True
except ImportError:
    HAS_TS = False

try:
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════
TICKERS = {
    "PTT (PTT-R.BK)": "PTT-R.BK",
    "Airports of Thailand (AOT.BK)": "AOT.BK",
    "Advanced Info Service (ADVANC.BK)": "ADVANC.BK",
    "CP ALL (CPALL.BK)": "CPALL.BK",
    "SCB X (SCB.BK)": "SCB.BK",
}
BENCHMARK = "^SET.BK"
START = "2000-01-01"
SPANS = {"1M": 21, "6M": 126, "YTD": None, "1Y": 252,
         "2Y": 504, "5Y": 1260, "10Y": 2520, "20Y": 5040}
TD = 252
RF = 0.02
BLUE = "#0070FF"; BLUE_L = "#D6EAFF"
RED = "#E74C3C"; GREEN = "#27AE60"; ORANGE = "#F39C12"; PURPLE = "#8E44AD"

# ════════════════════════════════════════════════════════════════════════
# PAGE + CSS
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Thai Equity Prediction Model", layout="wide", page_icon="📈")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
html,body,[class*="st-"]{font-family:'Source Sans Pro',sans-serif}
.stApp,.main,.block-container{background:#FFF!important;color:#1a1a2e!important}
h1{font-weight:700;letter-spacing:-.5px;color:#1a1a2e!important}
h2{font-weight:600;color:#1a1a2e!important}
h3{font-weight:600;color:#1a1a2e!important}
p,li,span,label{color:#1a1a2e}
div[data-testid="stMetric"]{background:#f8f9fb;border:1px solid #e0e5ec;border-radius:10px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
div[data-testid="stMetric"] label{font-size:.8rem!important;color:#606475!important;text-transform:uppercase;letter-spacing:.5px}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{font-size:1.5rem!important;font-weight:700!important;color:#1a1a2e!important}
div.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:2px solid #e0e5ec}
div.stTabs [data-baseweb="tab"]{padding:8px 20px;border-radius:6px 6px 0 0;font-weight:600;color:#808495!important;background:transparent!important}
div.stTabs [aria-selected="true"]{color:#0070FF!important;border-bottom:3px solid #0070FF!important;background:transparent!important}
div.stTabs [data-baseweb="tab-highlight"]{background-color:#0070FF!important}
section[data-testid="stSidebar"]{background:#f8f9fb!important;border-right:1px solid #e0e5ec}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#1a1a2e!important}
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{color:#3a3a4a!important}
div[data-baseweb="select"]>div{border-color:#0070FF!important;background:#fff!important;color:#1a1a2e!important}
div[data-baseweb="popover"]{background:#FFF!important;border:1px solid #e0e5ec!important;border-radius:8px!important}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li,ul[data-baseweb="menu"],ul[role="listbox"],ul[role="listbox"] li{background:#FFF!important;color:#1a1a2e!important}
ul[role="listbox"] li:hover,ul[role="listbox"] li[aria-selected="true"]{background:#EBF3FF!important;color:#1a1a2e!important}
div[data-baseweb="popover"] [role="option"],div[data-baseweb="popover"] [data-baseweb="menu"] li div{color:#1a1a2e!important}
div[data-testid="stSlider"] [role="slider"]{background-color:#0070FF!important;border-color:#0070FF!important}
div[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#0070FF!important;background:transparent!important}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"]{background-color:#0070FF!important}
div[data-testid="stSlider"] div[data-baseweb="slider"]>div>div>div:first-child,
div[data-testid="stSlider"] div[data-baseweb="slider"]>div>div>div:nth-child(2){background-color:#0070FF!important}
div[data-testid="stSlider"] div[style*="background-color: rgb(255"],
div[data-testid="stSlider"] div[style*="background: rgb(255"]{background-color:#0070FF!important;background:#0070FF!important}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[aria-hidden="true"] div{background-color:#0070FF!important}
div.stRadio label{color:#1a1a2e!important}
div[data-testid="stTable"] table{background:#FFF!important;color:#1a1a2e!important;border-collapse:collapse;width:100%}
div[data-testid="stTable"] th{background:#f0f2f6!important;color:#1a1a2e!important;font-weight:700!important;padding:10px 14px!important;border-bottom:2px solid #d0d5dd!important;text-align:left!important;font-size:.85rem}
div[data-testid="stTable"] td{background:#FFF!important;color:#1a1a2e!important;padding:9px 14px!important;border-bottom:1px solid #e8eaed!important;font-size:.88rem}
div[data-testid="stTable"] tr:hover td{background:#f8f9fb!important}
div[data-testid="stDataFrame"],div[data-testid="stDataFrame"] *{background:#FFF!important;color:#1a1a2e!important}
hr{border-color:#e0e5ec!important}
/* ═══ Date picker — force white everywhere with red selection ═══ */
div[data-baseweb="calendar"],
div[data-baseweb="calendar"] *,
div[data-baseweb="datepicker"],
div[data-baseweb="datepicker"] *,
div[data-baseweb="calendar"] div,
div[data-baseweb="calendar"] span,
div[data-baseweb="calendar"] td,
div[data-baseweb="calendar"] tr,
div[data-baseweb="calendar"] table,
div[data-baseweb="calendar"] thead,
div[data-baseweb="calendar"] tbody {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #1a1a2e !important;
}
/* Calendar container and month/year header */
div[data-baseweb="calendar"] > div,
div[data-baseweb="calendar"] > div > div,
div[data-baseweb="calendar"] > div > div > div {
    background: #FFFFFF !important;
}
/* Navigation arrows */
div[data-baseweb="calendar"] button[aria-label] {
    background: transparent !important;
    color: #0070FF !important;
    border: none !important;
}
/* Day cells - ALL buttons in calendar */
div[data-baseweb="calendar"] button {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
    border: none !important;
}
div[data-baseweb="calendar"] button:hover {
    background: #EBF3FF !important;
}
div[data-baseweb="calendar"] button:focus {
    background: #FFFFFF !important;
}
/* Selected day — RED */
div[data-baseweb="calendar"] button[aria-selected="true"],
div[data-baseweb="calendar"] div[role="gridcell"] button[aria-checked="true"],
div[data-baseweb="calendar"] button[aria-pressed="true"] {
    background: #E74C3C !important;
    color: #FFFFFF !important;
    border: none !important;
}
/* Day-of-week headers (Mo Tu We...) */
div[data-baseweb="calendar"] [role="columnheader"],
div[data-baseweb="calendar"] th {
    color: #808495 !important;
    background: #FFFFFF !important;
}
/* Month/year dropdown selects */
div[data-baseweb="calendar"] select,
div[data-baseweb="calendar"] [data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
}
/* The popover wrapper that contains the calendar */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] > div > div,
div[data-baseweb="popover"] > div > div > div {
    background: #FFFFFF !important;
}
/* Date input field */
div[data-testid="stDateInput"] input {
    background: #FFFFFF !important;
    color: #1a1a2e !important;
    border-color: #0070FF !important;
}
div[data-testid="stDateInput"] > div > div {
    background: #FFFFFF !important;
    border-color: #0070FF !important;
}
/* Out-of-range / disabled days - MUST be white */
div[data-baseweb="calendar"] button:disabled,
div[data-baseweb="calendar"] button[disabled],
div[data-baseweb="calendar"] [aria-disabled="true"] {
    background: #FFFFFF !important;
    color: #c0c4cc !important;
    border: none !important;
}
/* Catch-all for any remaining black backgrounds */
div[data-baseweb="calendar"] *[style*="rgb(0, 0, 0)"],
div[data-baseweb="calendar"] *[style*="#000000"],
div[data-baseweb="popover"] *[style*="rgb(0, 0, 0)"],
div[data-baseweb="popover"] *[style*="#000000"] {
    background-color: #FFFFFF !important !important;
    background: #FFFFFF !important !important;
}
.disclaimer-box{background:#FFF8E1;border-left:4px solid #FFB300;padding:14px 18px;border-radius:6px;font-size:.85rem;color:#5D4037!important;margin-top:24px;line-height:1.55}
.stats-card{background:#f8f9fb;border:1px solid #e0e5ec;border-radius:10px;padding:18px 20px;margin-bottom:12px}
.stats-card h4{color:#0070FF!important;font-size:.9rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;border-bottom:2px solid #D6EAFF;padding-bottom:6px}
.stats-card p{margin:4px 0;font-size:.88rem;color:#1a1a2e!important}
.stats-card .label{color:#606475!important;font-weight:400}
.stats-card .value{font-weight:700;color:#1a1a2e!important;font-family:'SF Mono','Fira Code',monospace}
</style>
<script>
(function(){function f(){document.querySelectorAll('[data-testid="stSlider"] div[role="progressbar"]').forEach(e=>{e.style.setProperty('background-color','#0070FF','important')});document.querySelectorAll('[data-testid="stSlider"] [data-baseweb="slider"] div').forEach(e=>{const b=e.style.backgroundColor;if(b&&b.includes('255'))e.style.setProperty('background-color','#0070FF','important')});document.querySelectorAll('[data-testid="stThumbValue"]').forEach(e=>{e.style.setProperty('background','transparent','important')})}f();setInterval(f,500)})();
</script>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Fetching market data …", ttl=3600)
def load_data(tickers, start):
    raw = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
            if not raw.empty: break
        except Exception: pass
        if attempt < 3: time.sleep(2 * attempt)
    if raw.empty:
        st.error("No data from Yahoo Finance."); st.stop()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(tickers)].copy()
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.droplevel(0)
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        if len(tickers) == 1: prices.columns = tickers
    for t in tickers:
        if t not in prices.columns: prices[t] = np.nan
    # Retry empty tickers individually
    for t in tickers:
        if t in prices.columns and prices[t].dropna().empty:
            try:
                s = yf.download(t, start=start, auto_adjust=True, progress=False)
                if not s.empty:
                    col = s["Close"] if "Close" in s.columns else s.iloc[:, 0]
                    if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
                    prices[t] = col.reindex(prices.index)
            except Exception: pass
    return prices.dropna(how="all")

def trim(df, k):
    if k == "YTD": return df.loc[df.index >= pd.Timestamp(datetime.now().year, 1, 1)]
    n = SPANS[k]
    return df if n is None else df.iloc[-min(n, len(df)):]

def logr(s): return np.log(s / s.shift(1)).dropna()

def safe_last(s, d=np.nan):
    if s is None or len(s) == 0: return d
    c = s.dropna()
    return c.iloc[-1] if len(c) else d

def compute_metrics(prices, ticker):
    na = {"Last Price":"N/A","CAGR":"N/A","Ann. Return":"N/A","Ann. Volatility":"N/A",
          "Sharpe Ratio":"N/A","RSI (14)":"N/A","5-Day MA":"N/A","20-Day MA":"N/A"}
    if ticker not in prices.columns: return na
    s = prices[ticker].dropna()
    if len(s) < 25: return na
    lr = logr(s)
    if len(lr) < 2: return na
    ar = lr.mean()*TD; av = lr.std()*np.sqrt(TD)
    sh = (ar-RF)/av if av else np.nan
    d = s.diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100-(100/(1+g/l))
    y = (s.index[-1]-s.index[0]).days/365.25
    cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1 if y>0 and s.iloc[0]>0 else np.nan
    def f(v,sp):
        try: return "N/A" if pd.isna(v) else f"{v:{sp}}"
        except: return "N/A"
    return {"Last Price":f(s.iloc[-1],",.2f"),"CAGR":f(cagr,".2%"),"Ann. Return":f(ar,".2%"),
            "Ann. Volatility":f(av,".2%"),"Sharpe Ratio":f(sh,".3f"),"RSI (14)":f(safe_last(rsi),".1f"),
            "5-Day MA":f(safe_last(s.rolling(5).mean()),",.2f"),"20-Day MA":f(safe_last(s.rolling(20).mean()),",.2f")}


# ════════════════════════════════════════════════════════════════════════
# BACKTEST + RMSFE HELPER
# ════════════════════════════════════════════════════════════════════════
def backtest_and_forecast(series, forecast_fn, h, cutoff_date=None):
    """Split series at cutoff_date (or last h obs if None).
    - Train on data BEFORE cutoff → model params come from train only
    - Backtest = predict from cutoff to end of series
    - Forecast = predict h steps beyond end of series from full data
    """
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        train = series.loc[series.index < cutoff]
        test = series.loc[series.index >= cutoff]
    else:
        if len(series) < h + 100:
            return None, None, None, None, {"Error": "Insufficient data"}
        train = series.iloc[:-h]
        test = series.iloc[-h:]

    if len(train) < 50:
        return None, None, None, None, {"Error": "Training period too short (need ≥50 obs)"}

    bt_h = len(test) if len(test) > 0 else h

    # Fit on TRAIN only → stats reflect the training period
    try:
        bt_pred, stats = forecast_fn(train, max(bt_h, 1))
        bt_pred = np.array(bt_pred)[:len(test)] if len(test) > 0 else None
    except Exception as e:
        bt_pred = None
        stats = {"Error": str(e)[:80]}

    # RMSFE on backtest period
    if bt_pred is not None and len(test) > 0:
        actual = test.values
        n = min(len(actual), len(bt_pred))
        rmsfe = float(np.sqrt(np.mean((actual[:n] - bt_pred[:n])**2)))
        stats["RMSFE"] = f"{rmsfe:.2f}"
    else:
        stats["RMSFE"] = "N/A"

    bt_dates = test.index if len(test) > 0 else None

    # Forecast future from full series
    try:
        fut_pred, _ = forecast_fn(series, h)
        fut_pred = np.array(fut_pred)
    except Exception:
        fut_pred = np.full(h, float(series.iloc[-1]))

    fut_dates = pd.bdate_range(start=series.index[-1] + timedelta(days=1), periods=h)

    return bt_pred, bt_dates, fut_pred, fut_dates, stats


# ════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS (cached)
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _rw(vals, last, h):
    s = pd.Series(vals); lr = logr(s)
    if len(lr) < 2: return list(np.full(h, last)), {"Drift (μ)":"N/A","Vol (σ)":"N/A"}
    mu, sig = lr.mean(), lr.std()
    np.random.seed(42)
    p = last * np.exp(np.cumsum(np.random.normal(mu, sig, h)))
    return list(p), {"Drift (μ daily)":f"{mu:.6f}","Volatility (σ daily)":f"{sig:.6f}"}

def random_walk_forecast(series, h=60):
    p, s = _rw(tuple(series.values), float(series.iloc[-1]), h)
    return np.array(p), s

@st.cache_data(show_spinner=False, ttl=3600)
def _arima(vals, last, h):
    s = pd.Series(vals); lr = logr(s).dropna()
    if len(lr) < 50: return list(np.full(h, last)), {"Error":"Insufficient data"}
    best_aic, best_o = np.inf, (1,0,1)
    for p, q in itertools.product(range(4), range(4)):
        if p==0 and q==0: continue
        try:
            r = ARIMA(lr.values, order=(p,0,q)).fit()
            if r.aic < best_aic: best_aic, best_o = r.aic, (p,0,q)
        except: pass
    m = ARIMA(lr.values, order=best_o).fit()
    fc = m.forecast(steps=h)
    pred = last * np.exp(np.cumsum(fc))
    return list(pred), {"Order (p,d,q)":str(best_o),"AIC":f"{best_aic:.2f}","BIC":f"{m.bic:.2f}","Log-lik":f"{m.llf:.2f}"}

def arima_forecast(series, h=60):
    p, s = _arima(tuple(series.values), float(series.iloc[-1]), h)
    return np.array(p), s

@st.cache_data(show_spinner=False, ttl=3600)
def _garch(vals, last, h):
    s = pd.Series(vals); lr = logr(s).dropna()*100
    if len(lr) < 50: return list(np.full(h, last)), {"Error":"Insufficient data"}
    best_bic, best_pq = np.inf, (1,1)
    for gp, gq in itertools.product(range(1,4), range(1,4)):
        try:
            r = arch_model(lr, vol="Garch", p=gp, q=gq, mean="AR", lags=1, rescale=False).fit(disp="off")
            if r.bic < best_bic: best_bic, best_pq = r.bic, (gp, gq)
        except: continue
    am = arch_model(lr, vol="Garch", p=best_pq[0], q=best_pq[1], mean="AR", lags=1, rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=h)
    mf = fc.mean.iloc[-1].values/100; vf = fc.variance.iloc[-1].values/10000
    np.random.seed(42)
    pred = last * np.exp(np.cumsum(np.random.normal(mf, np.sqrt(vf))))
    return list(pred), {"Order (p,q)":str(best_pq),"BIC":f"{best_bic:.2f}",
        "ω":f"{res.params.get('omega',0):.6f}",
        f"α(1‥{best_pq[0]})": ", ".join(f"{res.params.get(f'alpha[{i}]',0):.5f}" for i in range(1,best_pq[0]+1)),
        f"β(1‥{best_pq[1]})": ", ".join(f"{res.params.get(f'beta[{i}]',0):.5f}" for i in range(1,best_pq[1]+1)),
        "Log-lik":f"{res.loglikelihood:.2f}"}

def garch_forecast(series, h=60):
    p, s = _garch(tuple(series.values), float(series.iloc[-1]), h)
    return np.array(p), s

@st.cache_data(show_spinner=False, ttl=3600)
def _xgb(vals, last, h, nl):
    s = pd.Series(vals); lr = logr(s).dropna()
    if len(lr) < nl+50: return list(np.full(h, last)), {"Error":"Insufficient data"}
    df = pd.DataFrame({"r":lr.values})
    for i in range(1,nl+1): df[f"l{i}"] = df["r"].shift(i)
    df = df.dropna().reset_index(drop=True)
    X, y = df.drop(columns=["r"]).values, df["r"].values
    sp = int(len(X)*0.8)
    m = XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,subsample=0.8,
                     colsample_bytree=0.8,random_state=42,verbosity=0,early_stopping_rounds=30)
    m.fit(X[:sp],y[:sp],eval_set=[(X[sp:],y[sp:])],verbose=False)
    yp = m.predict(X[sp:])
    rmse = float(np.sqrt(mean_squared_error(y[sp:],yp)))
    mae = float(mean_absolute_error(y[sp:],yp))
    rec = list(lr.values[-nl:])
    preds = []
    for _ in range(h):
        feat = np.array(rec[-nl:][::-1]).reshape(1,-1)
        pr = float(m.predict(feat)[0]); preds.append(pr); rec.append(pr)
    pp = last * np.exp(np.cumsum(preds))
    bi = m.best_iteration+1 if hasattr(m,'best_iteration') and m.best_iteration else 500
    return list(pp), {"Lags":str(nl),"Trees (best)":str(bi),"max_depth":"5","lr":"0.03",
                      "Val RMSE":f"{rmse:.6f}","Val MAE":f"{mae:.6f}"}

def xgboost_forecast(series, h=60, nl=20):
    p, s = _xgb(tuple(series.values), float(series.iloc[-1]), h, nl)
    return np.array(p), s

def corr_stats(prices, ticker, span_key):
    """Correlation stats computed on the TRIMMED window."""
    pr = trim(prices, span_key)
    if ticker not in pr.columns or BENCHMARK not in pr.columns:
        return {"Correlation":"N/A","Beta":"N/A","R²":"N/A"}
    both = pr[[ticker, BENCHMARK]].dropna()
    if len(both) < 10: return {"Correlation":"N/A","Beta":"N/A","R²":"N/A"}
    lr = np.log(both/both.shift(1)).dropna()
    if len(lr) < 10: return {"Correlation":"N/A","Beta":"N/A","R²":"N/A"}
    c = lr[ticker].corr(lr[BENCHMARK])
    b = lr[ticker].cov(lr[BENCHMARK])/lr[BENCHMARK].var()
    r2 = c**2 if not pd.isna(c) else np.nan
    return {"Correlation":f"{c:.4f}" if not pd.isna(c) else "N/A",
            "Beta":f"{b:.4f}" if not pd.isna(b) else "N/A",
            "R²":f"{r2:.4f}" if not pd.isna(r2) else "N/A"}


# ════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ════════════════════════════════════════════════════════════════════════
CL = dict(template="plotly_white",
    font=dict(family="Source Sans Pro, sans-serif",size=13,color="#1a1a2e"),
    margin=dict(l=50,r=30,t=50,b=40), hovermode="x unified",
    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(color="#1a1a2e")),
    plot_bgcolor="#FFF",paper_bgcolor="#FFF",title_font=dict(color="#1a1a2e"))
AX = dict(gridcolor="#f0f0f0",zeroline=False,title_font=dict(color="#1a1a2e"),tickfont=dict(color="#1a1a2e"))

def price_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df.values,mode="lines",
        line=dict(color=BLUE,width=2),name=title,fill="tozeroy",fillcolor="rgba(0,112,255,0.07)",
        hovertemplate="%{x|%d %b %Y}<br>Price: %{y:,.2f}<extra></extra>"))
    fig.update_layout(title=title,yaxis_title="Price (THB)",**CL)
    fig.update_xaxes(**AX); fig.update_yaxes(**AX)
    return fig

def forecast_chart_single(hist, bt_pred, bt_dates, fut_pred, fut_dates, name, model, cutoff_date=None, fc=GREEN):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index,y=hist.values,mode="lines",
        line=dict(color=BLUE,width=1.5),name=f"{name} (hist)",opacity=0.7))
    if bt_pred is not None and bt_dates is not None:
        fig.add_trace(go.Scatter(x=bt_dates,y=bt_pred,mode="lines",
            line=dict(color=PURPLE,width=2,dash="dash"),name=f"Backtest ({model})"))
    fig.add_trace(go.Scatter(x=fut_dates,y=fut_pred,mode="lines",
        line=dict(color=fc,width=2.5,dash="dot"),name=f"Forecast ({model})"))
    if cutoff_date is not None:
        cd_str = pd.Timestamp(cutoff_date).strftime("%Y-%m-%d")
        fig.add_vline(x=cd_str, line_dash="longdash", line_color="#888", line_width=1)
        fig.add_annotation(x=cd_str, y=1, yref="paper", text="Cutoff",
                           showarrow=False, font=dict(size=10, color="#888"),
                           xanchor="right", yanchor="top")
    fig.update_layout(title=f"{model} — {name}",**CL)
    fig.update_xaxes(**AX); fig.update_yaxes(**AX)
    return fig

def forecast_chart_dual(hist_bench, hist_stock, bt_pred, bt_dates, fut_pred, fut_dates,
                        bench_name, stock_name, model, cutoff_date=None, fc=GREEN):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=hist_bench.index,y=hist_bench.values,mode="lines",
        line=dict(color=BLUE,width=1.5),name=f"{bench_name} (hist)",opacity=0.6),secondary_y=False)
    fig.add_trace(go.Scatter(x=hist_stock.index,y=hist_stock.values,mode="lines",
        line=dict(color=RED,width=1.5),name=f"{stock_name} (hist)",opacity=0.6),secondary_y=True)
    if bt_pred is not None and bt_dates is not None:
        fig.add_trace(go.Scatter(x=bt_dates,y=bt_pred,mode="lines",
            line=dict(color=PURPLE,width=2,dash="dash"),name=f"Backtest ({model})"),secondary_y=True)
    fig.add_trace(go.Scatter(x=fut_dates,y=fut_pred,mode="lines",
        line=dict(color=fc,width=2.5,dash="dot"),name=f"Forecast ({model})"),secondary_y=True)
    if cutoff_date is not None:
        cd_str = pd.Timestamp(cutoff_date).strftime("%Y-%m-%d")
        fig.add_vline(x=cd_str, line_dash="longdash", line_color="#888", line_width=1)
        fig.add_annotation(x=cd_str, y=1, yref="paper", text="Cutoff",
                           showarrow=False, font=dict(size=10, color="#888"),
                           xanchor="right", yanchor="top")
    fig.update_layout(title=f"{model} — {stock_name} vs {bench_name}",**CL)
    fig.update_yaxes(title_text=bench_name,secondary_y=False,**AX)
    fig.update_yaxes(title_text=stock_name,secondary_y=True,**AX)
    fig.update_xaxes(**AX)
    return fig

def stats_card(title, stats):
    rows = "".join(f'<p><span class="label">{k}:</span> <span class="value">{v}</span></p>' for k,v in stats.items())
    st.markdown(f'<div class="stats-card"><h4>{title}</h4>{rows}</div>', unsafe_allow_html=True)


def render_model_block(series, hist_main, hist_bench, forecast_fn, model_name,
                       span_key, prices, ticker, h, cutoff_date,
                       line_color=GREEN, is_benchmark=False):
    """Render one model with cutoff-based train/test split."""
    bt_pred, bt_dates, fut_pred, fut_dates, model_stats = backtest_and_forecast(
        series, forecast_fn, h, cutoff_date=cutoff_date)

    # Add train/backtest period info to stats
    cutoff_ts = pd.Timestamp(cutoff_date) if cutoff_date else None
    if cutoff_ts:
        train_data = series.loc[series.index < cutoff_ts]
        test_data = series.loc[series.index >= cutoff_ts]
    else:
        train_data = series.iloc[:-h]
        test_data = series.iloc[-h:]
    if len(train_data) > 0:
        model_stats["Train period"] = f"{train_data.index[0].strftime('%d %b %Y')} → {train_data.index[-1].strftime('%d %b %Y')}"
        model_stats["Train size"] = f"{len(train_data):,} obs"
    if len(test_data) > 0:
        model_stats["Backtest period"] = f"{test_data.index[0].strftime('%d %b %Y')} → {test_data.index[-1].strftime('%d %b %Y')}"
        model_stats["Backtest size"] = f"{len(test_data):,} obs"

    cc, cs = st.columns([3, 1])
    with cc:
        if is_benchmark:
            fig = forecast_chart_single(hist_main, bt_pred, bt_dates, fut_pred, fut_dates,
                                        "SET Index", model_name, cutoff_date=cutoff_date, fc=line_color)
        else:
            fig = forecast_chart_dual(hist_bench, hist_main, bt_pred, bt_dates, fut_pred, fut_dates,
                                      "SET Index", ticker, model_name, cutoff_date=cutoff_date, fc=line_color)
        st.plotly_chart(fig, use_container_width=True)
    with cs:
        stats_card("Model Parameters", model_stats)
        if not is_benchmark:
            ci = corr_stats(prices, ticker, span_key)
            stats_card("Correlation with SET", ci)


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    sel_label = st.selectbox("**Select Asset**", list(TICKERS.keys()))
    sel_ticker = TICKERS[sel_label]
    fh = st.slider("Forecast horizon (trading days)", 10, 120, 60, step=5)
    st.divider()
    st.markdown("### About")
    st.markdown("Built for an **academic project** on equity price prediction. "
        "The models do *not* predict exact future prices — they illustrate "
        "different statistical approaches applied to the Thai stock market.")
    st.markdown('<div class="disclaimer-box">⚠️ <b>Disclaimer</b> — For <b>educational and '
        'academic purposes only</b>. Not financial advice. Past performance ≠ future results.</div>',
        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════
all_t = [BENCHMARK] + list(TICKERS.values())
prices = load_data(all_t, START)

st.markdown("# 📈 Thai Equity Price Prediction Model")
st.markdown("**An academic dashboard for the Thai stock market (SET Index)** — "
    "comparing Random Walk, ARIMA, GARCH, and XGBoost forecasts.")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — SET INDEX OVERVIEW
# ════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · SET Index Overview")
set_full = prices[BENCHMARK].dropna() if BENCHMARK in prices.columns else pd.Series(dtype=float)

if len(set_full) < 2:
    st.warning("⚠️ SET Index data unavailable.")
else:
    s1_span = st.radio("Timespan", list(SPANS), horizontal=True, index=3, key="s1span")
    st_trim = trim(set_full.to_frame(), s1_span)[BENCHMARK].dropna()
    if len(st_trim) >= 2:
        lr1 = logr(st_trim)
        y1 = (st_trim.index[-1]-st_trim.index[0]).days/365.25
        cagr1 = (st_trim.iloc[-1]/st_trim.iloc[0])**(1/y1)-1 if y1>0 and st_trim.iloc[0]>0 else 0
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Last Close", f"{st_trim.iloc[-1]:,.2f}")
        c2.metric("CAGR", f"{cagr1:.2%}")
        c3.metric("Ann. Return", f"{lr1.mean()*TD:.2%}")
        c4.metric("Ann. Volatility", f"{lr1.std()*np.sqrt(TD):.2%}")
        c5.metric("Data Points", f"{len(st_trim):,}")
        st.plotly_chart(price_chart(st_trim, f"SET Index ({s1_span})"), use_container_width=True)
    else:
        st.info(f"Not enough data for {s1_span}.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# We need timespan choices from Sections 3 & 4 BEFORE rendering Section 2
# ════════════════════════════════════════════════════════════════════════
stock_full = prices[sel_ticker].dropna() if sel_ticker in prices.columns else pd.Series(dtype=float)

sec2 = st.container()
st.markdown("---")
sec3 = st.container()
st.markdown("---")
sec4 = st.container()

# ── Section 3: SET Index Forecast (get timespan + cutoff) ─────────────
with sec3:
    st.markdown("## 3 · SET Index Forecast")
    if len(set_full) < 100:
        st.error("⚠️ Not enough SET data."); s3_span = "1Y"; s3_cutoff = None
    else:
        s3c1, s3c2 = st.columns([2, 1])
        with s3c1:
            s3_span = st.radio("Training start (lookback)", list(SPANS), horizontal=True, index=3, key="s3span")
        with s3c2:
            # Default cutoff: 1 year ago
            default_cutoff = set_full.index[-1] - pd.DateOffset(years=1)
            s3_cutoff = st.date_input(
                "Train/Test cutoff date",
                value=default_cutoff.date(),
                min_value=set_full.index[50].date(),
                max_value=set_full.index[-1].date(),
                key="s3cutoff",
            )

# ── Section 4: Stock Forecast (get timespan + cutoff) ─────────────────
with sec4:
    st.markdown(f"## 4 · Stock Forecast — {sel_label} vs SET Index")
    if len(stock_full) < 100 or len(set_full) < 100:
        st.error("⚠️ Not enough data."); s4_span = "1Y"; s4_cutoff = None
    else:
        s4c1, s4c2 = st.columns([2, 1])
        with s4c1:
            s4_span = st.radio("Training start (lookback)", list(SPANS), horizontal=True, index=3, key="s4span")
        with s4c2:
            default_cutoff4 = stock_full.index[-1] - pd.DateOffset(years=1)
            s4_cutoff = st.date_input(
                "Train/Test cutoff date",
                value=default_cutoff4.date(),
                min_value=stock_full.index[50].date(),
                max_value=stock_full.index[-1].date(),
                key="s4cutoff",
            )

# ── Section 2: Asset Overview (synced with Section 4 timespan) ────────
with sec2:
    st.markdown(f"## 2 · Asset Overview — {sel_label}")
    st.caption(f"📅 Metrics computed over **{s4_span}** window (synced with Section 4)")
    pt = trim(prices, s4_span)
    st2 = pt[sel_ticker].dropna() if sel_ticker in pt.columns else pd.Series(dtype=float)
    if len(st2) >= 25:
        slr = logr(st2)
        sy = (st2.index[-1]-st2.index[0]).days/365.25
        scagr = (st2.iloc[-1]/st2.iloc[0])**(1/sy)-1 if sy>0 and st2.iloc[0]>0 else 0
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Last Close", f"{st2.iloc[-1]:,.2f}")
        m2.metric("CAGR", f"{scagr:.2%}")
        ar2 = slr.mean()*TD; av2 = slr.std()*np.sqrt(TD)
        m3.metric("Ann. Return", f"{ar2:.2%}")
        m4.metric("Ann. Volatility", f"{av2:.2%}")
        m5.metric("Sharpe Ratio", f"{(ar2-RF)/av2 if av2 else 0:.3f}")
    else:
        st.warning(f"⚠️ Insufficient data for **{sel_label}** in {s4_span} window.")
    rows = {lbl: compute_metrics(pt, t) for lbl, t in TICKERS.items()}
    st.table(pd.DataFrame(rows).T.rename_axis("Asset"))

# ── Section 3: SET Index Forecast (models) ────────────────────────────
with sec3:
    if len(set_full) >= 100:
        hist_set = trim(set_full.to_frame(), s3_span)[BENCHMARK].dropna()
        if len(hist_set) < 2:
            st.warning("Not enough data.")
        else:
            st.caption(f"📅 Training data starts from **{hist_set.index[0].strftime('%d %b %Y')}** "
                       f"(cutoff: **{pd.Timestamp(s3_cutoff).strftime('%d %b %Y') if s3_cutoff else 'auto'}**)")
            MODELS_SET = [
                ("3.1","Random Walk (GBM)", random_walk_forecast, "🎲 Simulating …", GREEN),
                ("3.2","ARIMA (p, 0, q)", arima_forecast, "🔍 Grid-searching ARIMA …", GREEN),
                ("3.3","GARCH (p, q)", garch_forecast, "📈 Fitting GARCH …", GREEN),
                ("3.4","XGBoost", xgboost_forecast, "🌲 Training XGBoost …", ORANGE),
            ]
            for num, name, fn, spinner_msg, color in MODELS_SET:
                has_lib = (HAS_TS if "ARIMA" in name or "GARCH" in name else
                           HAS_XGB if "XGBoost" in name else True)
                if not has_lib:
                    st.info(f"{name} unavailable — install required packages.")
                    continue
                st.markdown(f"### {num} · {name}")
                with st.spinner(spinner_msg):
                    render_model_block(
                        series=hist_set, hist_main=hist_set, hist_bench=None,
                        forecast_fn=fn, model_name=name, span_key=s3_span,
                        prices=prices, ticker=BENCHMARK, h=fh, cutoff_date=s3_cutoff,
                        line_color=color, is_benchmark=True)

# ── Section 4: Stock Forecast (models) ────────────────────────────────
with sec4:
    if len(stock_full) >= 100 and len(set_full) >= 100:
        hist_stk = trim(stock_full.to_frame(), s4_span)[sel_ticker].dropna()
        hist_bench = trim(set_full.to_frame(), s4_span)[BENCHMARK].dropna()
        if len(hist_stk) < 2 or len(hist_bench) < 2:
            st.warning("Not enough data.")
        else:
            st.caption(f"📅 Training data starts from **{hist_stk.index[0].strftime('%d %b %Y')}** "
                       f"(cutoff: **{pd.Timestamp(s4_cutoff).strftime('%d %b %Y') if s4_cutoff else 'auto'}**)")
            MODELS_STK = [
                ("4.1","Random Walk (GBM)", random_walk_forecast, "🎲 Simulating …", GREEN),
                ("4.2","ARIMA (p, 0, q)", arima_forecast, "🔍 Grid-searching ARIMA …", GREEN),
                ("4.3","GARCH (p, q)", garch_forecast, "📈 Fitting GARCH …", GREEN),
                ("4.4","XGBoost", xgboost_forecast, "🌲 Training XGBoost …", ORANGE),
            ]
            for num, name, fn, spinner_msg, color in MODELS_STK:
                has_lib = (HAS_TS if "ARIMA" in name or "GARCH" in name else
                           HAS_XGB if "XGBoost" in name else True)
                if not has_lib:
                    st.info(f"{name} unavailable — install required packages.")
                    continue
                st.markdown(f"### {num} · {name}")
                with st.spinner(spinner_msg):
                    render_model_block(
                        series=hist_stk, hist_main=hist_stk, hist_bench=hist_bench,
                        forecast_fn=fn, model_name=name, span_key=s4_span,
                        prices=prices, ticker=sel_ticker, h=fh, cutoff_date=s4_cutoff,
                        line_color=color, is_benchmark=False)

# ════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="disclaimer-box">⚠️ <b>Academic Disclaimer</b> — For educational and academic '
    'purposes only. Not investment advice. Models have well-known limitations. '
    'Consult a licensed financial professional. Data via Yahoo Finance.</div>', unsafe_allow_html=True)
st.caption("© 2026 · Academic Project · Data via Yahoo Finance · Built with Streamlit")
