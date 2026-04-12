# 📈 Thai Equity Price Prediction Model

An academic dashboard for the Thai stock market (SET Index) comparing Random Walk, ARIMA, and GARCH forecasts.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Features

| Section | Description |
|---------|-------------|
| **SET Index Overview** | Interactive chart of ^SET.BK with selectable timespans (1M → 20Y) |
| **Asset Overview** | Annualised return, volatility, Sharpe, RSI-14, 5/20-day MAs for all listed stocks |
| **Predictions** | Three forecast models (Random Walk, ARIMA, GARCH) plotted against the SET benchmark with correlation stats |

## Listed Assets

- PTT (PTT-R.BK)
- Airports of Thailand (AOT.BK)
- Advanced Info Service (ADVANC.BK)
- CP ALL (CPALL.BK)
- SCB X (SCB.BK)

## Methodology

- **Returns**: Logarithmic returns (`log(P_t / P_{t-1})`) on adjusted close prices.
- **Random Walk**: Geometric Brownian Motion with historical drift and volatility.
- **ARIMA**: Grid search over (p, 0, q) with p,q ∈ {0,1,2,3}; selects lowest AIC.
- **GARCH(1,1)**: Captures volatility clustering; fitted via `arch` library.

## Disclaimer

⚠️ This dashboard is for **educational and academic purposes only**. It does not constitute financial advice. Past performance does not guarantee future results.
