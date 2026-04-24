# ₿ Bitcoin Price Forecasting Portal

An interactive Streamlit web application for Bitcoin (BTC/USD) time-series analysis and forecasting.

## Features

- **Multi-model forecasting**: AutoARIMA, Prophet, Prophet+Regressors, Random Forest, Naive baseline
- **Interactive Plotly charts** with range sliders, hover tooltips, and confidence bands
- **Backtest metrics**: MAE, RMSE, MAPE, MASE (vs. naïve benchmark)
- **Future forecast** up to 180 days ahead with confidence intervals
- **Technical indicators**: SMA and EMA overlays
- **Handles any Kaggle-style BTC CSV** with automatic date/price column detection

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Any Kaggle BTC historical CSV works. Recommended:
- **Bitcoin Historical Data 2014–2024**: https://www.kaggle.com/datasets/kannapat/btc-usd-historical-price-2014-2024

Expected columns: `Date` (or `Timestamp`) and at least one of `Open, High, Low, Close`.

## Models

| Model | Type | Description |
|-------|------|-------------|
| AutoARIMA | Statistical | Auto-selects ARIMA(p,d,q) order with weekly seasonality |
| Prophet | Decomposition | Facebook's additive trend+seasonality model with logistic growth |
| Prophet+Regressors | Decomposition | Prophet extended with lag_7, volatility_7, and log_volume features |
| Random Forest | ML Ensemble | 300 trees on lag, volatility, calendar, and volume features |
| Naive | Baseline | Carries last observed price forward |

## How Models Handle Crypto Volatility

**AutoARIMA** fits on log-transformed prices, which stabilises variance and makes multiplicative shocks additive — better for BTC's explosive moves.

**Prophet** uses logistic growth with a cap at $200k, preventing unbounded parabolic extrapolation. The changepoint mechanism adapts to trend reversals, and `seasonality_mode='additive'` on log-prices implicitly captures percentage-based seasonality.

**Prophet+Regressors** adds a 7-day lag (short-term momentum), a 7-day rolling log-return standard deviation (market nervousness proxy), and log(volume) to help the model distinguish quiet consolidation from high-volume breakouts.

**Random Forest** is calibrated with `min_samples_leaf=5` on ~1000-row training windows to prevent overfitting. Features are built on the full dataset before splitting so boundary-row lags are correct.

## Project Structure

```
app.py           # Main Streamlit application
requirements.txt # Python dependencies
README.md        # This file
```
