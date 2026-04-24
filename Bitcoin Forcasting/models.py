import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor


# build three features from price & volume
# 1. lag_7: log(price) shifted 7 days back
# weekly momentum (if the price was high 7 days ago it will likely remain high in the short term)
# 2. volatility_7: volatility over the last 7 days,
# calculated from daily price changes (log returns) high volatility -> unstable market -> probability of trend reversal
# 3. log_volume: trading volume (converted to log) only used if there is real volume data

# {calculate the features on all the data first then split it into tain_test
# cause some values like lag_7 require data from previous days and we don't need to lose these values


def calc_metrics(y_true, y_pred, train_actual, horizon=90):

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    # naive baseline: predict price[t] = price[t - horizon]
    naive_errors = np.abs(train_actual[horizon:] - train_actual[:-horizon])
    mase = mae / (float(np.mean(naive_errors)) + 1e-8)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 4),
        "MASE": round(mase, 4),
    }


def build_regressors(df):
    
    df = df.copy()
    df["lag_7"] = np.log1p(df["y"]).shift(7)

    log_returns = np.log1p(df["y"]).diff()
    df["volatility_7"] = log_returns.rolling(7).std()

    if "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"])

    # drop first 7 rows that are NaN due to the rolling window
    df = df.dropna().reset_index(drop=True)
    return df


def run_autoarima(train, test):
   
    df_train = train[["ds", "y"]].copy()
    df_train["unique_id"] = "BTC"

    # apply log transform
    # cause ARIMA assume a stationary, normally distributed error & BTC prices are non-stationary
    # log(price) make it approximate stationary
    df_train["y"] = np.log(df_train["y"])
    df_train = df_train[["unique_id", "ds", "y"]]

    # season_length=7 captures the weekly trading cycle
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    sf.fit(df_train)

    fcst = sf.predict(h=len(test), level=[80, 95])

    # exp(): back transform predictions to return to USD
    y_pred  = np.exp(fcst["AutoARIMA"].values)

    # use 95% CI if available otherwise = 80%
    lo_col = "AutoARIMA-lo-95" if "AutoARIMA-lo-95" in fcst.columns else "AutoARIMA-lo-80"
    hi_col = "AutoARIMA-hi-95" if "AutoARIMA-hi-95" in fcst.columns else "AutoARIMA-hi-80"
    conf_lo = np.exp(fcst[lo_col].values)
    conf_hi = np.exp(fcst[hi_col].values)

    return test["y"].values, y_pred, conf_lo, conf_hi, test["ds"].values


def run_prophet(train, test, interval_width=0.95):

    # cutoff: use the last 3 years only cause the old prices of BTC is completely different from now (old data can ruin the trend)
    cutoff = train["ds"].max() - pd.Timedelta(days=3 * 365)
    tr = train[train["ds"] >= cutoff].copy()
    tr["y"] = np.log1p(tr["y"])
    # use log1p cause log(0) raise error but log1p can handel it (log1p(0) = 0)
    cap_val, floor_val = np.log1p(200_000), np.log1p(1_000)
    tr["cap"], tr["floor"] = cap_val, floor_val

    
    # linear growth: assumes that the target variable increase at a constant rate over time
    # which can lead to unrealistic predictions as it allows the value to grow without bounds
    # logistic growth: ensures that predictions remain within a realistic range by defining an upper limit (cap) and a lower bound (floor)
    # which is more realistic for BTC
    # interval_width: controls the width of the confidence band drawn on the chart
    # use additive seasonality cause of log transform
    model = Prophet(
        growth="logistic",
        seasonality_mode="additive",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        n_changepoints=20,
        interval_width=interval_width,
    )
    model.fit(tr[["ds", "y", "cap", "floor"]])

    future = pd.DataFrame({"ds": test["ds"]})
    future["cap"], future["floor"] = cap_val, floor_val
    forecast = model.predict(future)

    # i faced many errors so this function handel it
    # it returns the data from log to actual price USD & cleans up any incorrect values
    def _inv(arr):
        return np.nan_to_num(np.expm1(arr), nan=0.0, posinf=0.0, neginf=0.0)

    return (
        np.nan_to_num(test["y"].values, nan=0.0),
        _inv(forecast["yhat"].values),
        _inv(forecast["yhat_lower"].values),
        _inv(forecast["yhat_upper"].values),
        test["ds"].values,
    )


def run_prophet_regressors(train, test, full_df, interval_width=0.95):
    
    # build features (lag_7, volatility_7) and if volume is available add it
    full_feat = build_regressors(full_df)
    regressor_cols = ["lag_7", "volatility_7"]
    if "log_volume" in full_feat.columns:
        regressor_cols.append("log_volume")

    # resplit after feature engineering
    # build features on the full dataset first then supply the test set regressor values in the future df
    split_date  = train["ds"].max()
    cutoff_date = split_date - pd.Timedelta(days=3 * 365)

    train_feat = full_feat[
        (full_feat["ds"] >= cutoff_date) & (full_feat["ds"] <= split_date)
    ].copy()
    test_feat = full_feat[full_feat["ds"] > split_date].copy()

    if len(test_feat) == 0:
        raise ValueError("No test rows survived feature engineering")

    train_feat = train_feat.copy()
    train_feat["y"] = np.log1p(train_feat["y"])
    cap_val, floor_val = np.log1p(200_000), np.log1p(1_000)
    train_feat["cap"], train_feat["floor"] = cap_val, floor_val

    model = Prophet(
        growth="logistic",
        seasonality_mode="additive",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        n_changepoints=20,
        interval_width=interval_width,
    )
    for col in regressor_cols:
        model.add_regressor(col, standardize=True)

    fit_cols = ["ds", "y", "cap", "floor"] + regressor_cols
    model.fit(train_feat[fit_cols])

    future = test_feat[["ds"] + regressor_cols].copy()
    future["cap"], future["floor"] = cap_val, floor_val
    forecast = model.predict(future)

    # merge on ds
    fcst = forecast.merge(test_feat[["ds", "y"]], on="ds", how="inner")

    def _inv(arr):
        return np.nan_to_num(np.expm1(arr), nan=0.0, posinf=0.0, neginf=0.0)

    return (
        np.nan_to_num(fcst["y"].values, nan=0.0),
        _inv(fcst["yhat"].values),
        _inv(fcst["yhat_lower"].values),
        _inv(fcst["yhat_upper"].values),
        fcst["ds"].values,
    )


def run_random_forest(train, test, full_df, n_estimators=300):

    full_feat = build_regressors(full_df)
    full_feat = full_feat.copy()
    # day_of_week: captures weekend
    # day_of_year: captures yearly seasonality
    full_feat["day_of_week"] = full_feat["ds"].dt.dayofweek
    full_feat["day_of_year"] = full_feat["ds"].dt.dayofyear

    feature_cols = ["lag_7", "volatility_7", "day_of_week", "day_of_year"]
    if "log_volume" in full_feat.columns:
        feature_cols.append("log_volume")

    split_date  = train["ds"].max()
    cutoff_date = split_date - pd.Timedelta(days=3 * 365)

    train_feat = full_feat[
        (full_feat["ds"] >= cutoff_date) & (full_feat["ds"] <= split_date)
    ].copy()
    test_feat = full_feat[full_feat["ds"] > split_date].copy()

    if len(test_feat) == 0:
        raise ValueError("No test rows survived feature engineering")

    X_train = train_feat[feature_cols].values
    y_train = np.log1p(train_feat["y"].values)
    X_test = test_feat[feature_cols].values
    # raw price for metrics
    y_true = test_feat["y"].values

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = np.expm1(rf.predict(X_test))
    # to show the most important features of rf on dashboard
    importances = dict(zip(feature_cols, rf.feature_importances_))

    return y_true, y_pred, test_feat["ds"].values, importances


# baseline model to compare (predicts every future day as the last observed training price)
def run_naive(train, test):
    last_value = train["y"].iloc[-1]
    y_pred = np.full(len(test), last_value)
    return test["y"].values, y_pred, test["ds"].values




# forcasting future (for {horizon} days)
# train models on the full dataset

def forecast_future_arima(df, horizon):
    
    df_train = df[["ds", "y"]].copy()
    df_train["unique_id"] = "BTC"
    df_train["y"] = np.log(df_train["y"])
    df_train = df_train[["unique_id", "ds", "y"]]

    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    sf.fit(df_train)
    fcst = sf.predict(h=horizon, level=[80, 95])

    y_pred  = np.exp(fcst["AutoARIMA"].values)
    lo_col = "AutoARIMA-lo-95" if "AutoARIMA-lo-95" in fcst.columns else "AutoARIMA-lo-80"
    hi_col = "AutoARIMA-hi-95" if "AutoARIMA-hi-95" in fcst.columns else "AutoARIMA-hi-80"
    conf_lo = np.exp(fcst[lo_col].values)
    conf_hi = np.exp(fcst[hi_col].values)

    last_date   = pd.Timestamp(df["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    return y_pred, conf_lo, conf_hi, future_dates


def forecast_future_prophet(df, horizon, interval_width=0.95):

    cutoff = df["ds"].max() - pd.Timedelta(days=3 * 365)
    tr = df[df["ds"] >= cutoff].copy()
    tr["y"] = np.log1p(tr["y"])
    cap_val, floor_val = np.log1p(200_000), np.log1p(1_000)
    tr["cap"], tr["floor"] = cap_val, floor_val

    model = Prophet(
        growth="logistic",
        seasonality_mode="additive",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        n_changepoints=20,
        interval_width=interval_width,
    ).fit(tr[["ds", "y", "cap", "floor"]])

    last_date    = pd.Timestamp(df["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                  periods=horizon, freq="D")
    future = pd.DataFrame({"ds": future_dates})
    future["cap"], future["floor"] = cap_val, floor_val
    forecast = model.predict(future)

    def _inv(arr):
        return np.nan_to_num(np.expm1(arr), nan=0.0, posinf=0.0, neginf=0.0)

    return (
        _inv(forecast["yhat"].values),
        _inv(forecast["yhat_lower"].values),
        _inv(forecast["yhat_upper"].values),
        future_dates,
    )