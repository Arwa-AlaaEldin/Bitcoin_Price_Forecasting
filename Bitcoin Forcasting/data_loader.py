import io
import pandas as pd
import numpy as np

possible_date_cols = ["Date", "Timestamp", "date", "timestamp", "time", "Time"]
possible_price_cols = ["Price", "Close", "Open", "High", "Low", "close", "open", "high", "low", "price"]

# scan columns for a known date column name
def detect_date_col(df):
    for col in possible_date_cols:
        if col in df.columns:
            return col
    return None


# parse a csv from raw bytes, validate it, and return clean DataFrame
def load_data(file_bytes, price_col):
  
    load_warnings = []

    # read csv file
    try:
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Couldn't read csv file: {exc}") from exc

    # detect cols (date, price)
    date_col = detect_date_col(df_raw)
    if date_col is None:
        raise ValueError(
            f"No date column found, expected one of: {possible_date_cols}"
            f"Columns in file: {list(df_raw.columns)}"
        )

    if price_col not in df_raw.columns:
        available = [c for c in df_raw.columns if c in possible_price_cols]
        raise ValueError(
            f"Price column '{price_col}' not found"
            f"Available price columns: {available if available else 'none detected'}"
        )

    # convert date col to date time
    # errors="coerce": when date is not a time (NaT) instead of crashing
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")

    # clean price value
    # if price like that 1,234.56 --> remove ',' 1234,56
    price_clean = df_raw[price_col].astype(str).str.replace(",", "", regex=False)
    df_raw[price_col] = pd.to_numeric(price_clean, errors="coerce")

    # drop nan values (missing dates)
    n_bad_dates = df_raw[date_col].isna().sum()
    if n_bad_dates > 0:
        load_warnings.append(f"Removed {n_bad_dates} rows with missing date values")
        df_raw = df_raw.dropna(subset=[date_col])

    # drop nan values (missing prices)
    n_bad_prices = df_raw[price_col].isna().sum()
    if n_bad_prices > 0:
        load_warnings.append(f"Removed {n_bad_prices} rows with missing price values")
        df_raw = df_raw.dropna(subset=[price_col])


    # that is for prophet regressor model
    # check if there is a Volume col in ds and it has a real values
    # so can build a reg for the model without issus
    has_real_volume = ("Volume" in df_raw.columns and df_raw["Volume"].notna().sum() > 0)

    # create the ['ds', 'y'] DataFrame
    df = pd.DataFrame({
        "ds": df_raw[date_col].values,
        "y":  df_raw[price_col].values,
    })
    if has_real_volume:
        df["volume"] = df_raw["Volume"].values

    # sort by 'ds' and reset the index
    df = df.sort_values("ds").reset_index(drop=True)

    # remove duplicate dates
    n_dups = df["ds"].duplicated().sum()
    if n_dups > 0:
        load_warnings.append(f"Removed {n_dups} rows with duplicate dates")
        df = df.drop_duplicates(subset="ds", keep="first").reset_index(drop=True)

    # check for large gaps
    # cause this may indicate missing trading-day data
    df["ds"] = pd.to_datetime(df["ds"])
    max_gap = int(df["ds"].diff().dt.days.dropna().max())
    if max_gap > 7:
        load_warnings.append(f"Largest gap between rows: {max_gap} days")

    # check for min len of df
    if len(df) < 60:
        raise ValueError(f"Dataset too short ({len(df)} rows), at least 60 rows are required")

    return df, load_warnings


def train_test_split(df, test_days=90):
    if test_days >= len(df):
        raise ValueError(f"test_days ({test_days}) must be less than dataset length ({len(df)})")
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test