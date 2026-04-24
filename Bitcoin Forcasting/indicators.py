import pandas as pd

# market cycles:
# 1. short-term trend: One Month (20 days)
# 2. medium-term trend: 2-3 Months (50 days)
# 3. long-term trend: One Year (200 days)
win_size = {
    "One Month (20 days)": 20,
    "2-3 Months (50 days)": 50,
    "One Year (200 days)": 200,
}


def add_sma(df, window):
    df = df.copy()
    df[f"SMA_{window}"] = df["y"].rolling(window=window).mean()
    return df


def apply_indicators(df, selected):
    for label in selected:
        window = win_size[label]
        df = add_sma(df, window)
    return df


# returns all col names that start with 'SMA_'
# the visualizer will use this to know which lines to draw
def get_indicator_columns(df):
    return [col for col in df.columns if col.startswith("SMA_")]