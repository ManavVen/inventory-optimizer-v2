from __future__ import annotations
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_month"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    return out

def add_lag_roll_features(df: pd.DataFrame, lags=(1, 7, 14), rolls=(7, 14)) -> pd.DataFrame:
    out = df.copy().sort_values(["sku", "date"])
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("sku")["units_sold"].shift(lag)

    # rolling stats use shifted series so we don't leak the current day
    shifted = out.groupby("sku")["units_sold"].shift(1)
    for w in rolls:
        out[f"roll_mean_{w}"] = shifted.rolling(w).mean().reset_index(level=0, drop=True)
        out[f"roll_std_{w}"] = shifted.rolling(w).std(ddof=0).reset_index(level=0, drop=True)

    return out
