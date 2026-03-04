from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.features import add_time_features, add_lag_roll_features

FEATURE_COLS = [
    "day_of_week","day_of_month","month",
    "lag_1","lag_7","lag_14",
    "roll_mean_7","roll_std_7",
    "roll_mean_14","roll_std_14",
    "lead_time_days","unit_cost","on_hand_inventory",
    "sku_code"
]

def _prep(df: pd.DataFrame):
    w = df.copy()
    w = add_time_features(w)
    w = add_lag_roll_features(w)
    w["sku_code"] = w["sku"].astype("category").cat.codes.astype(int)
    w = w.dropna(subset=["lag_1","roll_mean_7","roll_mean_14"]).copy()
    X = w[FEATURE_COLS]
    y = w["units_sold"].astype(float)
    return X, y

def train_forecast_model(train_df: pd.DataFrame, random_state: int = 42):
    X, y = _prep(train_df)
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2
    )
    model.fit(X, y)
    return model

def evaluate_model(model: RandomForestRegressor, test_df: pd.DataFrame) -> dict:
    X, y = _prep(test_df)
    preds = np.maximum(model.predict(X), 0.0)
    mae = float(mean_absolute_error(y, preds))
    mape = float(np.mean(np.abs((y - preds) / np.maximum(y, 1e-6))) * 100.0)
    return {"mae": mae, "mape": mape}

def forecast_next_n_days(model: RandomForestRegressor, history_df: pd.DataFrame, n_days: int = 30) -> pd.DataFrame:
    hist = history_df.copy().sort_values(["sku","date"]).reset_index(drop=True)
    skus = sorted(hist["sku"].unique())
    out_rows = []

    for _ in range(n_days):
        next_date = hist["date"].max() + pd.Timedelta(days=1)
        new_rows = []

        for sku in skus:
            sku_hist = hist[hist["sku"] == sku].sort_values("date")
            last = sku_hist.iloc[-1]

            lag_1 = float(sku_hist["units_sold"].iloc[-1])
            lag_7 = float(sku_hist["units_sold"].iloc[-7]) if len(sku_hist) >= 7 else lag_1
            lag_14 = float(sku_hist["units_sold"].iloc[-14]) if len(sku_hist) >= 14 else lag_7

            tail7 = sku_hist["units_sold"].tail(7)
            tail14 = sku_hist["units_sold"].tail(14)
            roll_mean_7 = float(tail7.mean())
            roll_std_7 = float(tail7.std(ddof=0)) if len(tail7) > 1 else 0.0
            roll_mean_14 = float(tail14.mean())
            roll_std_14 = float(tail14.std(ddof=0)) if len(tail14) > 1 else 0.0

            sku_code = int(pd.Series([sku]).astype("category").cat.codes.iloc[0])

            X = pd.DataFrame([{
                "day_of_week": next_date.dayofweek,
                "day_of_month": next_date.day,
                "month": next_date.month,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "roll_mean_7": roll_mean_7,
                "roll_std_7": roll_std_7,
                "roll_mean_14": roll_mean_14,
                "roll_std_14": roll_std_14,
                "lead_time_days": float(last["lead_time_days"]),
                "unit_cost": float(last["unit_cost"]),
                "on_hand_inventory": float(last["on_hand_inventory"]),
                "sku_code": sku_code,
            }])

            yhat = max(0.0, float(model.predict(X)[0]))

            out_rows.append({"date": next_date, "sku": sku, "forecast_units": yhat})
            new_rows.append({
                "date": next_date,
                "sku": sku,
                "units_sold": yhat,  # used only for rolling forward features
                "lead_time_days": float(last["lead_time_days"]),
                "unit_cost": float(last["unit_cost"]),
                "on_hand_inventory": max(0.0, float(last["on_hand_inventory"]) - yhat),
            })

        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)

    return pd.DataFrame(out_rows).sort_values(["sku","date"]).reset_index(drop=True)
