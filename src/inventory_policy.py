from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_inventory_policy(
    history_df: pd.DataFrame,
    service_level: float = 0.95,
    review_period_days: int = 7
) -> pd.DataFrame:
    if not (0.5 <= service_level < 0.999):
        raise ValueError("service_level should be between 0.5 and 0.999")

    z = float(norm.ppf(service_level))
    rows = []

    for sku, hist in history_df.groupby("sku"):
        hist = hist.sort_values("date")
        avg_daily = float(hist["units_sold"].tail(30).mean())
        sigma_daily = float(hist["units_sold"].tail(30).std(ddof=0))
        sigma_daily = max(sigma_daily, 1e-6)

        lead_time = max(1.0, float(hist["lead_time_days"].iloc[-1]))
        on_hand = float(hist["on_hand_inventory"].iloc[-1])
        unit_cost = float(hist["unit_cost"].iloc[-1])

        demand_during_lead = avg_daily * lead_time
        safety_stock = z * sigma_daily * np.sqrt(lead_time)
        reorder_point = demand_during_lead + safety_stock

        target_days = lead_time + review_period_days
        target_stock = avg_daily * target_days + safety_stock
        reorder_qty = max(0.0, target_stock - on_hand)

        rows.append({
            "sku": sku,
            "avg_daily_demand": avg_daily,
            "daily_demand_std": sigma_daily,
            "lead_time_days": lead_time,
            "service_level": service_level,
            "z_score": z,
            "safety_stock": float(safety_stock),
            "reorder_point": float(reorder_point),
            "on_hand_inventory": on_hand,
            "reorder_qty": float(reorder_qty),
            "unit_cost": unit_cost,
            "inventory_gap": float(reorder_point - on_hand),
        })

    return pd.DataFrame(rows).sort_values("sku").reset_index(drop=True)
