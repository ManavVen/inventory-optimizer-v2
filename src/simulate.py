from __future__ import annotations
import numpy as np
import pandas as pd

def monte_carlo_stockout_risk(
    history_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    n_sims: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = []

    for _, row in policy_df.iterrows():
        sku = row["sku"]
        hist = history_df[history_df["sku"] == sku].sort_values("date")
        if hist.empty:
            continue

        mu = float(hist["units_sold"].tail(30).mean())
        sigma = float(hist["units_sold"].tail(30).std(ddof=0))
        sigma = max(sigma, 1.0)

        lead_mu = max(1.0, float(row["lead_time_days"]))
        lead_sigma = max(1.0, lead_mu * 0.2)

        on_hand = float(row["on_hand_inventory"])
        reorder_qty = float(row["reorder_qty"])

        stockouts = 0
        demands = []

        for _ in range(n_sims):
            lead_time = int(max(1, rng.normal(lead_mu, lead_sigma)))
            daily = rng.normal(mu, sigma, size=lead_time)
            total = float(np.maximum(daily, 0).sum())
            demands.append(total)
            if total > (on_hand + reorder_qty):
                stockouts += 1

        out.append({
            "sku": sku,
            "stockout_prob": float(stockouts / n_sims),
            "p95_leadtime_demand": float(np.percentile(demands, 95)),
            "expected_leadtime_demand": float(np.mean(demands))
        })

    return pd.DataFrame(out).sort_values("sku").reset_index(drop=True)
