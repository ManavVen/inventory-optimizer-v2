from __future__ import annotations
import pandas as pd

def build_recommendations(policy_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    df = policy_df.merge(risk_df, on="sku", how="left")
    df["stockout_prob"] = df["stockout_prob"].fillna(0.0)

    # cost-ish proxy: probability * unit cost * expected lead-time demand
    df["risk_cost_proxy"] = df["stockout_prob"] * df["unit_cost"] * df["expected_leadtime_demand"].fillna(0.0)

    def action(row):
        if row["on_hand_inventory"] < row["reorder_point"]:
            return "URGENT_REORDER" if row["stockout_prob"] >= 0.30 else "REORDER_SOON"
        if row["stockout_prob"] >= 0.30:
            return "MONITOR_HIGH_RISK"
        return "OK"

    df["action"] = df.apply(action, axis=1)
    df["priority_score"] = (
        0.6 * df["stockout_prob"] +
        0.25 * (df["inventory_gap"] > 0).astype(float) +
        0.15 * (df["reorder_qty"] > 0).astype(float)
    )

    cols = [
        "sku","action","priority_score","stockout_prob",
        "reorder_point","on_hand_inventory","reorder_qty","safety_stock",
        "risk_cost_proxy","lead_time_days","unit_cost"
    ]
    return df[cols].sort_values(["priority_score","risk_cost_proxy"], ascending=False).reset_index(drop=True)
