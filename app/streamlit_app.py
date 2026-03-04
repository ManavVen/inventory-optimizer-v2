from __future__ import annotations
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ingest import load_sales_data, train_test_split_by_date
from src.forecast import train_forecast_model, evaluate_model, forecast_next_n_days
from src.inventory_policy import calculate_inventory_policy
from src.simulate import monte_carlo_stockout_risk
from src.recommend import build_recommendations

st.set_page_config(page_title="Inventory Optimizer V2", layout="wide")
st.title("📦 Inventory Optimizer V2 — Forecast + Reorder + Risk")

default_path = os.path.join("data","raw","sales.csv")
data_path = st.text_input("CSV path", value=default_path)

c1, c2, c3 = st.columns(3)
service_level = c1.slider("Service level", 0.80, 0.99, 0.95, 0.01)
forecast_days = c2.selectbox("Forecast horizon (days)", [14, 30, 45], index=1)
n_sims = c3.selectbox("Monte Carlo simulations", [200, 500, 1000], index=1)

run = st.button("Run V2 pipeline")

if run:
    with st.spinner("Running pipeline..."):
        df = load_sales_data(data_path)
        train_df, test_df = train_test_split_by_date(df, test_days=30)

        model = train_forecast_model(train_df)
        metrics = evaluate_model(model, test_df)

        fcst = forecast_next_n_days(model, df, n_days=int(forecast_days))
        policy = calculate_inventory_policy(df, service_level=float(service_level))
        risk = monte_carlo_stockout_risk(df, policy, n_sims=int(n_sims))
        recs = build_recommendations(policy, risk)

        os.makedirs(os.path.join("data","processed"), exist_ok=True)
        recs.to_csv(os.path.join("data","processed","recommendations.csv"), index=False)

    st.subheader("Forecast Accuracy (holdout last 30 days)")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{metrics['mae']:.2f}")
    m2.metric("MAPE", f"{metrics['mape']:.2f}%")

    st.subheader("Top Recommendations")
    st.dataframe(recs.head(25), use_container_width=True)

    st.subheader("Risk View: Top 20 SKUs by Stockout Probability")
    fig_risk = px.bar(
        recs.sort_values("stockout_prob", ascending=False).head(20),
        x="sku", y="stockout_prob", color="action",
        title="Simulated Stockout Probability"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("SKU Drilldown: Demand History + Forecast")
    sku = st.selectbox("Select SKU", sorted(df["sku"].unique()))
    hist_sku = df[df["sku"] == sku].sort_values("date").tail(120)
    fcst_sku = fcst[fcst["sku"] == sku].sort_values("date")

    hist_plot = hist_sku.rename(columns={"units_sold": "value"})[["date","value"]].assign(series="history")
    fcst_plot = fcst_sku.rename(columns={"forecast_units": "value"})[["date","value"]].assign(series="forecast")
    combo = pd.concat([hist_plot, fcst_plot], ignore_index=True)

    fig_line = px.line(combo, x="date", y="value", color="series", title=f"{sku}: history vs forecast")
    st.plotly_chart(fig_line, use_container_width=True)

    st.success("✅ V2 complete — exported to data/processed/recommendations.csv")
