from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Temporal Shopping Analytics", page_icon="🛒", layout="wide")
st.title("🛒 Temporal Shopping Pattern & Resource Optimization")
st.caption("Retail demand analytics • clustering • forecasting • operational recommendations")

needed = ["hourly_demand.csv", "daily_demand.csv", "hourly_clusters.csv", "forecast_test.csv", "forecast_metrics.json"]
missing = [f for f in needed if not (OUT / f).exists()]
if missing:
    st.warning("Project outputs are missing. First run `python src/prepare_data.py` and `python src/train_models.py`.")
    st.write("Missing:", ", ".join(missing))
    st.stop()

hourly = pd.read_csv(OUT / "hourly_clusters.csv")
daily = pd.read_csv(OUT / "daily_demand.csv", parse_dates=["date"])
forecast = pd.read_csv(OUT / "forecast_test.csv", parse_dates=["date"])
metrics = json.loads((OUT / "forecast_metrics.json").read_text(encoding="utf-8"))

c1, c2, c3 = st.columns(3)
c1.metric("Total orders (aggregated days)", f"{int(daily['orders'].sum()):,}")
c2.metric("Peak observed hour", f"{int(hourly.loc[hourly['orders'].idxmax(), 'hour']):02d}:00")
c3.metric("Forecast MAE", f"{metrics['MAE']:.2f} orders/day")

st.subheader("Hourly shopping pattern")
fig, ax = plt.subplots()
ax.bar(hourly["hour"], hourly["orders"])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Distinct invoices")
ax.set_xticks(range(0, 24))
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("Demand patterns discovered by K-Means")
st.dataframe(hourly[["hour", "orders", "units_sold", "revenue", "demand_pattern"]], use_container_width=True)
if metrics.get("hourly_cluster_silhouette") is not None:
    st.write(f"Silhouette score: **{metrics['hourly_cluster_silhouette']:.3f}**")

st.subheader("Daily orders and seven-day trend")
fig, ax = plt.subplots()
ax.plot(daily["date"], daily["orders"], label="Daily orders", linewidth=1)
ax.plot(daily["date"], daily["orders"].rolling(7).mean(), label="7-day rolling mean")
ax.set_xlabel("Date")
ax.set_ylabel("Orders")
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("Forecast performance on the chronological holdout")
fig, ax = plt.subplots()
ax.plot(forecast["date"], forecast["orders"], label="Actual")
ax.plot(forecast["date"], forecast["predicted_orders"], label="Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Orders")
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.2f}")
m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
m3.metric("R²", "N/A" if metrics["R2"] is None else f"{metrics['R2']:.3f}")
st.caption(f"Chronological test window: {metrics['test_start']} to {metrics['test_end']}")

st.subheader("Operational recommendations (illustrative)")
peak_hour = int(hourly.loc[hourly["orders"].idxmax(), "hour"])
low_hour = int(hourly.loc[hourly["orders"].idxmin(), "hour"])
st.markdown(f"""
- **Staffing:** review coverage around **{peak_hour:02d}:00** and nearby periods using local operational data; do not schedule solely from one historical peak.
- **Low-demand windows:** **{low_hour:02d}:00** was the lowest observed hour in this dataset. Consider training, replenishment, or maintenance tasks during low-demand windows where operationally suitable.
- **Inventory:** use forecasts at product/category level before setting reorder quantities. This starter dataset's daily total-order forecast alone is not enough to calculate SKU-level safety stock.
- **Promotions:** this dataset does not include promotion flags. Test promotion timing only after joining a reliable promotion/calendar dataset and evaluating a controlled comparison.
""")
st.info("The UCI dataset represents a historical online retailer, not a physical supermarket. Recommendations are demonstration outputs and should be validated against a target store's own data.")
