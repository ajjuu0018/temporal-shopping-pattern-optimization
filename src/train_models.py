from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

hourly_path = OUT / "hourly_demand.csv"
daily_path = OUT / "daily_demand.csv"
if not hourly_path.exists() or not daily_path.exists():
    raise FileNotFoundError("Run `python src/prepare_data.py` first.")

# 1) Cluster hour-of-day demand profiles. Each hour is represented by the
# observed orders, units, and revenue aggregated over the full dataset.
hourly = pd.read_csv(hourly_path)
cluster_features = hourly[["orders", "units_sold", "revenue"]].copy()
cluster_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=20))
])
hourly["cluster"] = cluster_pipe.fit_predict(cluster_features)
scaled = cluster_pipe.named_steps["scale"].transform(cluster_features)
labels = hourly["cluster"].to_numpy()
sil = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 and len(labels) > 3 else None

profiles = hourly.groupby("cluster")[["orders", "units_sold", "revenue"]].mean()
# Label clusters by mean orders for easier interpretation, not by raw KMeans ID.
ranked = profiles["orders"].sort_values().index.tolist()
name_map = {ranked[0]: "Low demand", ranked[1]: "Typical demand", ranked[2]: "High demand"}
hourly["demand_pattern"] = hourly["cluster"].map(name_map)
hourly.to_csv(OUT / "hourly_clusters.csv", index=False)
profiles.assign(demand_pattern=[name_map[i] for i in profiles.index]).to_csv(OUT / "cluster_profiles.csv")
joblib.dump(cluster_pipe, MODELS / "hourly_kmeans.joblib")

# 2) Daily order forecasting. Chronological split; lag features only use prior
# observations, avoiding random shuffling and target leakage.
daily = pd.read_csv(daily_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
daily["lag_1"] = daily["orders"].shift(1)
daily["lag_7"] = daily["orders"].shift(7)
daily["rolling_mean_7"] = daily["orders"].shift(1).rolling(7, min_periods=7).mean()
daily["rolling_mean_28"] = daily["orders"].shift(1).rolling(28, min_periods=28).mean()
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["weekday"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
model_features = ["day_of_year", "weekday", "month", "is_weekend",
                  "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28"]
model_data = daily.dropna(subset=model_features + ["orders"]).copy()

if len(model_data) < 60:
    raise ValueError("Not enough daily observations for lagged forecasting; inspect the dataset.")

split_idx = int(len(model_data) * 0.8)
train, test = model_data.iloc[:split_idx], model_data.iloc[split_idx:]
X_train, y_train = train[model_features], train["orders"]
X_test, y_test = test[model_features], test["orders"]

model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = np.maximum(0, model.predict(X_test))

metrics = {
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "test_start": str(test["date"].min().date()),
    "test_end": str(test["date"].max().date()),
    "MAE": float(mean_absolute_error(y_test, pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "R2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
    "hourly_cluster_silhouette": sil,
}
with open(OUT / "forecast_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

test_result = test[["date", "orders"]].copy()
test_result["predicted_orders"] = pred
test_result.to_csv(OUT / "forecast_test.csv", index=False)
joblib.dump({"model": model, "features": model_features}, MODELS / "daily_demand_model.joblib")

print("Training complete.")
print(json.dumps(metrics, indent=2))
print(f"Outputs: {OUT}")
