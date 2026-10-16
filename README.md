# Temporal Shopping Pattern & Resource Optimization

An educational retail analytics project using timestamped transactions to:
- analyze hourly/daily/monthly demand patterns,
- cluster hourly profiles into low/normal/high-demand patterns,
- forecast daily order demand using a time-aware train/test split,
- generate illustrative staffing and inventory recommendations.

## Dataset

Recommended starter dataset: UCI Online Retail (real transaction data, CC BY 4.0).
Official page: https://archive.ics.uci.edu/dataset/352/online+retail

Download `Online Retail.xlsx` from UCI and put it in `data/Online Retail.xlsx`.

Important limitation: this dataset is from a UK online retailer and does not contain staffing, stock-on-hand, or promotion columns. Therefore, the project can make demand-based illustrative recommendations, but cannot claim measured staffing savings or promotion effectiveness. For a grocery-specific daily sales/promotion extension, consider Kaggle Store Sales - Time Series Forecasting:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run pipeline

```powershell
python src/prepare_data.py
python src/train_models.py
streamlit run app.py
```

## Expected files

- `data/Online Retail.xlsx` (download manually from UCI)
- `outputs/hourly_demand.csv`
- `outputs/daily_demand.csv`
- `outputs/cluster_profiles.csv`
- `outputs/forecast_metrics.json`
- `outputs/forecast_test.csv`
- `models/hourly_kmeans.joblib`
- `models/daily_demand_model.joblib`

## Methods

- Preprocessing: parse dates, remove duplicate rows, exclude cancellation invoices and non-positive quantities/prices from demand analysis, handle missing timestamps.
- Clustering: K-Means on standardized hour-of-day profiles; silhouette score is reported when meaningful.
- Forecasting: Random Forest regression with a chronological split. Features use calendar/time and lagged demand; evaluation uses MAE, RMSE, and R².
- Recommendations: demand tiers are relative to the dataset's own hourly/daily demand distribution, not universal operational thresholds.

## Dataset citation

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0.
