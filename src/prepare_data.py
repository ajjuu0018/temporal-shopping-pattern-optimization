from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# UCI's original Excel workbook usually has a sheet named "Online Retail".
candidates = [DATA / "Online Retail.xlsx", DATA / "online_retail.xlsx"]
source = next((p for p in candidates if p.exists()), None)
if source is None:
    raise FileNotFoundError(
        "Dataset not found. Download Online Retail.xlsx from "
        "https://archive.ics.uci.edu/dataset/352/online+retail and place it in data/"
    )

print(f"Reading {source} ...")
df = pd.read_excel(source)
df.columns = [str(c).strip() for c in df.columns]

required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

rows_before = len(df)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df = df.drop_duplicates()
df = df.dropna(subset=["InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"])

# Cancellation invoices and negative quantities are returns, not positive demand.
# Keep a separate raw copy if you later want returns/refunds analysis.
invoice_text = df["InvoiceNo"].astype(str).str.strip().str.upper()
df = df[~invoice_text.str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

df["revenue"] = df["Quantity"] * df["UnitPrice"]
df["date"] = df["InvoiceDate"].dt.floor("D")
df["hour"] = df["InvoiceDate"].dt.hour
df["weekday"] = df["InvoiceDate"].dt.dayofweek
df["weekday_name"] = df["InvoiceDate"].dt.day_name()
df["month"] = df["InvoiceDate"].dt.month
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["year_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# A transaction count is the number of distinct invoices, not line-item rows.
hourly = (
    df.groupby("hour")
      .agg(orders=("InvoiceNo", "nunique"),
           units_sold=("Quantity", "sum"),
           revenue=("revenue", "sum"))
      .reset_index()
)
hourly.to_csv(OUT / "hourly_demand.csv", index=False)

daily = (
    df.groupby("date")
      .agg(orders=("InvoiceNo", "nunique"),
           units_sold=("Quantity", "sum"),
           revenue=("revenue", "sum"))
      .asfreq("D")
)
# Dates without transactions are set to zero only after aggregation to a daily calendar.
daily.index.name = "date"
daily[["orders", "units_sold", "revenue"]] = daily[["orders", "units_sold", "revenue"]].fillna(0)
daily = daily.reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily["weekday"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)
daily.to_csv(OUT / "daily_demand.csv", index=False)

weekday_hour = (
    df.groupby(["weekday", "hour"])
      .agg(orders=("InvoiceNo", "nunique"), units_sold=("Quantity", "sum"))
      .reset_index()
)
weekday_hour.to_csv(OUT / "weekday_hour_demand.csv", index=False)

print(f"Rows loaded: {rows_before:,}")
print(f"Rows retained for positive-demand analysis: {len(df):,}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Saved aggregated data to {OUT}")
