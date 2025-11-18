# app_aqi.py
# Streamlit app for NCAP AQI city analysis (EDA, Forecasting, Clustering, Seasonal Insights)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="City AQI Analysis", layout="wide")

st.title("City AQI — EDA, Forecasting, Clustering & Seasonal Insights")
st.markdown(
    "Upload the NCAP / UrbanEmissions CSV for a single city (2015–2023). "
    "Expected columns (map if necessary): `Date`, `City`, `No. Stations`, `Air Quality`, "
    "`Index Value`, `Prominent Pollutant`."
)

# -------------------------
# File upload / load
# -------------------------
uploaded = st.file_uploader("Upload CSV / TXT / ZIP (semicolon or comma separated)", type=["csv", "txt", "zip"])
local_path = st.text_input("Or local CSV path (optional)", value="")
use_sample = st.checkbox("Use default /mnt/data/aqi.csv if present", value=False)

def detect_sep_text(text: str):
    # crude separator detection using first non-empty line
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ","
    header = lines[0]
    return ";" if header.count(";") > header.count(",") else ","

@st.cache_data
def read_buffer(buffer, try_sep=None):
    # buffer may be StreamlitUploadedFile or bytes
    if hasattr(buffer, "getvalue"):
        raw = buffer.getvalue()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        sep = try_sep or detect_sep_text(text[:2000])
        return pd.read_csv(StringIO(text), sep=sep)
    else:
        # buffer is a path
        try:
            return pd.read_csv(buffer, sep=";")
        except Exception:
            return pd.read_csv(buffer, sep=",")

@st.cache_data
def load_dataframe(uploaded_file, local_path, use_sample_flag):
    # priority: upload -> sample -> local_path
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".zip"):
                b = uploaded_file.getvalue()
                z = zipfile.ZipFile(BytesIO(b))
                csv_names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
                if not csv_names:
                    return pd.DataFrame()
                with z.open(csv_names[0]) as f:
                    raw = f.read().decode("utf-8", errors="replace")
                    sep = detect_sep_text(raw[:2000])
                    return pd.read_csv(StringIO(raw), sep=sep)
            else:
                return read_buffer(uploaded_file)
        except Exception:
            return pd.DataFrame()
    if use_sample_flag:
        # try default path
        for sep in [";", ","]:
            try:
                df = pd.read_csv("/mnt/data/aqi.csv", sep=sep)
                if not df.empty:
                    return df
            except Exception:
                pass
    if local_path:
        try:
            return read_buffer(local_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

with st.spinner("Loading data..."):
    df = load_dataframe(uploaded, local_path, use_sample)

if df.empty:
    st.info("Upload a CSV/TXT/ZIP or provide local path (or enable the sample file).")
    st.stop()

# -------------------------
# Preview and map columns
# -------------------------
st.subheader("Preview (first rows)")
st.dataframe(df.head())

# normalize column names (strip)
df.columns = [str(c).strip() for c in df.columns]

# auto-detect core columns
col_map = {}
for c in df.columns:
    lc = c.lower()
    if "date" in lc and "time" not in lc:
        col_map["Date"] = c
    if "city" == lc or "city" in lc:
        col_map["City"] = c
    if "station" in lc:
        col_map["No. Stations"] = c
    if "index" in lc and ("value" in lc or "index" in lc):
        col_map["Index Value"] = c
    if "prominent" in lc and "pollut" in lc:
        col_map["Prominent Pollutant"] = c
    if "air quality" in lc or (lc in ["air quality","air_quality"]):
        col_map["Air Quality"] = c

st.write("Detected column mapping (you can override):")
mapped = {}
for key in ["Date","City","No. Stations","Air Quality","Index Value","Prominent Pollutant"]:
    default = col_map.get(key, "")
    mapped[key] = st.selectbox(f"{key} column", options=[""] + list(df.columns), index=(list(df.columns).index(default)+1 if default in df.columns else 0))

# use selections if provided
for k, v in mapped.items():
    if v:
        col_map[k] = v

# required columns
if "Date" not in col_map or "Index Value" not in col_map:
    st.error("Please map at least the 'Date' and 'Index Value' columns.")
    st.stop()

# -------------------------
# Parse date and index value
# -------------------------
df["_date_parsed_"] = pd.to_datetime(df[col_map["Date"]].astype(str), dayfirst=True, errors="coerce")
df = df.dropna(subset=["_date_parsed_"]).copy()
df = df.set_index("_date_parsed_").sort_index()

df["_index_val_"] = pd.to_numeric(df[col_map["Index Value"]].astype(str).str.replace(",", "."), errors="coerce")

# optional: filter city
if "City" in col_map and col_map["City"] in df.columns:
    cities = sorted(df[col_map["City"]].dropna().unique().tolist())
    city_choice = st.selectbox("Select city to analyze", options=["All"] + cities, index=1 if cities else 0)
    if city_choice != "All":
        df = df[df[col_map["City"]] == city_choice]

# daily series (mean)
series = df["_index_val_"].resample("D").mean()

st.write(f"Data loaded: {len(df)} rows, date range {series.index.min().date()} — {series.index.max().date()}")

# -------------------------
# Task 1 — EDA
# -------------------------
st.header("Task 1 — EDA")
st.subheader("Time-series trend of Index Value (daily average)")
fig, ax = plt.subplots(figsize=(12, 4))
series.plot(ax=ax)
ax.set_ylabel("Index Value (AQI)")
st.pyplot(fig)

st.write("Summary statistics")
st.write(series.describe())

# missing and abnormal
missing = int(series.isna().sum())
st.write(f"Missing days (daily resample): {missing}")

z_thresh = st.slider("Z-score threshold to flag abnormal days", 2.5, 6.0, 3.5)
valid = series.dropna()
if len(valid) > 1:
    zvals = np.abs(stats.zscore(valid.values))
    abnormal_dates = valid.index[zvals > z_thresh]
    st.write(f"Abnormal days flagged: {len(abnormal_dates)}")
    if len(abnormal_dates) > 0:
        st.dataframe(pd.DataFrame({"Date": abnormal_dates, "Index Value": valid.loc[abnormal_dates].values}).set_index("Date").head(25))
else:
    st.info("Not enough non-null values to compute z-score.")

# distribution of prominent pollutant
st.subheader("Distribution of Prominent Pollutant")
if "Prominent Pollutant" in col_map and col_map["Prominent Pollutant"] in df.columns:
    pollutant_counts = df[col_map["Prominent Pollutant"]].fillna("Unknown").value_counts().head(25)
    fig2, ax2 = plt.subplots(figsize=(10,3))
    pollutant_counts.plot.bar(ax=ax2)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
else:
    st.info("Prominent Pollutant column not mapped or not available.")

# -------------------------
# Task 2 — Forecasting (next-day)
# -------------------------
st.header("Task 2 — Forecasting next-day Index Value")

def make_lag_features(s: pd.Series, lags=[1,2,3,7,14]):
    df_l = pd.DataFrame({"y": s})
    for lag in lags:
        df_l[f"lag_{lag}"] = df_l["y"].shift(lag)
    df_l["dayofyear"] = df_l.index.dayofyear
    df_l["month"] = df_l.index.month
    return df_l

lags_input = st.text_input("Lags (comma-separated)", value="1,2,3,7,14")
try:
    lags = [int(x.strip()) for x in lags_input.split(",") if x.strip()]
except Exception:
    lags = [1,2,3,7,14]

lagged = make_lag_features(series, lags=lags).dropna()
if len(lagged) < 30:
    st.warning("After creating lags there may be insufficient rows to train robust models; consider using fewer lags or more historical data.")

# time-based split
test_frac = st.slider("Test fraction (time-based)", 0.05, 0.4, 0.12)
cut = int(len(lagged) * (1 - test_frac))
train = lagged.iloc[:cut]
test = lagged.iloc[cut:]

X_train = train.drop(columns=["y"])
y_train = train["y"]
X_test = test.drop(columns=["y"])
y_test = test["y"]

model_choice = st.selectbox("Forecasting model", ["RandomForest", "Persistence"], index=0)
if model_choice == "RandomForest":
    n_est = st.slider("n_estimators (RF)", 10, 300, 100)
    model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        preds = np.array([np.nan]*len(X_test))
else:
    preds = X_test[f"lag_1"].values if f"lag_1" in X_test.columns else X_test.iloc[:,0].values

# evaluate
if len(y_test) > 0 and len(preds) == len(y_test):
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    st.write(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
else:
    st.info("Insufficient test predictions for evaluation.")

# plot predicted vs actual (last N points)
nplot = int(st.slider("Days to plot from test set", 20, min(500, max(20, len(y_test))), 60))
if len(y_test) > 0:
    nplot = min(nplot, len(y_test))
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(y_test.index[-nplot:], y_test.values[-nplot:], label="Actual")
    ax3.plot(y_test.index[-nplot:], preds[-nplot:], label="Predicted")
    ax3.legend()
    ax3.set_ylabel("Index Value")
    st.pyplot(fig3)

st.session_state["aqi_preds"] = preds if 'preds' in locals() else None

# -------------------------
# Task 3 — Clustering (Index Value)
# -------------------------
st.header("Task 3 — Clustering Pollution Patterns (low/med/high)")
daily_vals = series.dropna().to_frame(name="y")
if len(daily_vals) < 10:
    st.warning("Not enough daily values to cluster.")
else:
    scaler = StandardScaler()
    vals_scaled = scaler.fit_transform(daily_vals[["y"]])
    k = st.slider("K (clusters)", 2, 5, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(vals_scaled)
    daily_vals["cluster"] = km.labels_
    # order clusters by mean so we can label low/med/high
    cluster_means = daily_vals.groupby("cluster")["y"].mean().sort_values()
    ordered_clusters = cluster_means.index.tolist()
    label_names = ["Low", "Medium", "High"][:len(ordered_clusters)]
    cluster_label_map = {c: label_names[i] for i, c in enumerate(ordered_clusters)}
    daily_vals["cluster_label"] = daily_vals["cluster"].map(cluster_label_map)

    st.write("Cluster counts & means:")
    st.dataframe(daily_vals.groupby(["cluster_label"])["y"].agg(["count","mean","min","max"]).reset_index())

    # temporal scatter view colored by cluster
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    cmap = {"Low":"green","Medium":"orange","High":"red"}
    colors = daily_vals["cluster_label"].map(cmap).fillna("gray")
    ax4.scatter(daily_vals.index, daily_vals["y"], c=colors, s=8)
    ax4.set_ylabel("Index Value")
    st.pyplot(fig4)

# -------------------------
# Task 4 — Seasonal insights
# -------------------------
st.header("Task 4 — AI Seasonal Pollution Pattern Detector")

monthly = series.resample("M").mean()
st.subheader("Monthly average Index Value (time-series)")
st.line_chart(monthly)

# month-of-year averages across years
monthly_by_month = series.groupby(series.index.month).mean().rename_axis("month").reset_index()
monthly_by_month["month_name"] = monthly_by_month["month"].apply(lambda m: datetime(2000, m, 1).strftime("%b"))
monthly_by_month = monthly_by_month.rename(columns={0:"index_mean"}).set_index("month_name")
monthly_by_month["index_mean"] = monthly.groupby(monthly.index.month).mean().values

# categorize months using quantiles
q1 = monthly_by_month["index_mean"].quantile(0.33)
q2 = monthly_by_month["index_mean"].quantile(0.66)
def categorize(val):
    if val <= q1:
        return "Clean"
    elif val <= q2:
        return "Moderate"
    else:
        return "High"
monthly_by_month["category"] = monthly_by_month["index_mean"].apply(categorize)

st.subheader("Monthly averages and categories")
st.dataframe(monthly_by_month[["index_mean","category"]])

high_months = monthly_by_month[monthly_by_month["category"] == "High"].index.tolist()
if high_months:
    insight = f"Consistently high pollution months: {', '.join(high_months)}. Consider seasonal interventions in those months."
else:
    insight = "No months consistently fall into the 'High' pollution category."

st.subheader("Automated seasonal insight")
st.write(insight)

st.markdown("---")
st.write("Notes: this starter app uses simple models and heuristics. For production use: careful preprocessing, missing-value imputation, cross-validated time-series models (SARIMA/LSTM), and domain expert validation are recommended.")
