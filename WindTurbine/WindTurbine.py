# app_wind_turbine.py
# Streamlit app for Wind Turbine SCADA dataset analysis
# Save as app_wind_turbine.py and run: streamlit run app_wind_turbine.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import zipfile
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Wind Turbine SCADA Analyzer", layout="wide")
st.title("Wind Turbine SCADA Analyzer — EDA, Forecasting, Anomaly Detection, Score")

st.markdown("""
**Expected columns** (choose the file for one turbine):
- `Date/Time`
- `LV ActivePower (kW)`
- `Wind Speed (m/s)`
- `Theoretical_Power_Curve (kWh)`
- `Wind Direction (°)`

This app:
- Task 1: EDA & power-curve scatter
- Task 2: Forecasting (next-step) for 4 variables
- Task 3: Anomaly detection (underperformance vs theoretical)
- Task 4: Performance scoring and recommendation
""")

# ---------------- File upload
uploaded = st.file_uploader("Upload CSV/TXT/ZIP (many public files use ';' delimiter)", type=["csv","txt","zip"])
use_sample = st.sidebar.checkbox("Try default file at /mnt/data/wind_turbine.csv (if present)", value=False)
local_path = st.sidebar.text_input("Or supply local path (optional)", value="")

def detect_sep(text):
    if not text:
        return ","
    header = text.splitlines()[0]
    return ";" if header.count(";") > header.count(",") else ","

@st.cache_data
def read_buffer(buf):
    if hasattr(buf, "getvalue"):
        raw = buf.getvalue()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        sep = detect_sep(text[:2000])
        return pd.read_csv(StringIO(text), sep=sep)
    else:
        # path
        for sep in [";", ","]:
            try:
                df = pd.read_csv(buf, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        return pd.DataFrame()

@st.cache_data
def load_df(uploaded_file, local_path, use_sample_flag):
    if use_sample_flag:
        # try default sample path
        for sep in [";", ","]:
            try:
                df = pd.read_csv("/mnt/data/wind_turbine.csv", sep=sep)
                if not df.empty:
                    return df
            except Exception:
                pass
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".zip"):
                b = uploaded_file.getvalue()
                z = zipfile.ZipFile(BytesIO(b))
                csvs = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
                if not csvs:
                    return pd.DataFrame()
                with z.open(csvs[0]) as f:
                    raw = f.read().decode("utf-8", errors="replace")
                    sep = detect_sep(raw[:2000])
                    return pd.read_csv(StringIO(raw), sep=sep)
            else:
                return read_buffer(uploaded_file)
        except Exception:
            return pd.DataFrame()
    if local_path:
        return read_buffer(local_path)
    return pd.DataFrame()

with st.spinner("Loading data..."):
    df = load_df(uploaded, local_path, use_sample)

if df.empty:
    st.info("Please upload the dataset or enable default/local path.")
    st.stop()

# Preview
st.subheader("Raw data sample")
st.dataframe(df.head())

# Normalize columns: common names used in dataset (try to map)
col_map = {}
cols = [c.strip() for c in df.columns]
for c in cols:
    lc = c.lower()
    if "date" in lc and "time" in lc:
        col_map["dt"] = c
    if "active" in lc and "power" in lc:
        col_map["power"] = c
    if "wind speed" in lc or ("wind" in lc and "speed" in lc):
        col_map["wind_speed"] = c
    if "theoretical" in lc and "power" in lc:
        col_map["theoretical"] = c
    if "wind direction" in lc or ("wind" in lc and "direction" in lc):
        col_map["wind_dir"] = c

st.write("Detected columns (auto):")
st.write(col_map)

# Let user remap if detection incorrect
st.markdown("**Map columns manually if detection is incorrect**")
for key, label in [("dt","Date/Time"),("power","LV ActivePower (kW)"),("wind_speed","Wind Speed (m/s)"),
                   ("theoretical","Theoretical_Power_Curve (kWh)"),("wind_dir","Wind Direction (°)")]:
    options = [""] + cols
    default = col_map.get(key,"")
    sel = st.selectbox(f"{label} column", options=options, index=(options.index(default) if default in options else 0))
    if sel:
        col_map[key] = sel

required_keys = ["dt","power","wind_speed","theoretical","wind_dir"]
if not all(k in col_map and col_map[k] for k in required_keys):
    st.error(f"Please map all required columns: {required_keys}")
    st.stop()

# Convert types
df[col_map["dt"]] = pd.to_datetime(df[col_map["dt"]].astype(str), dayfirst=True, errors="coerce")
for key in ["power","wind_speed","theoretical","wind_dir"]:
    df[col_map[key]] = pd.to_numeric(df[col_map[key]].astype(str).str.replace(",", "."), errors="coerce")

df = df.dropna(subset=[col_map["dt"]]).set_index(col_map["dt"]).sort_index()

st.write("Data range:", df.index.min(), "to", df.index.max())
st.write("Rows after parsing:", len(df))

# ---------------- Task 1: EDA ----------------
st.header("Task 1 — EDA")

st.subheader("Time-series plots")
params = [("LV ActivePower (kW)", col_map["power"]),
          ("Wind Speed (m/s)", col_map["wind_speed"]),
          ("Theoretical Power Curve (kWh)", col_map["theoretical"]),
          ("Wind Direction (°)", col_map["wind_dir"])]

fig, axs = plt.subplots(4,1, figsize=(12,10), sharex=True)
for ax, (title, col) in zip(axs, params):
    df[col].plot(ax=ax)
    ax.set_ylabel(title)
    ax.grid(True)
st.pyplot(fig)

# Missing/abnormal detection summary
st.subheader("Missing / Abnormal readings summary")
for title, col in params:
    s = df[col]
    miss = int(s.isna().sum())
    st.write(f"{title}: missing = {miss}, mean = {s.mean():.3f}, std = {s.std():.3f}")

# Scatter: Wind Speed vs LV ActivePower (power curve)
st.subheader("Wind Speed vs LV ActivePower (power curve)")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(df[col_map["wind_speed"]], df[col_map["power"]], s=6, alpha=0.5)
ax2.set_xlabel("Wind Speed (m/s)")
ax2.set_ylabel("LV ActivePower (kW)")
ax2.set_title("Power Curve Scatter")
st.pyplot(fig2)

# Basic z-score abnormal detection per column (user chooses threshold)
z_thresh = st.slider("Z-score threshold for detecting abnormal readings (applies to each series)", 2.0, 6.0, 3.5)
st.write("Example abnormal samples (by z-score):")
for title, col in params:
    valid = df[col].dropna()
    if len(valid) > 1:
        z = np.abs(stats.zscore(valid.values))
        abnormal_idx = valid.index[z > z_thresh]
        st.write(f"{title} - abnormal count: {len(abnormal_idx)}")
        if len(abnormal_idx) > 0:
            st.dataframe(valid.loc[abnormal_idx].head(10))

# ---------------- Task 2: Forecasting ----------------
st.header("Task 2 — Forecasting (next-step) for all four variables)")

# Resample frequency (let user choose, commonly dataset is 10-min or 10s; default to hourly aggregation if dense)
resample_freq = st.selectbox("Resample frequency for modeling", options=["10T","30T","H","D"], index=2,
                             format_func=lambda x: {"10T":"10 minutes","30T":"30 minutes","H":"Hourly","D":"Daily"}[x])
df_res = df[[col_map["power"], col_map["wind_speed"], col_map["theoretical"], col_map["wind_dir"]]].resample(resample_freq).mean()

st.write("Resampled data sample:")
st.dataframe(df_res.head())

# Create lagged features for forecasting next step (user input lags)
default_lags = "1,2,3,24"
lags_input = st.text_input("Lags (steps) to create for windowed features (comma-separated)", default_lags)
try:
    lags = [int(x.strip()) for x in lags_input.split(",") if x.strip()]
except:
    lags = [1,2,3,24]

def make_features_for_targets(df_series, lags):
    # df_series is dataframe with target columns; returns a dict of feature/target pairs for each column
    results = {}
    for col in df_series.columns:
        s = df_series[col]
        tmp = pd.DataFrame({"y": s})
        for lag in lags:
            tmp[f"lag_{lag}"] = tmp["y"].shift(lag)
        tmp["hour"] = tmp.index.hour
        tmp["dayofweek"] = tmp.index.dayofweek
        results[col] = tmp
    return results

feat_dict = make_features_for_targets(df_res, lags)

# Train/test split proportion
test_prop = st.slider("Test set proportion (time-based)", 0.05, 0.4, 0.15)

models_choice = st.selectbox("Model for forecasting", ["RandomForest","Persistence"], index=0)

forecast_results = {}
for col, feat in feat_dict.items():
    st.subheader(f"Forecasting for {col}")
    feat_clean = feat.dropna()
    if len(feat_clean) < 10:
        st.warning(f"Not enough data for {col} after lagging. Skipping.")
        continue
    cutoff = int(len(feat_clean)*(1-test_prop))
    train = feat_clean.iloc[:cutoff]
    test = feat_clean.iloc[cutoff:]
    X_train = train.drop(columns=["y"])
    y_train = train["y"]
    X_test = test.drop(columns=["y"])
    y_test = test["y"]
    if models_choice == "RandomForest":
        n_est = st.slider(f"n_estimators for {col}", 20, 300, 100, key=f"ne_{col}")
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        # persistence baseline using lag_1 if available otherwise last column
        if "lag_1" in X_test.columns:
            preds = X_test["lag_1"].values
        else:
            preds = X_test.iloc[:,0].values

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds) if len(y_test)>0 else np.nan
    st.write(f"Test samples: {len(y_test)} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    # Plot predicted vs actual (last N points)
    nplot = min(len(y_test), st.slider(f"Points to plot for {col}", 20, 1500, 200, key=f"n_{col}"))
    if nplot > 0 and len(y_test) > 0:
        figp, axp = plt.subplots(figsize=(10,3))
        axp.plot(y_test.index[-nplot:], y_test.values[-nplot:], label="Actual")
        axp.plot(y_test.index[-nplot:], preds[-nplot:], label="Predicted")
        axp.legend()
        axp.set_title(f"{col} — Predicted vs Actual (last {nplot})")
        st.pyplot(figp)

    forecast_results[col] = {"y_test": y_test, "preds": preds, "mae": mae, "rmse": rmse, "r2": r2}

# ---------------- Task 3: Anomaly detection (underperformance) ----------------
st.header("Task 3 — Anomaly Detection: Underperformance vs Theoretical Power Curve")

# Create performance gap = theoretical - actual (or actual/theoretical ratio)
tp_col = col_map["theoretical"]
actual_col = col_map["power"]
resampled = df_res.copy().dropna(subset=[actual_col, tp_col])
resampled["gap"] = resampled[tp_col] - resampled[actual_col]   # positive = theoretical > actual => underperformance
resampled["ratio"] = resampled[actual_col] / resampled[tp_col].replace(0, np.nan)

st.write("Underperformance summary:")
st.write(resampled[["gap","ratio"]].describe())

# Flag anomalies where gap is significantly large relative to distribution
gap_z = np.abs(stats.zscore(resampled["gap"].fillna(0)))
z_cut = st.slider("Z-score threshold for underperformance anomaly", 2.0, 6.0, 3.0)
resampled["underperf_anomaly"] = gap_z > z_cut
st.write("Underperformance anomalies count:", int(resampled["underperf_anomaly"].sum()))
if int(resampled["underperf_anomaly"].sum()) > 0:
    st.dataframe(resampled[resampled["underperf_anomaly"]].sort_values("gap", ascending=False).head(30))

# Option: IsolationForest on gap
if st.checkbox("Run IsolationForest for anomalies on gap (extra)", value=False):
    iso = IsolationForest(contamination=0.01, random_state=42)
    vals = resampled["gap"].fillna(0).values.reshape(-1,1)
    iso.fit(vals)
    preds_iso = iso.predict(vals)
    resampled["iso_anom"] = preds_iso == -1
    st.write("IsolationForest anomalies:", int(resampled["iso_anom"].sum()))
    if int(resampled["iso_anom"].sum())>0:
        st.dataframe(resampled[resampled["iso_anom"]].head(30))

# Visualize gap over time and mark anomalies
fig_gap, ax_gap = plt.subplots(figsize=(12,3))
ax_gap.plot(resampled.index, resampled["gap"], label="Theoretical - Actual")
ax_gap.scatter(resampled.index[resampled["underperf_anomaly"]], resampled["gap"][resampled["underperf_anomaly"]], color="red", label="Anomaly")
ax_gap.legend()
ax_gap.set_ylabel("Gap (kW)")
st.pyplot(fig_gap)

# ---------------- Task 4: AI Turbine Performance Score Generator ----------------
st.header("Task 4 — Turbine Performance Score & Recommendation")

# Performance ratio: actual / theoretical, clipped 0-1, then scale to 0-100
perf = resampled["ratio"].clip(lower=0, upper=2)  # allow >1 if actual > theoretical but cap
# normalize: assume realistic range 0-1.2 -> map to 0-100
min_r = perf.quantile(0.01)
max_r = perf.quantile(0.99)
# avoid divide by zero
if max_r - min_r <= 0:
    max_r = perf.max()
    min_r = perf.min()
score = ((perf - min_r) / (max_r - min_r)).clip(0,1) * 100
resampled["perf_score"] = score.round(1)

# Categorize
def categorize_score(x):
    if x >= 80:
        return "Good"
    if x >= 50:
        return "Moderate"
    return "Poor"

resampled["perf_category"] = resampled["perf_score"].apply(categorize_score)

st.write("Performance score distribution:")
st.write(resampled["perf_score"].describe())

# Show latest N samples with score and category
n_show = st.slider("Rows to show with score", 5, 100, 20)
st.dataframe(resampled[["ratio","perf_score","perf_category"]].sort_index(ascending=False).head(n_show))

# Summary counts
st.write("Categories counts:")
st.write(resampled["perf_category"].value_counts())

# Automated suggestion based on last average score
recent_score = resampled["perf_score"].tail(24).mean()  # last 24 resampled points
if np.isnan(recent_score):
    st.info("Not enough data to compute recent performance suggestion.")
else:
    recent_score = float(recent_score)
    if recent_score >= 80:
        suggestion = "Performance is Good. Keep regular maintenance and monitor for rare anomalies."
    elif recent_score >= 50:
        suggestion = "Performance Moderate. Investigate periods of underperformance, inspect blade cleanliness and pitch control."
    else:
        suggestion = "Performance Poor. Immediate inspection recommended: check mechanical faults, pitch/yaw control, gearbox, and sensors."

    st.subheader("Automated suggestion (based on recent average score)")
    st.write(f"Recent average score (last 24 points): {recent_score:.1f}/100")
    st.write(suggestion)

# Provide download of scored results
if st.button("Download scored results (CSV)"):
    out = resampled.reset_index().to_csv(index=False)
    st.download_button("Click to download CSV", data=out, file_name="wind_turbine_scored.csv", mime="text/csv")

st.markdown("---")
st.write("Notes: This is an educational starter app. For production: use domain-calibrated power curves, physically-grounded models, better gap normalization, sophisticated time-series models (LSTM/Seq2Seq, SARIMAX), and expert validation.")
