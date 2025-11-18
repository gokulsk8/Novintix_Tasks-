# app.py
# Robust Streamlit app for Household Power Consumption (auto-detect CSV/; and ZIP)
# Save as app.py and run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from io import BytesIO, StringIO
import zipfile
import csv
import warnings
from scipy import stats

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Household Electric Power — Full Analysis App", layout="wide")

st.title("⚡ Household Electric Power — Full Analysis App")
st.markdown(
    """
This app supports semicolon (`;`) or comma (`,`) separated CSVs and ZIP files containing CSVs.
Expected columns: `Date`, `Time`, `Global_active_power`, `Global_reactive_power`, `Voltage`,
`Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.
"""
)

# ---------- File input options ----------
use_default = st.sidebar.checkbox("Load default file from /mnt/data/household_power.csv (if present)", value=False)
uploaded = st.sidebar.file_uploader("Upload CSV / TXT / ZIP (semicolon-separated commonly)", type=["csv","txt","zip"])
st.sidebar.markdown("Or provide a local path (for advanced users):")
local_path = st.sidebar.text_input("Local path", value="")

# ---------- Helper functions ----------
@st.cache_data
def detect_sep_from_bytes(b: bytes, sample_lines: int = 5):
    # examine first lines to detect delimiter ',' or ';' (fallback to ',')
    try:
        text = b.decode("utf-8", errors="replace")
    except Exception:
        text = b.decode("latin1", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()][:sample_lines]
    if not lines:
        return ","
    header = lines[0]
    # Count delimiters
    comma_count = header.count(",")
    semi_count = header.count(";")
    if semi_count > comma_count:
        return ";"
    return ","

def read_csv_from_buffer(buf, sep=None):
    # buf: bytes or file-like; sep: optional
    if isinstance(buf, bytes):
        sep_to_use = sep or detect_sep_from_bytes(buf)
        return pd.read_csv(StringIO(buf.decode("utf-8", errors="replace")), sep=sep_to_use)
    else:
        # file-like object (StreamlitUploadedFile)
        try:
            raw = buf.getvalue()
            sep_to_use = sep or detect_sep_from_bytes(raw[:8192])
            buf.seek(0)
            return pd.read_csv(buf, sep=sep_to_use)
        except Exception as e:
            # fallback: try reading with both separators
            buf.seek(0)
            for s in [";", ","]:
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, sep=s)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    pass
            raise e

@st.cache_data
def load_dataframe(uploaded_file, local_path, use_default_flag):
    # Priority: uploaded_file -> default /mnt/data path if requested -> local_path
    # Returns pd.DataFrame or empty DataFrame
    # Handle ZIP containing CSV
    # Note: caching is helpful to avoid reloading heavy CSV repeatedly
    # 1) default path
    if use_default_flag:
        try:
            df = pd.read_csv("/mnt/data/household_power.csv", sep=";")
            return df
        except Exception:
            # try comma
            try:
                df = pd.read_csv("/mnt/data/household_power.csv", sep=",")
                return df
            except Exception:
                pass

    # 2) uploaded_file
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".zip"):
                # Streamlit uploaded file behaves like a file-like binary buffer
                uploaded_file.seek(0)
                b = uploaded_file.read()
                z = zipfile.ZipFile(BytesIO(b))
                # find first csv-like file
                csv_candidates = [f for f in z.namelist() if f.lower().endswith((".csv", ".txt"))]
                if not csv_candidates:
                    raise ValueError("ZIP archive contains no CSV/TXT file.")
                # read first candidate
                with z.open(csv_candidates[0]) as f:
                    raw = f.read()
                    return read_csv_from_buffer(raw)
            else:
                # CSV/TXT directly
                uploaded_file.seek(0)
                return read_csv_from_buffer(uploaded_file)
        except zipfile.BadZipFile:
            st.error("Uploaded file is not a valid ZIP archive.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return pd.DataFrame()

    # 3) local path provided
    if local_path:
        try:
            # try semicolon then comma
            try:
                df = pd.read_csv(local_path, sep=";")
                return df
            except Exception:
                df = pd.read_csv(local_path, sep=",")
                return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


# ---------- Load dataframe ----------
with st.spinner("Loading data..."):
    df = load_dataframe(uploaded, local_path, use_default)

if df.empty:
    st.error("No data loaded yet. Upload a CSV/TXT/ZIP (semicolon-separated commonly), or enable default/local path.")
    st.stop()

st.subheader("Raw data preview (first 5 rows)")
st.dataframe(df.head())

# ---------- Normalize column names if combined header discovered ----------
# Some files include a single column like "Date;Time;..." because wrong sep was used earlier.
if df.shape[1] == 1:
    # try splitting the single column by semicolon or comma
    col0 = df.columns[0]
    sample = "\n".join(df[col0].astype(str).head(5).tolist())
    # decide sep
    sep_guess = ";" if ";" in sample else ","
    try:
        new_df = df[col0].str.split(sep_guess, expand=True)
        # adopt header from first row if it looks like header
        header_row = new_df.iloc[0].tolist()
        # check basic header names
        if any("Date" in str(h) for h in header_row) or any("Global_active_power" in str(h) for h in header_row):
            new_df.columns = header_row
            new_df = new_df.drop(index=0).reset_index(drop=True)
        df = new_df
        st.success(f"Auto-split single-column input using separator '{sep_guess}'.")
    except Exception:
        pass

# Standardize columns: strip whitespace
df.columns = [str(c).strip() for c in df.columns]

# ---------- Expect Date and Time columns ----------
if "Date" not in df.columns or "Time" not in df.columns:
    # try alternative column names or combined column "DateTime" or "Date;Time"
    combined_candidates = [c for c in df.columns if "date" in c.lower() and "time" in c.lower()]
    if combined_candidates:
        # split that column into Date and Time
        col = combined_candidates[0]
        temp = df[col].astype(str).str.split(r'\s+', n=1, expand=True)
        if temp.shape[1] == 2:
            df["Date"] = temp[0]
            df["Time"] = temp[1]
    else:
        # also support header "Date;Time;..." case missed earlier: check first column text for ; separator
        col0 = df.columns[0]
        if df.shape[1] == 1:
            # attempt one more split pass
            first_vals = df.iloc[:,0].astype(str)
            if first_vals.str.contains(";").any():
                split_df = first_vals.str.split(";", expand=True)
                # we don't know headers — try to guess
                # attempt assign columns names based on expected order
                expected = ["Date","Time","Global_active_power","Global_reactive_power","Voltage",
                            "Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]
                ncols = split_df.shape[1]
                names = expected[:ncols] + [f"col_{i}" for i in range(ncols - len(expected))]
                split_df.columns = names[:ncols]
                df = split_df
        # final check
if "Date" not in df.columns or "Time" not in df.columns:
    st.error("Dataset must contain 'Date' and 'Time' columns after parsing. Found columns: " + ", ".join(df.columns[:15]))
    st.stop()

# ---------- Convert numeric columns ----------
expected_num = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]
for c in expected_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".").replace("?", np.nan), errors="coerce")

# ---------- Combine Date + Time into timestamp ----------
try:
    # many public datasets use dd/mm/yyyy
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                     dayfirst=True, errors="coerce")
except Exception:
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                     errors="coerce")

df = df.drop(columns=[c for c in ["Date","Time"] if c in df.columns], errors="ignore")
df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

st.write("Data range:", df.index.min(), "to", df.index.max())
st.write("Total rows (after parsing):", len(df))

# ---------- Resample options ----------
resample_freq = st.selectbox("Resample frequency for analysis", options=["T","30T","H","D"],
                            format_func=lambda x: {"T":"Minute (T)","30T":"30 minutes (30T)","H":"Hourly (H)","D":"Daily (D)"}[x],
                            index=2)
series = None
if "Global_active_power" in df.columns:
    series = df["Global_active_power"].resample(resample_freq).mean()
else:
    st.error("Column 'Global_active_power' not found in dataset.")
    st.stop()

# ---------- Task 1: EDA ----------
st.header("Task 1 — EDA")
st.subheader("Time-series trend of Global_active_power")
fig, ax = plt.subplots(figsize=(12,4))
series.plot(ax=ax)
ax.set_ylabel("Global_active_power (kW)")
st.pyplot(fig)

st.write("Statistics:")
st.write(series.describe())

st.subheader("Missing / Abnormal readings")
missing_count = int(series.isna().sum())
st.write(f"Missing readings after resample: {missing_count}")

# Z-score abnormal detection
z_thresh = st.slider("Z-score threshold for abnormal values", min_value=2.0, max_value=6.0, value=3.5)
valid = series.dropna()
if len(valid) > 1:
    zvals = np.abs(stats.zscore(valid.values))
    abnormal_idx = valid.index[zvals > z_thresh]
    st.write(f"Abnormal readings flagged: {len(abnormal_idx)}")
    if len(abnormal_idx) > 0:
        st.dataframe(valid.loc[abnormal_idx].head(30))
else:
    st.info("Not enough non-null values to compute z-score abnormalities.")

# Hourly and daily patterns
st.subheader("Hourly & Daily patterns")
if len(series.dropna()) > 0:
    hourly_pattern = series.groupby(series.index.hour).mean()
    fig2, ax2 = plt.subplots(figsize=(10,3))
    hourly_pattern.plot.bar(ax=ax2)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Avg Global_active_power")
    st.pyplot(fig2)

    dow_pattern = series.groupby(series.index.dayofweek).mean()
    fig3, ax3 = plt.subplots(figsize=(8,3))
    dow_pattern.plot.bar(ax=ax3)
    ax3.set_xlabel("Day of week (0=Mon)")
    ax3.set_ylabel("Avg Global_active_power")
    st.pyplot(fig3)

# ---------- Task 2: Forecasting (next-step) ----------
st.header("Task 2 — Forecasting next-step Global_active_power")
st.markdown("Create lag features and train a model to predict the next resampled step.")

def make_lag_features(s, lags=[1,2,3,24,48,168]):
    df_l = pd.DataFrame({"y": s})
    for lag in lags:
        df_l[f"lag_{lag}"] = df_l["y"].shift(lag)
    df_l["rolling_mean_24"] = df_l["y"].rolling(window=24, min_periods=1).mean().shift(1)
    df_l["rolling_std_24"] = df_l["y"].rolling(window=24, min_periods=1).std().shift(1)
    df_l["hour"] = df_l.index.hour
    df_l["dayofweek"] = df_l.index.dayofweek
    return df_l

lags_input = st.text_input("Lags (comma separated)", "1,2,3,24")
try:
    lags_list = [int(x.strip()) for x in lags_input.split(",") if x.strip()!='']
except Exception:
    lags_list = [1,2,3,24]

feat_df = make_lag_features(series, lags=lags_list)
st.write("Feature sample (first valid rows):")
st.dataframe(feat_df.dropna().head())

feat_df = feat_df.dropna()
if len(feat_df) < 20:
    st.warning("Not enough rows after lagging to train a model. Try shorter lag list or different resample frequency.")

test_size = st.slider("Test set proportion (time-based)", 0.05, 0.4, 0.12)
cut = int(len(feat_df) * (1 - test_size))
train = feat_df.iloc[:cut]
test = feat_df.iloc[cut:]

X_train = train.drop(columns=["y"])
y_train = train["y"]
X_test = test.drop(columns=["y"])
y_test = test["y"]

st.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

model_choice = st.selectbox("Model choice", ["RandomForest", "Persistence"], index=0)
if model_choice == "RandomForest":
    n_est = st.slider("RF n_estimators", 10, 300, 100)
    try:
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        preds = np.array([np.nan]*len(X_test))
else:
    preds = X_test["lag_1"].values

# Evaluate forecasting
if len(y_test) > 0 and len(preds) == len(y_test):
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mape = np.nanmean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100
    st.write(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}  MAPE: {np.nanmean(mape):.2f}%")
else:
    st.info("Forecast evaluation skipped due to insufficient test data or prediction length mismatch.")

# Plot predicted vs actual for last N points
nplot = int(st.slider("Points to plot (from test set)", 50, min(500, max(50, len(y_test))), 200))
if len(y_test) >= 1:
    nplot = min(nplot, len(y_test))
    fig4, ax4 = plt.subplots(figsize=(12,4))
    ax4.plot(y_test.index[-nplot:], y_test.values[-nplot:], label="Actual")
    ax4.plot(y_test.index[-nplot:], preds[-nplot:], label="Predicted")
    ax4.legend()
    ax4.set_ylabel("Global_active_power")
    st.pyplot(fig4)

# store preds for Task 4
st.session_state["last_predictions"] = preds if 'preds' in locals() else None

# ---------- Task 3: Unsupervised (anomaly detection & clustering) ----------
st.header("Task 3 — Anomaly detection & Clustering")
st.subheader("Anomaly detection on resampled series (IsolationForest)")
contamination = st.slider("Anomaly contamination (fraction)", 0.001, 0.1, 0.01)
valid_series = series.dropna().to_frame(name="y")
if len(valid_series) > 10:
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(valid_series[["y"]])
    iso_pred = iso.predict(valid_series[["y"]])
    valid_series["anomaly"] = iso_pred == -1
    st.write("Anomaly counts:")
    st.write(valid_series["anomaly"].value_counts())
    if valid_series["anomaly"].sum() > 0:
        st.dataframe(valid_series[valid_series["anomaly"]].head(30))
else:
    st.info("Not enough valid values for anomaly detection.")

# Clustering daily 24-hour profiles (prefer hourly resample)
st.subheader("Clustering daily consumption profiles (KMeans)")
if resample_freq != "H":
    st.info("For 24-hour profiles, choose hourly (H) resample for better clusters.")

hourly_full = series.resample("H").mean()
day_groups = hourly_full.groupby(hourly_full.index.date)
profiles = []
dates = []
for d, g in day_groups:
    vals = g.values
    if len(vals) >= 20:  # prefer near-full days
        arr = vals[:24] if len(vals) >= 24 else np.pad(vals, (0, 24-len(vals)), 'constant', constant_values=np.nan)
        profiles.append(arr)
        dates.append(pd.to_datetime(d))
profiles = np.array(profiles)

if profiles.size == 0:
    st.warning("Not enough hourly data to compute daily 24-hour profiles.")
else:
    # fill NaNs per-day with that day's mean
    for i in range(profiles.shape[0]):
        row = profiles[i]
        nan_mask = np.isnan(row)
        if nan_mask.any():
            fill = np.nanmean(row)
            if np.isnan(fill):
                fill = 0.0
            row[nan_mask] = fill
            profiles[i] = row
    scaler = StandardScaler()
    profiles_s = scaler.fit_transform(profiles)
    n_clusters = st.slider("KMeans clusters (K)", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(profiles_s)
    labels = kmeans.labels_
    st.write("Cluster counts:")
    st.write(pd.Series(labels).value_counts())

    # PCA visualization
    pca = PCA(n_components=2)
    proj = pca.fit_transform(profiles_s)
    fig5, ax5 = plt.subplots(figsize=(8,5))
    for k in range(n_clusters):
        ax5.scatter(proj[labels==k,0], proj[labels==k,1], label=f"Cluster {k}", alpha=0.7)
    ax5.legend()
    ax5.set_title("PCA projection of daily profiles")
    st.pyplot(fig5)

    # Cluster mean profiles
    st.write("Cluster mean 24-hour profiles:")
    mean_profiles = {}
    for k in range(n_clusters):
        mean_profiles[k] = profiles[labels==k].mean(axis=0)
        figp, axp = plt.subplots(figsize=(10,2))
        axp.plot(mean_profiles[k])
        axp.set_title(f"Cluster {k} mean profile")
        axp.set_xlabel("Hour (0-23)")
        st.pyplot(figp)

    # cluster description
    cluster_desc = []
    for k in range(n_clusters):
        avg = float(np.nanmean(mean_profiles[k]))
        peak = int(np.nanargmax(mean_profiles[k]))
        cluster_desc.append({"cluster":k, "mean_power":round(avg,3), "peak_hour":peak})
    st.write(pd.DataFrame(cluster_desc))

# ---------- Task 4: Simple rule-based AI ----------
st.header("Task 4 — Simple Consumption Category Generator (Rule-Based)")
last_preds = st.session_state.get("last_predictions", None)
if last_preds is not None and len(last_preds) > 0:
    last_pred_value = float(last_preds[-1])
    st.write("Last predicted Global_active_power:", round(last_pred_value, 3))
    q_low = series.quantile(0.33)
    q_high = series.quantile(0.66)
    if last_pred_value <= q_low:
        category = "Low Usage"
        suggestion = "Low consumption. Consider scheduled maintenance check."
    elif last_pred_value <= q_high:
        category = "Medium Usage"
        suggestion = "Normal consumption. Shift heavy tasks to off-peak if possible."
    else:
        category = "High Usage"
        suggestion = "High consumption. Inspect for faulty appliances and consider efficiency upgrades."

    st.write("Assigned category:", category)
    st.write("Suggestion:", suggestion)

    st.subheader("Example output")
    st.json({"Predicted_Global_active_power": round(last_pred_value,3),
             "Category": category,
             "Suggestion": suggestion})
else:
    st.info("Run the forecasting section to produce predictions (Task 2).")

st.markdown("---")
st.write("Notes: This is a starter educational app. For production use: robust cleaning, model selection (ARIMA/LSTM), hyperparameter tuning, and ethical checks are required.")
