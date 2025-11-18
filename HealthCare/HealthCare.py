# streamlit_healthcare_app.py
# Streamlit application for Healthcare Dataset EDA, Supervised learning, Anomaly detection,
# and an LLM-style AI doctor recommendation generator.
# Run with: streamlit run streamlit_healthcare_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Healthcare ML & EDA App", layout="wide")

st.title("Healthcare Dataset — EDA, ML & Anomaly Detection")
st.markdown("Upload a CSV file (or provide local path). If you don't upload, the app will look for 'healthcare_dataset.csv' in the working directory.")

uploaded_file = st.file_uploader("Upload healthcare CSV file", type=["csv"])
local_path = st.text_input("Or enter local CSV path (optional)", value="healthcare_dataset.csv")

@st.cache_data
def load_data(uploaded, path):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()
    return df

df = load_data(uploaded_file, local_path)

if df.empty:
    st.warning("No data loaded yet — upload a CSV or provide a valid path.")
    st.stop()

st.subheader("Raw Data Sample")
st.dataframe(df.head())

# --- Task 1: EDA ---
st.header("Task 1 — Exploratory Data Analysis (EDA)")

# Ensure columns exist
required_cols = ["Name","Age","Gender","Blood Type","Medical Condition","Date of Admission",
                 "Doctor","Hospital","Insurance Provider","Billing Amount","Room Number",
                 "Admission Type","Discharge Date","Medication","Test Results"]

present = [c for c in required_cols if c in df.columns]
missing = [c for c in required_cols if c not in df.columns]

st.write(f"Columns present: {len(present)} / {len(required_cols)}")
if missing:
    st.info(f"Missing columns in uploaded file: {missing}")

# Convert numeric-like columns
for col in ["Age","Billing Amount","Room Number"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Distribution plots
st.subheader("Distributions")
col1, col2, col3 = st.columns(3)

with col1:
    if "Age" in df.columns:
        st.write("Age — histogram and summary")
        fig, ax = plt.subplots()
        df['Age'].dropna().plot.hist(bins=20, ax=ax)
        ax.set_xlabel('Age')
        st.pyplot(fig)
        st.write(df['Age'].describe())
    else:
        st.write("Age column not found")

with col2:
    if "Billing Amount" in df.columns:
        st.write("Billing Amount — histogram and summary")
        fig, ax = plt.subplots()
        df['Billing Amount'].dropna().plot.hist(bins=30, ax=ax)
        ax.set_xlabel('Billing Amount')
        st.pyplot(fig)
        st.write(df['Billing Amount'].describe())
    else:
        st.write("Billing Amount column not found")

with col3:
    if "Room Number" in df.columns:
        st.write("Room Number — histogram and summary")
        fig, ax = plt.subplots()
        df['Room Number'].dropna().astype('float').plot.hist(bins=20, ax=ax)
        ax.set_xlabel('Room Number')
        st.pyplot(fig)
        st.write(df['Room Number'].describe())
    else:
        st.write("Room Number column not found")

# Frequency of categorical fields
st.subheader("Frequencies — Medical Condition, Admission Type, Medication")
cat_cols = [c for c in ["Medical Condition","Admission Type","Medication"] if c in df.columns]

for c in cat_cols:
    st.write(f"Top frequencies for {c}")
    counts = df[c].fillna('Unknown').value_counts().head(20)
    fig, ax = plt.subplots()
    counts.plot.bar(ax=ax)
    ax.set_ylabel('Count')
    st.pyplot(fig)

# --- Task 2: Supervised Learning ---
st.header("Task 2 — Supervised Learning: Predict Test Results")

if "Test Results" not in df.columns:
    st.error("No 'Test Results' column found — cannot run supervised learning.")
else:
    # Prepare dataset
    target = 'Test Results'
    st.write("Preparing features and target...")

    # Basic feature selection: drop identifiers and text-heavy fields that are unlikely to help
    drop_cols = ['Name','Doctor','Hospital','Discharge Date','Date of Admission']
    features = [c for c in df.columns if c != target and c not in drop_cols]
    X = df[features].copy()
    y = df[target].copy()

    # Handle Date of Admission into derived features if present in original df
    if 'Date of Admission' in df.columns:
        try:
            dt = pd.to_datetime(df['Date of Admission'], errors='coerce')
            X['Admission_month'] = dt.dt.month
            X['Admission_day'] = dt.dt.day
        except Exception:
            pass

    # Identify numeric and categorical features robustly
    numeric_feats = X.select_dtypes(include=['int64','float64','number']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    st.write(f"Numeric features: {numeric_feats}")
    st.write(f"Categorical features: {categorical_feats}")

    # Build transformers conditionally (avoid empty-list errors)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # NOTE: scikit-learn >=1.4 removed `sparse=`; use sparse_output instead
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if len(numeric_feats) > 0:
        transformers.append(('num', numeric_transformer, numeric_feats))
    if len(categorical_feats) > 0:
        transformers.append(('cat', categorical_transformer, categorical_feats))

    if len(transformers) == 0:
        # If no features at all (edge case), stop gracefully
        st.error("No usable features found after preprocessing. Aborting supervised learning.")
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # Decide whether target is numeric -> regression else classification
        is_target_numeric = pd.api.types.is_numeric_dtype(y)

        # Encode classification labels if needed
        label_encoder = None
        if not is_target_numeric:
            y_encoded = y.fillna('missing').astype(str)
            label_encoder = LabelEncoder()
            try:
                y_enc = label_encoder.fit_transform(y_encoded)
            except Exception:
                # fallback: convert to str then encode
                y_enc = label_encoder.fit_transform(y_encoded.astype(str))
        else:
            y_enc = pd.to_numeric(y, errors='coerce')

        # Drop rows where target is missing
        mask = ~pd.isna(y_enc)
        X = X.loc[mask].reset_index(drop=True)
        y_enc = pd.Series(y_enc[mask]).reset_index(drop=True)

        # If after filtering no rows remain, stop
        if len(y_enc) < 2:
            st.error("Not enough samples with non-missing target to train a model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=(y_enc if not is_target_numeric else None))

            if is_target_numeric:
                st.write("Detected numeric target — using RandomForestRegressor")
                model = Pipeline(steps=[('pre', preprocessor), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mae = mean_absolute_error(y_test, preds)
                rmse = mean_squared_error(y_test, preds, squared=False)
                st.write(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                # Show predicted vs actual
                comparison = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': preds})
                st.subheader('Predicted vs Actual (sample)')
                st.write(comparison.head(20))

            else:
                st.write("Detected categorical target — using RandomForestClassifier")
                model = Pipeline(steps=[('pre', preprocessor), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                st.write(f"Accuracy on test set: {acc:.4f}")
                st.text("Classification report:")
                report = classification_report(y_test, preds, zero_division=0)
                st.text(report)

                # Confusion matrix
                cm = confusion_matrix(y_test, preds)
                st.write("Confusion matrix:")
                st.write(cm)

            # Save sample predictions to the dataframe for AI task
            sample_for_ai = None
            try:
                if len(X_test) > 0:
                    sample_for_ai = X_test.copy().reset_index(drop=True)
                    # Attach predicted values in human-readable form
                    if not is_target_numeric and label_encoder is not None:
                        # make sure preds are ints
                        try:
                            preds_int = np.array(preds, dtype=int)
                            decoded = label_encoder.inverse_transform(preds_int)
                        except Exception:
                            # fallback: safest string conversion
                            decoded = [str(p) for p in preds]
                        sample_for_ai['Predicted_Result'] = decoded
                    else:
                        # regression
                        sample_for_ai['Predicted_Result'] = preds
            except Exception:
                sample_for_ai = None

# --- Task 3: Unsupervised Anomaly Detection on Billing Amount ---
st.header("Task 3 — Anomaly Detection in Billing Amounts")

if "Billing Amount" not in df.columns:
    st.error("Billing Amount column not present — cannot run anomaly detection.")
else:
    ba = df[['Billing Amount']].copy()
    ba['Billing Amount'] = pd.to_numeric(ba['Billing Amount'], errors='coerce')
    ba = ba.dropna().reset_index(drop=True)

    if ba.empty:
        st.error("Billing Amount series is empty after conversion.")
    else:
        st.write("Basic statistics:")
        st.write(ba.describe())

        # Z-score method
        if len(ba['Billing Amount']) > 1:
            z_scores = np.abs(stats.zscore(ba['Billing Amount']))
        else:
            z_scores = np.zeros(len(ba['Billing Amount']))
        z_thresh = st.slider('Z-score threshold for anomaly', 2.0, 6.0, 3.0)
        ba['z_anomaly'] = z_scores > z_thresh

        # IsolationForest only if enough samples
        iso_anomaly = np.array([False] * len(ba))
        if len(ba) >= 5:
            iso = IsolationForest(contamination=0.01, random_state=42)
            try:
                iso_pred = iso.fit_predict(ba[['Billing Amount']])
                iso_anomaly = iso_pred == -1
            except Exception:
                iso_anomaly = np.array([False] * len(ba))
        ba['iso_anomaly'] = iso_anomaly

        # Combine
        ba['anomaly'] = ba['z_anomaly'] | ba['iso_anomaly']

        st.write("Anomaly counts:")
        st.write(ba['anomaly'].value_counts())

        st.subheader('Detected anomalies (sample)')
        st.write(ba[ba['anomaly']].head(50))

        st.markdown("**Short interpretation:** Anomalies flagged by Z-score and/or IsolationForest are billing values that are unusually large or small. They may indicate rare expensive treatments, data entry errors, or insurance adjustments. Flagged entries should be manually reviewed.")

# --- Task 4: AI Doctor Recommendation Generator ---
st.header("Task 4 — AI Doctor Recommendation Generator")

st.markdown("This generates a short doctor-style recommendation using a local template + the model's predicted test result and key attributes.")

def generate_recommendation(age, medical_condition, medication, predicted_result, billing_mean):
    rec = []
    rec.append(f"Patient age: {age} — Condition: {medical_condition}.")
    rec.append(f"Predicted test result: {predicted_result}.")
    rec.append("Recommendation:")
    rec.append(f"- Please consult a specialist in the area related to {medical_condition}.")
    try:
        pred_str = str(predicted_result).lower()
    except Exception:
        pred_str = ""
    if 'positive' in pred_str:
        rec.append("- The test indicates a positive finding; further confirmatory testing is recommended and consider starting treatment based on clinical assessment.")
    else:
        # If numeric and billing_mean available, compare numeric
        try:
            pr = float(predicted_result)
            if billing_mean is not None and pr > billing_mean:
                rec.append("- The numeric result is high compared to population mean — clinical correlation required.")
            else:
                rec.append("- The test result appears within expected range; continue monitoring and supportive care.")
        except Exception:
            rec.append("- The test result appears within expected range; continue monitoring and supportive care.")
    rec.append(f"- Current medication: {medication}. Review medication interactions and adjust as needed.")
    rec.append("- General advice: maintain hydration, balanced diet, follow up in 7 days or earlier if symptoms worsen.")
    return '\n'.join(rec)

st.subheader('Generate example recommendation')

billing_mean = None
if 'Billing Amount' in df.columns:
    try:
        billing_mean = float(df['Billing Amount'].dropna().mean())
    except Exception:
        billing_mean = None

if 'sample_for_ai' in locals() and sample_for_ai is not None and not sample_for_ai.empty:
    example_idx = st.number_input('Choose an index from sample predictions (0-based)', min_value=0, max_value=len(sample_for_ai)-1, value=0)
    row = sample_for_ai.reset_index().iloc[example_idx]
    age_val = row.get('Age', 'Unknown') if 'Age' in row.index else 'Unknown'
    cond_val = row.get('Medical Condition', 'Unknown') if 'Medical Condition' in row.index else 'Unknown'
    med_val = row.get('Medication', 'Unknown') if 'Medication' in row.index else 'Unknown'
    pred_val = row.get('Predicted_Result', 'Unknown')

    st.write('Sample patient attributes:')
    st.write({'Age': age_val, 'Medical Condition': cond_val, 'Medication': med_val, 'Predicted Result': pred_val})

    if st.button('Generate recommendation for chosen sample'):
        rec_text = generate_recommendation(age_val, cond_val, med_val, pred_val, billing_mean)
        st.text_area('AI Doctor Recommendation', value=rec_text, height=260)
else:
    st.info('No sample predictions available to generate AI recommendation. Run the supervised learning section successfully first.')

st.markdown('---')
st.write('Notes:')
st.write('- This app uses simple pipelines and is designed as an educational starter template. For production use: proper feature engineering, careful handling of dates and clinical text, stronger validation, and ethical review are required.')
st.write('- If your Test Results column is textual, the classifier will be used; if numeric, a regressor will be used.')
