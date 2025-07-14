from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import streamlit as st

# === PARAMETERS ===
DATA_FILE = "/home/duch/ai_hiep/data/PM25_RH_T_Wind_SW_2018_2022.csv"
MODEL_FILE = "cnn_lstm_pm25_met_72input.h5"
SCALER_FILE = "pm25_scaler.save"
COLUMNS_FILE = "pm25_columns.save"
N_INPUT = 72
N_OUTPUT = 6
N_SUBSEQ = 6
N_STEPS_PER_SUBSEQ = N_INPUT // N_SUBSEQ

# === LOAD DATA ===
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"], dayfirst=True)
df = df.rename(columns={
    'BRINGELLY': 'PM2.5_BRINGELLY',
    'CAMPBELLTOWN_WEST': 'PM2.5_CAMPBELLTOWN_WEST',
    'CAMDEN': 'PM2.5_CAMDEN',
    'LIVERPOOL': 'PM2.5_LIVERPOOL'
})
df = df.set_index("datetime").dropna()

# === LOAD SCALER AND MODEL ===
scaler = joblib.load(SCALER_FILE)
feature_columns = joblib.load(COLUMNS_FILE)
model = load_model(MODEL_FILE)

# === CREATE SEQUENCES FUNCTION ===
def create_sequences(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i + n_input])
        y.append(data[i + n_input:i + n_input + n_output, :4])  # Only PM2.5 outputs
    return np.array(X), np.array(y)

# === PREPARE INPUT ===
df_model = df[feature_columns + ["CTM"]].dropna()
scaled = scaler.transform(df_model[feature_columns])
X, y = create_sequences(scaled, N_INPUT, N_OUTPUT)

# === ALIGN CTM ===
ctm_series = df_model["CTM"].values[N_INPUT:N_INPUT + len(y)]
X = X.reshape((X.shape[0], N_SUBSEQ, N_STEPS_PER_SUBSEQ, len(feature_columns)))

# === SPLIT INTO TEST SET ===
split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]
ctm_test = ctm_series[split:]

# === TIME INDEX FOR TEST SET ===
time_index_test = df_model.index[N_INPUT + split:N_INPUT + split + len(y_test)]
expanded_time_index = np.repeat(time_index_test, N_OUTPUT)

# === PREDICT ===
y_pred = model.predict(X_test)

# Flatten outputs for inverse scaling
target_len = 4  # number of PM2.5 sites
y_test_flat = y_test.reshape(-1, target_len)
y_pred_flat = y_pred.reshape(-1, target_len)

# Pad to full features for scaler inverse transform
pad_len = len(feature_columns) - target_len
y_test_padded = np.hstack([y_test_flat, np.zeros((y_test_flat.shape[0], pad_len))])
y_pred_padded = np.hstack([y_pred_flat, np.zeros((y_pred_flat.shape[0], pad_len))])

# Inverse scale back to original units
y_test_inv = scaler.inverse_transform(y_test_padded)[:, :target_len]
y_pred_inv = scaler.inverse_transform(y_pred_padded)[:, :target_len]

# Align CTM to length of predictions (repeat for each forecast hour)
ctm_repeated = np.repeat(ctm_test, N_OUTPUT)
ctm_repeated = ctm_repeated[:len(y_pred_inv)]

# === SITES LIST ===
sites = feature_columns[:4]  # Only PM2.5 sites

# === COMPUTE MAE for CNN predictions and CTM (only Liverpool) ===
mae_cnn_dict = {}
mae_ctm_dict = {}

for i, site in enumerate(sites):
    mae_cnn_dict[site] = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
    if site == "PM2.5_LIVERPOOL":
        mae_ctm_dict[site] = mean_absolute_error(y_test_inv[:, i], ctm_repeated)
    else:
        mae_ctm_dict[site] = np.nan  # no CTM comparison for other sites

# === STREAMLIT APP ===
st.set_page_config(layout="wide")
st.title("🌫️ PM2.5 Forecast Evaluation: CNN-LSTM vs CTM")

# Show MAE table with CTM only for Liverpool
mae_df = pd.DataFrame({
    "Site": [s.replace("PM2.5_", "") for s in sites],
    "CNN-LSTM MAE": [mae_cnn_dict[s] for s in sites],
    "CTM MAE": [f"{mae_ctm_dict[s]:.2f}" if not pd.isna(mae_ctm_dict[s]) else "-" for s in sites]
})
st.markdown("### 📊 Mean Absolute Error (MAE) per Site (µg/m³)")
st.dataframe(mae_df.style.format({"CNN-LSTM MAE": "{:.2f}"}))

# Site selection for plotting
site = st.selectbox("Select site to visualize", [s.replace("PM2.5_", "") for s in sites])
site_col = f"PM2.5_{site}"

# Extract observed, predicted for selected site
site_index = sites.index(site_col)
observed = y_test_inv[:, site_index]
predicted = y_pred_inv[:, site_index]

# Only Liverpool gets CTM line
if site_col == "PM2.5_LIVERPOOL":
    ctm = ctm_repeated[:len(observed)]
else:
    ctm = None

# Plot time series
st.markdown(f"### 📈 PM2.5 at {site}: Observed vs CNN-LSTM" + (" vs CTM" if ctm is not None else ""))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(expanded_time_index[:len(observed)], observed, label="Observed", color="blue")
ax.plot(expanded_time_index[:len(predicted)], predicted, label="CNN-LSTM Prediction", color="green")
if ctm is not None:
    ax.plot(expanded_time_index[:len(ctm)], ctm, label="CTM Model", color="red")

ax.set_title(f"PM2.5 at {site} (µg/m³)")
ax.set_xlabel("Time")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

