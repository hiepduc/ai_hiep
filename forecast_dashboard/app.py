import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from tensorflow.keras.models import load_model
#from utils.forecasting_utils import create_sequences
import plotly.graph_objects as go
import geopandas as gpd

def create_sequences(data, n_input, n_output=6):
    # Assume data shape = (1, n_input, n_features)
    X, y = [], []
    for i in range(data.shape[1] - n_input - n_output + 1):
        X.append(data[:, i:i+n_input, :])
        y.append(data[:, i+n_input:i+n_input+n_output, :])
    return np.array(X)[0], np.array(y)[0]  # remove batch dim

# === CONFIG ===
st.set_page_config(layout="wide", page_title="PM2.5 Forecasting Dashboard")

REGION_PREFIX = {
    "Southwest Sydney (SW)": "sw",
    "Northwest Sydney (NW)": "nw",
    "Upper Hunter (UH)": "uh"
}

REGION_FILES = {
    "Southwest Sydney (SW)": {
        "data": "data/Imputed_data_SW_PM2.5.csv",
        "model": {
            72: "models/cnn_lstm_model_pm25_sw_72input.h5",
            120: "models/cnn_lstm_model_pm25_sw_120input.h5"
        },
        "scaler": {
            72: "scalers/pm25_sw_scaler_72.save",
            120: "scalers/pm25_sw_scaler_120.save"
        }
    },
    "Northwest Sydney (NW)": {
        "data": "data/Imputed_data_NW_PM2.5.csv",
        "model": {
            72: "models/cnn_lstm_model_pm25_nw_72input.h5",
            120: "models/cnn_lstm_model_pm25_nw_120input.h5"
        },
        "scaler": {
            72: "scalers/pm25_nw_scaler_72.save",
            120: "scalers/pm25_nw_scaler_120.save"
        }
    },
    "Upper Hunter (UH)": {
        "data": "data/Imputed_data_UH_PM2.5.csv",
        "model": {
            72: "models/cnn_lstm_model_pm25_uh_72input.h5",
            120: "models/cnn_lstm_model_pm25_uh_120input.h5"
        },
        "scaler": {
            72: "scalers/pm25_uh_scaler_72.save",
            120: "scalers/pm25_uh_scaler_120.save"
        }
    }
}

# === SIDEBAR ===
st.sidebar.title("🔍 Forecast Configuration")
region = st.sidebar.selectbox("Select Region", list(REGION_FILES.keys()))
n_input = st.sidebar.radio("Model Input Length (hours)", [72, 120], horizontal=True)

start_date = st.sidebar.date_input("Forecast Start Date", datetime.date(2025, 7, 1))
end_date = st.sidebar.date_input("Forecast End Date", datetime.date(2025, 7, 2))

# === LOAD DATA ===
data_path = REGION_FILES[region]["data"]
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
station_names = df.columns.tolist()

# Ensure enough data available before forecast date
df = df.sort_index()
#forecast_datetime = pd.Timestamp(forecast_date).tz_localize("Australia/Sydney")
#if forecast_datetime not in df.index:
#    forecast_datetime = df.index[df.index.get_indexer([forecast_datetime], method="nearest")[0]]
# Ensure datetime is timezone-aware and aligned with index resolution

forecast_datetime = pd.Timestamp(forecast_date).tz_localize("Australia/Sydney")

# Snap to the exact hour only if it's in the data
if forecast_datetime in df.index:
    pass  # use as-is
else:
    # Filter for timestamps *on or after* the selected date
    valid_times = df.index[df.index >= forecast_datetime]
    if len(valid_times) == 0:
        st.error("No data available after the selected date.")
        st.stop()
    forecast_datetime = valid_times[0]  # use the next available timestamp

st.write("🕒 Forecast will start at:", forecast_datetime)

forecast_idx = df.index.get_loc(forecast_datetime)
if forecast_idx < n_input:
    st.warning("Not enough past data to run forecast for this date.")
    st.stop()


# === PREPARE SEQUENCE ===
X_input = df.iloc[forecast_idx - n_input: forecast_idx].values
scaler = joblib.load(REGION_FILES[region]["scaler"][n_input])

# Load expected column order used during training
columns_path = f"scalers/pm25_{REGION_PREFIX[region]}_columns_{n_input}.save"
expected_cols = joblib.load(columns_path)

# Create dataframe and reindex to expected order
X_input_df = pd.DataFrame(X_input, columns=station_names)
X_input_df = X_input_df.reindex(columns=expected_cols)

# Fill missing columns with zeros if any (just in case)
X_input_df = X_input_df.fillna(0)

# Now safely scale
X_scaled = scaler.transform(X_input_df)

# Determine how to reshape input based on model design
# Set manually per model — or store this in REGION_FILES
if n_input == 72:
    n_subseq = 6   # 6 x 12 = 72
elif n_input == 120:
    n_subseq = 10  # 10 x 12 = 120
else:
    raise ValueError("Unsupported input length")

timesteps_per_subseq = n_input // n_subseq
n_features = X_scaled.shape[1]
X_seq = X_scaled.reshape((1, n_subseq, timesteps_per_subseq, n_features))


# === LOAD MODEL AND PREDICT ===
model_path = REGION_FILES[region]["model"][n_input]
model = load_model(model_path)
y_pred = model.predict(X_seq).reshape(-1, n_features)
y_pred_inv = scaler.inverse_transform(y_pred)

# === VISUALIZE ===
st.title("📈 PM2.5 Forecasting Dashboard")
st.markdown(f"**Region:** {region}  |  **Forecast Date:** {forecast_datetime.strftime('%Y-%m-%d %H:%M')}  |  **Input Hours:** {n_input}")
#st.markdown(f"**Region:** {region}  |  **Forecast Date:** {forecast_datetime.strftime('%Y-%m-%d %H:%M')}  |  **Input Hours:** {n_input}")

forecast_hours = pd.date_range(start=forecast_datetime, periods=6, freq='H')
df_pred = pd.DataFrame(y_pred_inv, index=forecast_hours, columns=station_names)

fig = go.Figure()
for site in station_names:
    fig.add_trace(go.Scatter(x=forecast_hours, y=df_pred[site], mode="lines+markers", name=site))

fig.update_layout(
    title="PM2.5 Forecast (Next 6 Hours)",
    xaxis_title="Datetime",
    yaxis_title="PM2.5 (µg/m³)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# === MAP OF STATIONS ===
st.subheader("🗺️ Station Locations")
gdf = gpd.read_file("stations.geojson")
gdf_selected = gdf[gdf["Region"] == region.split(" ")[0]]
st.map(gdf_selected)


