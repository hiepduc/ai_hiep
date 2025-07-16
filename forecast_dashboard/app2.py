import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
from datetime import timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import geopandas as gpd

# --------------------------
# API Data Fetch Function
# --------------------------
def get_forecast_from_api(start_date_str, end_date_str):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    site_map = {site["SiteName"]: site["Site_Id"]
                for site in requests.get(sites_url, headers=HEADERS).json()}
    param_map = {}
    for param in requests.get(params_url, headers=HEADERS).json():
        code = param.get("ParameterCode")
        freq = param.get("Frequency", "").lower()
        if code and code not in param_map and "hour" in freq:
            param_map[code] = param.get("ParameterCode")

    target_sites = {
        "BARGO": site_map["BARGO"],
        "BRINGELLY": site_map["BRINGELLY"],
        "CAMPBELLTOWN_WEST": site_map["CAMPBELLTOWN WEST"],
        "LIVERPOOL": site_map["LIVERPOOL"]
    }

    parameter_id = param_map["PM2.5"]
    site_dfs = []
    present_sites = []

    for site_name, site_id in target_sites.items():
        payload = {
            "Sites": [site_id],
            "Parameters": [parameter_id],
            "StartDate": start_date_str,
            "EndDate": end_date_str,
            "Categories": ["Averages"],
            "SubCategories": ["Hourly"],
            "Frequency": ["Hourly average"]
        }

        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.warning(f"Error fetching data for {site_name}: {e}")
            continue

        records = []
        for rec in data:
            if rec["Value"] is not None:
                dt = datetime.datetime.strptime(rec["Date"], "%Y-%m-%d") + timedelta(hours=rec["Hour"])
                records.append({"datetime": dt, f"PM2.5_{site_name}": rec["Value"]})

        if records:
            df_site = pd.DataFrame(records).set_index("datetime")
            site_dfs.append(df_site)
            present_sites.append(site_name)
        else:
            st.warning(f"⚠️ No valid data for site {site_name}, skipping.")

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    # Fill in missing sites with mean
    missing_sites = [s for s in target_sites if s not in present_sites]
    for site in missing_sites:
        col_name = f"PM2.5_{site}"
        df_api[col_name] = df_api.mean(axis=1)
        st.info(f"Interpolated missing site {site} with mean.")

    return df_api.dropna()


# --------------------------
# Streamlit App
# --------------------------

st.set_page_config(layout="wide", page_title="PM2.5 Forecasting Dashboard")

REGION_PREFIX = {
    "Southwest Sydney (SW)": "sw",
    "Northwest Sydney (NW)": "nw",
    "Upper Hunter (UH)": "uh"
}

REGION_FILES = {
    "Southwest Sydney (SW)": {
        "model": {
            72: "models/cnn_lstm_model_pm25_sw_72input.h5",
            120: "models/cnn_lstm_model_pm25_sw_120input.h5"
        },
        "scaler": {
            72: "scalers/pm25_sw_scaler_72.save",
            120: "scalers/pm25_sw_scaler_120.save"
        }
    },
    # Add other regions similarly if needed
}

# === SIDEBAR ===
st.sidebar.title("🔍 Forecast Configuration")
region = st.sidebar.selectbox("Select Region", list(REGION_FILES.keys()))
n_input = st.sidebar.radio("Model Input Length (hours)", [72, 120], horizontal=True)
start_date = st.sidebar.date_input("Forecast Start Date", datetime.date(2025, 7, 1))
end_date = st.sidebar.date_input("Forecast End Date", datetime.date(2025, 7, 2))

# === FETCH DATA ===
st.info("📡 Fetching PM2.5 data from NSW Air Quality API...")
start_buffer_date = start_date - datetime.timedelta(hours=n_input)
df = get_forecast_from_api(start_buffer_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
df = df.sort_index()
station_names = df.columns.tolist()

# === LOAD MODEL AND SCALER ===
model_path = REGION_FILES[region]["model"][n_input]
scaler_path = REGION_FILES[region]["scaler"][n_input]
scaler = joblib.load(scaler_path)

columns_path = f"scalers/pm25_{REGION_PREFIX[region]}_columns_{n_input}.save"
expected_cols = joblib.load(columns_path)

model = load_model(model_path)

# === FORECAST LOOP ===
n_subseq = 6 if n_input == 72 else 10
timesteps_per_subseq = n_input // n_subseq
n_features = len(expected_cols)

#forecast_times = pd.date_range(start=start_date, end=end_date, freq='H')
#all_predictions = []

#for forecast_time in forecast_times:
#    if forecast_time not in df.index:
#        continue
#
#    idx = df.index.get_indexer([forecast_time])[0]
#    if idx < n_input:
#        continue

#    X_input = df.iloc[idx - n_input:idx].values
#    X_input_df = pd.DataFrame(X_input, columns=station_names)
#    X_input_df = X_input_df.reindex(columns=expected_cols).fillna(0)
#    X_scaled = scaler.transform(X_input_df)
#    X_seq = X_scaled.reshape((1, n_subseq, timesteps_per_subseq, n_features))

#    y_pred = model.predict(X_seq).reshape(-1, n_features)
#    y_pred_inv = scaler.inverse_transform(y_pred)

#    pred_times = pd.date_range(start=forecast_time, periods=6, freq='H')
#    df_pred = pd.DataFrame(y_pred_inv, index=pred_times, columns=expected_cols)
#    all_predictions.append(df_pred)
#if not all_predictions:
#    st.error("❌ No forecast could be generated. Try a different date range.")
#    st.stop()

#df_all = pd.concat(all_predictions)

forecast_time = pd.Timestamp(end_date)

idx = df.index.get_indexer([forecast_time])[0]
if idx < n_input:
    st.error("Not enough data before forecast time.")
    st.stop()

X_input = df.iloc[idx - n_input:idx].values
X_input_df = pd.DataFrame(X_input, columns=station_names)
X_input_df = X_input_df.reindex(columns=expected_cols).fillna(0)
X_scaled = scaler.transform(X_input_df)
X_seq = X_scaled.reshape((1, n_subseq, timesteps_per_subseq, n_features))

y_pred = model.predict(X_seq).reshape(-1, n_features)
y_pred_inv = scaler.inverse_transform(y_pred)

pred_times = pd.date_range(start=forecast_time, periods=6, freq='H')
df_all = pd.DataFrame(y_pred_inv, index=pred_times, columns=expected_cols)

# === VISUALIZE ===
st.title("📈 PM2.5 Forecasting Dashboard")
st.markdown(f"**Region:** {region}  |  **Forecast:** {start_date} to {end_date}  |  **Input Hours:** {n_input}")

fig = go.Figure()
for site in expected_cols:
    fig.add_trace(go.Scatter(x=df_all.index, y=df_all[site], mode="lines", name=site))

fig.update_layout(
    title="Forecasted PM2.5 for Selected Period",
    xaxis_title="Datetime",
    yaxis_title="PM2.5 (µg/m³)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# === STATION MAP ===
st.subheader("🗺️ Station Locations")
try:
    gdf = gpd.read_file("stations.geojson")
    gdf_selected = gdf[gdf["Region"] == region.split(" ")[0]]
    st.map(gdf_selected)
except:
    st.warning("⚠️ Unable to load station map.")

