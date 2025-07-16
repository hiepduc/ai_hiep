import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import geopandas as gpd

# ========== CONFIG ==========
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
    # Add NW, UH if you have models
}

# ========== API CALL ==========
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

    # Sites for Southwest Sydney
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
                dt = datetime.strptime(rec["Date"], "%Y-%m-%d") + timedelta(hours=rec["Hour"])
                records.append({"datetime": dt, f"PM2.5_{site_name}": rec["Value"]})

        if records:
            df_site = pd.DataFrame(records).set_index("datetime")
            site_dfs.append(df_site)
            present_sites.append(site_name)
        else:
            st.warning(f"⚠️ No valid data for site {site_name}, skipping.")

    if not site_dfs:
        st.error("❌ No site data available from API.")
        st.stop()

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    # Fill missing site columns
    missing_sites = [s for s in target_sites if s not in present_sites]
    for site in missing_sites:
        col_name = f"PM2.5_{site}"
        df_api[col_name] = df_api.mean(axis=1)
        st.info(f"Interpolated missing site {site} with mean.")

    return df_api.dropna()

# ========== UI ==========
st.sidebar.title("🔍 Forecast Configuration")
region = st.sidebar.selectbox("Select Region", list(REGION_FILES.keys()))
n_input = st.sidebar.radio("Model Input Length (hours)", [72, 120], horizontal=True)
start_date = st.sidebar.date_input("Start Date", value=datetime(2025, 7, 1).date())
end_date = st.sidebar.date_input("End Date", value=datetime(2025, 7, 5).date())

if end_date <= start_date:
    st.error("End date must be after start date.")
    st.stop()

# ========== FETCH DATA ==========
df = get_forecast_from_api(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
df = df.sort_index()

# ========== MODEL INPUT ==========
forecast_time = df.index.max()
forecast_idx = df.index.get_loc(forecast_time)

if forecast_idx < n_input:
    st.error("❌ Not enough historical data before forecast time.")
    st.stop()

X_input = df.iloc[forecast_idx - n_input: forecast_idx].values
station_names = df.columns.tolist()

# ========== LOAD SCALER & REORDER COLUMNS ==========
scaler = joblib.load(REGION_FILES[region]["scaler"][n_input])
columns_path = f"scalers/pm25_{REGION_PREFIX[region]}_columns_{n_input}.save"
expected_cols = joblib.load(columns_path)

X_input_df = pd.DataFrame(X_input, columns=station_names)
X_input_df = X_input_df.reindex(columns=expected_cols).fillna(0)
X_scaled = scaler.transform(X_input_df)

# ========== RESHAPE ==========
n_features = X_scaled.shape[1]
if n_input == 72:
    n_subseq = 6
elif n_input == 120:
    n_subseq = 10
else:
    st.error("Unsupported model input length.")
    st.stop()

X_seq = X_scaled.reshape((1, n_subseq, n_input // n_subseq, n_features))

# ========== LOAD MODEL & PREDICT ==========
model_path = REGION_FILES[region]["model"][n_input]
model = load_model(model_path)

y_pred = model.predict(X_seq).reshape(-1, n_features)
y_pred_inv = scaler.inverse_transform(y_pred)

# ========== FORECAST TIME RANGE ==========
forecast_hours = pd.date_range(start=forecast_time + pd.Timedelta(hours=1), periods=6, freq='H')
df_pred = pd.DataFrame(y_pred_inv, index=forecast_hours, columns=expected_cols)

# ========== VISUALIZATION ==========
st.title("📈 PM2.5 Forecasting Dashboard")
st.markdown(f"**Region:** {region}  |  **Forecast Based on:** {forecast_time.strftime('%Y-%m-%d %H:%M')}  |  **Model Input:** {n_input}h")

fig = go.Figure()
for site in expected_cols:
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred[site], mode="lines+markers", name=site))

fig.update_layout(
    title="PM2.5 Forecast (Next 6 Hours)",
    xaxis_title="Time",
    yaxis_title="PM2.5 (µg/m³)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ========== MAP ==========
st.subheader("🗺️ Station Locations")
try:
    gdf = gpd.read_file("stations.geojson")
    gdf_selected = gdf[gdf["Region"] == region.split(" ")[0]]
    st.map(gdf_selected)
except:
    st.warning("Could not load station map.")

