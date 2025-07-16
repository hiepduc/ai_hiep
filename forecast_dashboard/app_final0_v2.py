import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import folium
from streamlit_folium import st_folium

# ------------------------------
# Station Coordinates
# ------------------------------
site_coords = {
    "PARRAMATTA_NORTH": {"lat": -33.797, "lon": 151.002},
    "RICHMOND": {"lat": -33.60, "lon": 150.7514},
    "ROUSE_HILL": {"lat": -33.68, "lon": 150.92},
    "BARGO": {"lat": -34.30, "lon": 150.57},
    "CAMPBELLTOWN_WEST": {"lat": -34.07, "lon": 150.82},
    "LIVERPOOL": {"lat": -33.92, "lon": 150.923},
    "BRINGELLY": {"lat": -33.93, "lon": 150.73},
    "SINGLETON": {"lat": -32.57, "lon": 151.178},
    "MUSWELLBROOK": {"lat": -32.261, "lon": 150.89},
    "MERRIWA": {"lat": -32.139, "lon": 150.356}
}

# ------------------------------
# Fetch PM2.5 Data
# ------------------------------
def fetch_pm25_data(start_date_str, end_date_str, region):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    site_map = {s["SiteName"]: s["Site_Id"] for s in requests.get(sites_url, headers=HEADERS).json()}
    param_map = {
        p["ParameterCode"]: p["ParameterCode"]
        for p in requests.get(params_url, headers=HEADERS).json()
        if "hour" in p.get("Frequency", "").lower()
    }

    region_sites = {
        "SW": ["BARGO", "BRINGELLY", "CAMPBELLTOWN WEST", "LIVERPOOL"],
        "NW": ["PARRAMATTA NORTH", "RICHMOND", "ROUSE HILL"],
        "UH": ["MUSWELLBROOK", "SINGLETON", "MERRIWA"]
    }

    parameter_id = param_map["PM2.5"]
    target_sites = region_sites[region.upper()]
    target_ids = {s: site_map[s] for s in target_sites if s in site_map}

    site_dfs, present_sites = [], []

    for site_name, site_id in target_ids.items():
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
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60).json()
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch {site_name}: {e}")
            continue

        records = [
            {
                "datetime": datetime.strptime(r["Date"], "%Y-%m-%d") + timedelta(hours=r["Hour"]),
                f"PM2.5_{site_name.replace(' ', '_')}": r["Value"]
            }
            for r in resp if r["Value"] is not None
        ]
        if records:
            site_df = pd.DataFrame(records).set_index("datetime")
            site_dfs.append(site_df)
            present_sites.append(site_name)

    df_api = pd.concat(site_dfs, axis=1).sort_index()

    for missing in [s for s in target_sites if s not in present_sites]:
        col = f"PM2.5_{missing.replace(' ', '_')}"
        df_api[col] = df_api.mean(axis=1)
        st.info(f"ℹ️ Filled missing site {missing} with mean.")

    return df_api.dropna()

# ------------------------------
# Forecast Function
# ------------------------------
def run_forecast(df, model, scaler, feature_columns, n_input, forecast_hours):
    n_subseq = n_input // 12
    n_steps = n_input // n_subseq
    df_input = df[-n_input:][feature_columns]
    scaled_input = scaler.transform(df_input)
    X = scaled_input.reshape((1, n_subseq, n_steps, scaled_input.shape[1]))

    predictions_scaled = []
    for _ in range(forecast_hours // 6):
        yhat = model.predict(X, verbose=0)[0]
        predictions_scaled.append(yhat)
        scaled_input = np.vstack([scaled_input[6:], yhat])
        X = scaled_input.reshape((1, n_subseq, n_steps, scaled_input.shape[1]))

    y_pred = scaler.inverse_transform(np.vstack(predictions_scaled))
    forecast_start = df.index[-1] + timedelta(hours=1)
    return pd.DataFrame(y_pred, index=pd.date_range(forecast_start, periods=forecast_hours, freq='H'), columns=feature_columns)

# ------------------------------
# App Layout
# ------------------------------
st.set_page_config(layout="wide")
st.title("🌫️ PM2.5 Forecast Dashboard")

main_col, map_col = st.columns([5, 1])

# --- Station Map ---
with map_col:
    st.markdown("### 🧭 Station Map")
    fmap = folium.Map(location=[-33.8, 150.8], zoom_start=8)
    for site, loc in site_coords.items():
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            tooltip=site.replace("_", " ").title(),
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(fmap)
    st_folium(fmap, width=250, height=300, returned_objects=[])

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Model Configuration")
    model_type = st.selectbox("Model Type", ["CNN_LSTM", "CNN_LSTM_MET (coming)", "CNN_LSTM_TRANSFORMER (coming)"])
    regions_selected = st.multiselect("Select Regions", ["SW", "NW", "UH"], default=["SW"])
    n_input = st.radio("Model Input Hours", [72, 120], horizontal=True)
    forecast_days = st.slider("Forecast Horizon (days)", 1, 7, 3)
    start_date = st.date_input("Start Date", datetime(2025, 7, 1))
    end_date = st.date_input("End Date", datetime(2025, 7, 10))

# --- Main Forecast Logic ---
forecast_hours = forecast_days * 24
all_forecasts = []
all_errors = []

with main_col:
    if st.button("🚀 Run Forecast"):
        for region in regions_selected:
            st.subheader(f"🔍 Region: {region}")

            df_obs = fetch_pm25_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), region)
            if len(df_obs) < n_input:
                st.warning(f"Not enough past data for {region}. Need at least {n_input} hours.")
                continue

            try:
                if model_type == "CNN_LSTM":
                    model = load_model(f"models/cnn_lstm_model_pm25_{region.lower()}_{n_input}input.h5")
                    scaler = joblib.load(f"scalers/pm25_{region.lower()}_scaler_{n_input}.save")
                    features = joblib.load(f"scalers/pm25_{region.lower()}_columns_{n_input}.save")
                else:
                    st.info(f"Model type '{model_type}' is a placeholder.")
                    continue
            except Exception as e:
                st.error(f"Missing model/scaler/columns for {region}: {e}")
                continue

            df_obs = df_obs[features].dropna()
            forecast_df = run_forecast(df_obs, model, scaler, features, n_input, forecast_hours)
            forecast_df["Region"] = region
            all_forecasts.append(forecast_df)

            # Fetch actuals for future
            actual_future = fetch_pm25_data(
                (df_obs.index[-1] + timedelta(hours=1)).strftime("%Y-%m-%d"),
                (df_obs.index[-1] + timedelta(hours=forecast_hours)).strftime("%Y-%m-%d"),
                region
            )
            actual_future = actual_future[features].reindex(forecast_df.index).dropna()
            common_index = forecast_df.index.intersection(actual_future.index)

            if not actual_future.empty:
                for col in features:
                    mae = mean_absolute_error(actual_future[col], forecast_df.loc[common_index, col])
                    rmse = mean_squared_error(actual_future[col], forecast_df.loc[common_index, col], squared=False)
                    all_errors.append({"Region": region, "Site": col, "MAE": mae, "RMSE": rmse})

            # Plot observed + forecast
            for col in features:
                fig_df = pd.concat([
                    df_obs[[col]].rename(columns={col: "Observed"}),
                    forecast_df[[col]].rename(columns={col: "Forecast"})
                ])
                fig_df["datetime"] = fig_df.index
                fig = px.line(fig_df, x="datetime", y=["Observed", "Forecast"],
                              title=col.replace("_", " "), labels={"value": "PM2.5", "variable": "Type"})
                st.plotly_chart(fig, use_container_width=True)

        if all_errors:
            st.markdown("### 🧪 Forecast Accuracy")
            st.dataframe(pd.DataFrame(all_errors).style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))

        if all_forecasts:
            combined_df = pd.concat(all_forecasts)
            csv = combined_df.drop(columns="Region").to_csv().encode("utf-8")
            st.download_button("📥 Download Forecast CSV", csv, file_name="pm25_forecast.csv")

