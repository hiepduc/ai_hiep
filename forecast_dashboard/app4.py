import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

# -------------------------------
# API Data Fetching
# -------------------------------
def fetch_pm25_data(start_date_str, end_date_str, region):
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

    region_sites = {
        "SW": ["BARGO", "BRINGELLY", "CAMPBELLTOWN WEST", "LIVERPOOL"],
        "NW": ["PARRAMATTA NORTH", "RICHMOND", "ROUSE HILL"],
        "UH": ["MUSWELLBROOK", "SINGLETON", "MERRIWA"]
    }
    target_sites = region_sites.get(region.upper(), [])
    target_ids = {s: site_map[s] for s in target_sites if s in site_map}

    parameter_id = param_map["PM2.5"]
    site_dfs = []
    present_sites = []

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
                records.append({"datetime": dt, f"PM2.5_{site_name.replace(' ', '_')}": rec["Value"]})

        if records:
            df_site = pd.DataFrame(records).set_index("datetime")
            site_dfs.append(df_site)
            present_sites.append(site_name)

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    missing = [s for s in target_sites if s not in present_sites]
    for m in missing:
        col = f"PM2.5_{m.replace(' ', '_')}"
        df_api[col] = df_api.mean(axis=1)
        st.info(f"Filled missing site {m} with mean.")

    df_api = df_api.dropna()
    return df_api

# -------------------------------
# Rolling Forecast
# -------------------------------
def run_forecast(df_recent, model, scaler, forecast_hours, feature_columns):
    input_hours = 72
    n_subseq = 6
    n_steps_per_subseq = 12

    df_input = df_recent[-input_hours:][feature_columns]
    df_input = df_input[feature_columns]
    current_input = scaler.transform(df_input)
    n_features = current_input.shape[1]

    predictions_scaled = []
    step_ahead = 6
    n_loops = forecast_hours // step_ahead

    for _ in range(n_loops):
        x_input = current_input.reshape((1, n_subseq, n_steps_per_subseq, n_features))
        yhat = model.predict(x_input, verbose=0)[0]
        predictions_scaled.append(yhat)
        current_input = np.vstack((current_input[step_ahead:], yhat))

    predictions_scaled = np.vstack(predictions_scaled)
    pred_df_scaled = pd.DataFrame(predictions_scaled, columns=feature_columns)
    predictions = scaler.inverse_transform(pred_df_scaled)

    forecast_start = df_recent.index[-1] + pd.Timedelta(hours=1)
    forecast_index = pd.date_range(forecast_start, periods=forecast_hours, freq='H')
    forecast_df = pd.DataFrame(predictions, columns=feature_columns, index=forecast_index)
    forecast_df.index.name = "datetime"

    return forecast_df

# -------------------------------
# UI: Streamlit App
# -------------------------------
st.set_page_config(layout="wide")
st.title("🌫️ PM2.5 Forecast Dashboard (CNN-LSTM Model)")

col1, col2, col3 = st.columns(3)
with col1:
    region = st.selectbox("Region", ["SW", "NW", "UH"])
with col2:
    start_date = st.date_input("Start Date", value=datetime(2025, 7, 1))
with col3:
    end_date = st.date_input("End Date", value=datetime(2025, 7, 7))

forecast_days = st.slider("Forecast Horizon (days)", 1, 10, value=3)
forecast_hours = forecast_days * 24

if st.button("Run Forecast"):
    with st.spinner("Fetching data and running model..."):
        df = fetch_pm25_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), region)
        if len(df) < 72:
            st.error("❌ Not enough past data for forecasting (need at least 72 hours).")
            st.stop()

        model = load_model(f"models/cnn_lstm_model_pm25_{region.lower()}_72input.h5")
        scaler = joblib.load(f"scalers/pm25_{region.lower()}_scaler_72.save")
        feature_columns = joblib.load(f"scalers/pm25_{region.lower()}_columns_72.save")
        forecast_df = run_forecast(df, model, scaler, forecast_hours, feature_columns)

    st.success(f"✅ Forecast completed for {forecast_days} days ({forecast_hours} hours).")

    # Merge for plotting
    observed_df = df.tail(72).copy()
    observed_df["type"] = "Observed"
    forecast_df["type"] = "Forecast"

    combined = pd.concat([observed_df, forecast_df]).reset_index()
    df_long = combined.melt(id_vars=["datetime", "type"], var_name="Site", value_name="PM2.5")
    df_long["datetime"] = pd.to_datetime(df_long["datetime"])
    df_long["datetime"] = df_long["datetime"].dt.tz_localize(None)

    fig = px.line(df_long, x="datetime", y="PM2.5", color="Site", line_dash="type",
                  title="PM2.5 Forecast vs Observed", labels={"PM2.5": "PM2.5 (µg/m³)"})
    fig.update_layout(legend_title="Site")
    st.plotly_chart(fig, use_container_width=True)

    csv = forecast_df.drop(columns="type").to_csv().encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "pm25_forecast.csv")

    st.markdown("### 📋 Forecast Summary (mean PM2.5 µg/m³)")
    st.dataframe(forecast_df.drop(columns="type").mean().round(2).to_frame("Mean PM2.5"))

