import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

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
        "PARRAMATTA_NORTH": site_map["PARRAMATTA NORTH"],
        "RICHMOND": site_map["RICHMOND"],
        "ROUSE_HILL": site_map["ROUSE HILL"]
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
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
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

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    # Fill in missing sites with mean
    missing_sites = [s for s in target_sites if s not in present_sites]
    for site in missing_sites:
        col_name = f"PM2.5_{site}"
        df_api[col_name] = df_api.mean(axis=1)
        st.info(f"Interpolated missing site {site} with mean.")

    df_api = df_api.dropna()
    return df_api

# --------------------------
# Forecast Function
# --------------------------
def run_forecast(df_recent, model, scaler, forecast_hours, feature_columns):
    input_hours = 120
    n_subseq = 10
    n_steps_per_subseq = 12

    df_input = df_recent[-input_hours:][feature_columns]
    df_input = df_input[feature_columns]  # <--- enforce column order explicitly

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


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(layout="wide")
st.title("🌫️ PM2.5 Forecast (CNN-LSTM Model)")

# Select input date range and forecast horizon
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2025, 7, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2025, 7, 7))
with col3:
    forecast_days = st.number_input("Forecast Horizon (Days)", min_value=1, max_value=14, value=6, step=1)

if start_date >= end_date:
    st.warning("⚠️ End date must be after start date.")
    st.stop()

if st.button("Run Forecast"):
    try:
        with st.spinner("Fetching data and running forecast..."):
            df = get_forecast_from_api(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if len(df) < 24:
                st.error("❌ Not enough data (need at least 24 hours of data).")
                st.stop()

            # Load model and scaler
            model = load_model("cnn_lstm_model_pm25_nw_120input.h5")

            forecast_hours = forecast_days * 24
            scaler = joblib.load("pm25_nw_scaler.save")
            feature_columns = joblib.load("pm25_nw_columns.save")
            forecast_df = run_forecast(df, model, scaler, forecast_hours, feature_columns)

        st.success(f"✅ Forecast completed for {forecast_days} days ({forecast_hours} hours)!")

        # Download CSV
        csv = forecast_df.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download CSV", csv, file_name="pm25_forecast.csv", mime='text/csv')

        # --------------------------
        # Combine observed and forecast
        # --------------------------
        n_past_hours = min(120, len(df))
        observed_df = df.tail(n_past_hours).copy()
        observed_df["type"] = "Observed"

        forecast_df_display = forecast_df.copy()
        forecast_df_display["type"] = "Forecast"

        combined_df = pd.concat([observed_df, forecast_df_display])
        combined_df = combined_df.reset_index()

        # Melt for Altair
        df_melted = combined_df.melt(id_vars=["datetime", "type"],
                                     var_name="Site", value_name="PM2.5")

        df_melted["datetime"] = pd.to_datetime(df_melted["datetime"])
        df_melted["datetime"] = df_melted["datetime"].dt.tz_localize(None)

        # Debug output
        st.markdown("### 📊 Melted Data Preview")
        st.dataframe(df_melted.head())
        st.write("🔍 Unique Sites:", df_melted["Site"].unique())
        st.write("🔍 Unique Types:", df_melted["type"].unique())

        import plotly.express as px

        if not df_melted.empty:
            st.markdown("### 📈 PM2.5 Forecast vs Observed")

            # Plotly line plot with color by Site and line dash by type
            fig = px.line(
                df_melted,
                x="datetime",
                y="PM2.5",
                color="Site",
                line_dash="type",
                labels={"datetime": "Date & Time", "PM2.5": "PM2.5 (µg/m³)"},
                title="PM2.5 Forecast vs Observed",
                hover_data={"datetime": "|%Y-%m-%d %H:%M:%S", "PM2.5": ":.2f"}
            )
            fig.update_layout(legend_title_text='Monitoring Site')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Nothing to plot — no data available.")

    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")

