import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------
# API: Fetch hourly PM2.5 data
# -----------------------------------
def fetch_pm25_data(start_date_str, end_date_str, region):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    site_map = {site["SiteName"]: site["Site_Id"] for site in requests.get(sites_url, headers=HEADERS).json()}
    param_map = {
        p.get("ParameterCode"): p.get("ParameterCode")
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
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30).json()
        except Exception as e:
            st.warning(f"Failed to fetch {site_name}: {e}")
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

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    for missing in [s for s in target_sites if s not in present_sites]:
        col = f"PM2.5_{missing.replace(' ', '_')}"
        df_api[col] = df_api.mean(axis=1)
        st.info(f"Filled missing site {missing} with mean.")

    return df_api.dropna()


# -----------------------------------
# Rolling Forecast Function
# -----------------------------------
def run_forecast(df, model, scaler, feature_columns, n_input, forecast_hours):
    if n_input == 72:
        n_subseq = 6
    elif n_input == 120:
        n_subseq = 10
    else:
        raise ValueError("Unsupported input")

    step = forecast_hours // 6
    n_steps = n_input // n_subseq
    df_input = df[-n_input:][feature_columns]
    scaled_input = scaler.transform(df_input)
    n_features = scaled_input.shape[1]

    X = scaled_input.reshape((1, n_subseq, n_steps, n_features))
    result_scaled = []
    for _ in range(step):
        yhat = model.predict(X, verbose=0)[0]
        result_scaled.append(yhat)
        scaled_input = np.vstack([scaled_input[6:], yhat])
        X = scaled_input.reshape((1, n_subseq, n_steps, n_features))

    y_pred_scaled = np.vstack(result_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    forecast_start = df.index[-1] + timedelta(hours=1)
    time_index = pd.date_range(forecast_start, periods=forecast_hours, freq="H")
    return pd.DataFrame(y_pred, index=time_index, columns=feature_columns)


# -----------------------------------
# Streamlit App UI
# -----------------------------------
st.set_page_config(layout="wide")
st.title("🌫️ Multi-Region PM2.5 Forecast Dashboard (CNN-LSTM)")

with st.sidebar:
    regions_selected = st.multiselect("Select Regions", ["SW", "NW", "UH"], default=["SW"])
    n_input = st.radio("Model Input Hours", [72, 120], index=0, horizontal=True)
    forecast_days = st.slider("Forecast Horizon (days)", 1, 7, 3)
    start_date = st.date_input("Start Date", value=datetime(2025, 7, 1))
    end_date = st.date_input("End Date", value=datetime(2025, 7, 10))

forecast_hours = forecast_days * 24
all_forecasts = []
all_errors = []

if st.button("Run Forecast"):
    for region in regions_selected:
        st.subheader(f"🔍 Region: {region}")
        df = fetch_pm25_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), region)
        if len(df) < n_input:
            st.warning(f"Not enough past data for {region}. Need at least {n_input} hours.")
            continue

        try:
            model = load_model(f"models/cnn_lstm_model_pm25_{region.lower()}_{n_input}input.h5")
            scaler = joblib.load(f"scalers/pm25_{region.lower()}_scaler_{n_input}.save")
            features = joblib.load(f"scalers/pm25_{region.lower()}_columns_{n_input}.save")
        except Exception as e:
            st.error(f"Missing model/scaler/columns for {region}: {e}")
            continue

        df = df[features].dropna()
        forecast_df = run_forecast(df, model, scaler, features, n_input, forecast_hours)
        forecast_df["Region"] = region
        all_forecasts.append(forecast_df)

        # Compare with actual if available
        future_actual = fetch_pm25_data(
            (df.index[-1] + timedelta(hours=1)).strftime("%Y-%m-%d"),
            (df.index[-1] + timedelta(hours=forecast_hours)).strftime("%Y-%m-%d"),
            region
        )
        actual = future_actual[features].reindex(forecast_df.index).dropna()
        common_index = forecast_df.index.intersection(actual.index)

        if not actual.empty:
            for col in features:
                mae = mean_absolute_error(actual[col], forecast_df.loc[common_index, col])
                rmse = mean_squared_error(actual[col], forecast_df.loc[common_index, col], squared=False)
                all_errors.append({"Region": region, "Site": col, "MAE": mae, "RMSE": rmse})

    # ------------------------------
    # Combine and Visualize Forecasts
    # ------------------------------
    if all_forecasts:
        combined_df = pd.concat(all_forecasts)
        combined_long = combined_df.drop(columns="Region").copy()
        combined_long["datetime"] = combined_df.index
        combined_long = combined_long.melt(id_vars=["datetime"], var_name="Site", value_name="PM2.5")

        fig = px.line(combined_long, x="datetime", y="PM2.5", color="Site",
                      title="PM2.5 Forecast (All Regions)")
        st.plotly_chart(fig, use_container_width=True)

        # Download forecast
        forecast_download = combined_df.drop(columns="Region").copy()
        csv = forecast_download.to_csv().encode("utf-8")
        st.download_button("📥 Download Forecast CSV", csv, "multi_region_pm25_forecast.csv")

    # ------------------------------
    # Show Error Table
    # ------------------------------
    if all_errors:
        st.markdown("### 🧪 Forecast Accuracy (Compared to Observed)")
        err_df = pd.DataFrame(all_errors)
        st.dataframe(err_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))

