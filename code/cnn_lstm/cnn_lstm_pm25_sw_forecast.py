import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import argparse
import os

# --------------------------
# Fetch PM2.5 data from API
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
            print(f"Error fetching data for {site_name}: {e}")
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
            print(f"⚠️ No valid data for site {site_name}, skipping.")

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    # Interpolate missing site columns with mean of available ones
    missing_sites = [s for s in target_sites.keys() if s not in present_sites]
    for site in missing_sites:
        col_name = f"PM2.5_{site}"
        df_api[col_name] = df_api.mean(axis=1)
        print(f"ℹ️ Interpolated missing site {site} using mean of other sites.")

    df_api = df_api.dropna()
    return df_api


# --------------------------
# Argument Parser
# --------------------------
parser = argparse.ArgumentParser(description="PM2.5 Forecast Script using CNN-LSTM")
parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD")
parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD")
args = parser.parse_args()

# --------------------------
# Load Model and Scaler
# --------------------------
model = load_model("cnn_lstm_model_pm25.h5")
scaler = joblib.load("pm25_scaler.save")

# --------------------------
# Get data and forecast
# --------------------------
df = get_forecast_from_api(args.start, args.end)

if len(df) < 24:
    raise ValueError("❌ Not enough data for forecasting. At least 24 hours are required.")

initial_input = scaler.transform(df[-24:].values)

# Forecast 6 days = 144 hours (6-hour steps)
n_forecast_hours = 144
n_features = initial_input.shape[1]
n_steps = 24
n_subseq = 4
n_steps_per_subseq = n_steps // n_subseq

predictions_scaled = []
current_input = initial_input.copy()

for _ in range(n_forecast_hours // 6):
    x_input = current_input.reshape((1, n_subseq, n_steps_per_subseq, n_features))
    yhat = model.predict(x_input, verbose=0)[0]
    predictions_scaled.append(yhat)
    current_input = np.vstack((current_input[6:], yhat))

predictions_scaled = np.vstack(predictions_scaled)
predictions = scaler.inverse_transform(predictions_scaled)

forecast_start = df.index[-1] + pd.Timedelta(hours=1)
forecast_index = pd.date_range(forecast_start, periods=n_forecast_hours, freq='H')
forecast_df = pd.DataFrame(predictions, columns=df.columns, index=forecast_index)
forecast_df.index.name = "datetime"

# Save output
outname = f"forecast_PM25_SW_{forecast_start.strftime('%Y%m%d')}_{(forecast_start + pd.Timedelta(hours=n_forecast_hours - 1)).strftime('%Y%m%d')}.csv"
forecast_df.to_csv(outname)
print(f"✅ Forecast saved to {outname}")
print(forecast_df.head())

import matplotlib.pyplot as plt

forecast_df.plot(figsize=(12, 6))
plt.title(f"Forecast PM2.5 at SW Sydney Sites {forecast_start.strftime('%Y%m%d')}_{(forecast_start + pd.Timedelta(hours=n_forecast_hours - 1)).strftime('%Y%m%d')}")
plt.ylabel("PM2.5 (µg/m³)")
plt.grid(True)
plt.tight_layout()
plt.show()

