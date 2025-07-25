import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import argparse

# Load the trained CNN-LSTM model and scaler
model = load_model('cnn_lstm_pm25_met_72input.h5')
scaler = joblib.load("pm25_scaler.save")
feature_columns = joblib.load("pm25_columns.save")

# --------------------------
# Argument Parser
# --------------------------
parser = argparse.ArgumentParser(description="PM2.5 Forecast Script using CNN-LSTM")
parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD")
parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD")
args = parser.parse_args()


def convert_wind_components(df_api, site_name):
    """
    Convert WSP (wind speed) and WDR (wind direction) to U (east-west) and V (north-south) components.
    """
    wsp_column = f"WSP_{site_name}"
    wdr_column = f"WDR_{site_name}"

    # Check if the columns exist in the dataframe
    if wsp_column in df_api and wdr_column in df_api:
        df_api[wdr_column] = np.deg2rad(df_api[wdr_column])  # Convert to radians
        df_api[f"LIV_U"] = df_api[wsp_column] * np.sin(df_api[wdr_column])
        df_api[f"LIV_V"] = df_api[wsp_column] * np.cos(df_api[wdr_column])

    return df_api

def fetch_recent_data(start_date_str, end_date_str):
    """
    Fetch recent PM2.5 and Liverpool meteorological data from API.
    Returns a cleaned DataFrame with the required model input columns.
    """
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"

    # Get site mapping
    site_map = {site["SiteName"]: site["Site_Id"]
                for site in requests.get(sites_url, headers=HEADERS).json()}

    # Target sites for PM2.5 and met
    target_sites = {
        "LIVERPOOL": site_map["LIVERPOOL"],
        "BRINGELLY": site_map["BRINGELLY"],
        "CAMPBELLTOWN_WEST": site_map["CAMPBELLTOWN WEST"],
        "CAMDEN": site_map["CAMDEN"]
    }

    site_dfs = []
    for site_name, site_id in target_sites.items():
        # Liverpool needs both PM2.5 + met, others only PM2.5
        parameters = ["PM2.5"] + (["WDR", "WSP", "TEMP", "HUMID"] if site_name == "LIVERPOOL" else [])

        payload = {
            "Sites": [site_id],
            "Parameters": parameters,
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

            # Check if the response contains data
            if not data:
                print(f"No data returned for {site_name}.")
                continue

        except Exception as e:
            print(f"Error fetching data for {site_name}: {e}")
            continue

        # Check if data contains records
        records = []
        for rec in data:
            if rec["Value"] is None:
                continue
            dt = datetime.strptime(rec["Date"], "%Y-%m-%d") + timedelta(hours=rec["Hour"])
            param_code = rec["Parameter"]["ParameterCode"]
            key = f"{param_code}_{site_name}"
            records.append({"datetime": dt, key: rec["Value"]})

        if records:
            df_site = pd.DataFrame(records).groupby("datetime").first()
            site_dfs.append(df_site)
        else:
            print(f"No valid records for {site_name} in the date range {start_date_str} to {end_date_str}.")
            continue

    if not site_dfs:
        print("No valid data fetched. Exiting.")
        return pd.DataFrame()

    # Merge all site dataframes on datetime
    df_api = pd.concat(site_dfs, axis=1).sort_index()

    # Check if wind data exists for Liverpool
    if "WSP_LIVERPOOL" in df_api.columns and "WDR_LIVERPOOL" in df_api.columns:
        df_api["WDR_LIVERPOOL_rad"] = np.deg2rad(df_api["WDR_LIVERPOOL"])
        df_api["LIV_U"] = df_api["WSP_LIVERPOOL"] * np.sin(df_api["WDR_LIVERPOOL_rad"])
        df_api["LIV_V"] = df_api["WSP_LIVERPOOL"] * np.cos(df_api["WDR_LIVERPOOL_rad"])
    else:
        print(f"Wind data (WSP, WDR) missing for LIVERPOOL. Skipping wind component conversion.")

    # Rename met vars to model feature names
    rename_map = {
        "TEMP_LIVERPOOL": "LIV_Temp",
        "HUMID_LIVERPOOL": "LIV_RH"
    }
    df_api = df_api.rename(columns=rename_map)

    # Ensure all required columns exist
    required_columns = [
        "LIV_Temp", "LIV_RH", "LIV_U", "LIV_V",
        "PM2.5_LIVERPOOL", "PM2.5_BRINGELLY", "PM2.5_CAMDEN", "PM2.5_CAMPBELLTOWN_WEST"
    ]
    for col in required_columns:
        if col not in df_api.columns:
            print(f"Warning: Missing column {col}. Adding empty column.")
            df_api[col] = np.nan

    # Drop rows with any missing data in required columns
    df_api.dropna(subset=required_columns, inplace=True)

    # Check if any data remains after cleaning
    if df_api.empty:
        print(f"No valid data after cleaning for the date range {start_date_str} to {end_date_str}.")
    
    return df_api

def preprocess_data(df_recent, scaler, feature_columns, n_input=72, n_output=6, n_subseq=6):
    """
    Preprocess the data: scale, create sequences and reshape for model input.
    Includes all features (PM2.5 and Liverpool met) used in training.
    """
    # Check that all required columns are present
    missing_cols = [col for col in feature_columns if col not in df_recent.columns]
    if missing_cols:
        print(f"Warning: Missing columns in input data: {missing_cols}")
        return None, None

    # Ensure the column order matches the scaler training
    df_processed = df_recent[feature_columns].copy()

    # Scale using pre-trained scaler
    df_scaled = scaler.transform(df_processed)

    # Create input-output sequences
    X, y = create_sequences(df_scaled, n_input, n_output)

    # Reshape input for CNN-LSTM: (samples, subseq, steps per subseq, features)
    n_steps_per_subseq = n_input // n_subseq
    X = X.reshape((X.shape[0], n_subseq, n_steps_per_subseq, len(feature_columns)))

    return X, y

pm25_columns = [c for c in feature_columns if "PM2.5" in c]
pm25_indices = [feature_columns.index(c) for c in pm25_columns]

def create_sequences(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i + n_input])
        y.append(data[i + n_input:i + n_input + n_output, pm25_indices])
        #y.append(data[i + n_input:i + n_input + n_output, :4])  # First 4 features assumed to be PM2.5
    return np.array(X), np.array(y)

def run_forecast(start_date_str, end_date_str, scaler, feature_columns):
    # Step 1: Fetch recent data from API
    df_recent = fetch_recent_data(start_date_str, end_date_str)

    if df_recent.empty:
        print("No valid data fetched. Exiting.")
        return

    # Step 2: Preprocess data for input to the model
    X_processed, y_processed = preprocess_data(df_recent, scaler, feature_columns)

    if X_processed is None or y_processed is None:
        print("Data preprocessing failed. Exiting.")
        return

    # Step 3: Load the trained model
    model = load_model("cnn_lstm_pm25_met_72input.h5")

    # Step 4: Make predictions
    predictions = model.predict(X_processed)

    # Step 5: Post-process the predictions (inverse scaling, etc.)
    y_pred_flat = predictions.reshape(-1, len(feature_columns))

    # Reconstruct full feature matrix for inverse scaling
    y_pred_padded = np.zeros((y_pred_flat.shape[0], len(feature_columns)))
    for i, idx in enumerate(pm25_indices):
        y_pred_padded[:, idx] = y_pred_flat[:, i]

    # Apply inverse scaling only to the PM2.5 columns
    y_pred_inv = scaler.inverse_transform(y_pred_padded)

    # Retain only the PM2.5 columns after inverse scaling
    y_pred_inv = y_pred_inv[:, :len(pm25_columns)]  # Only the PM2.5 columns

    # Step 6: Save and print the results
    results_df = pd.DataFrame({
        f"{c}_pred": y_pred_inv[:, i] for i, c in enumerate(pm25_columns)
    })

    results_df.to_csv("cnn_lstm_predictions.csv", index=False)
    print("Predictions saved to 'cnn_lstm_predictions.csv'")

    return results_df

# --------------------------
# Get data and forecast
# --------------------------
df_recent = fetch_recent_data(args.start, args.end)

if len(df_recent) < 24:
    raise ValueError("❌ Not enough data for forecasting. At least 24 hours are required.")

# Example call for a forecast
df_predictions = run_forecast(args.start, args.end, scaler, feature_columns)

start_date_str = args.start
end_date_str = args.end

def plot_predictions_and_actual(df_predictions, df_input, start_date_str, end_date_str, actual_pm25_data):
    """
    Plots the time series of predicted and actual PM2.5 along with the input features (WSP, Temp, RH) from Liverpool.
    """
    # Align time range for predictions (based on the number of predictions)
    time_range_predictions = pd.date_range(start=start_date_str, periods=len(df_predictions), freq='H')

    # Align time range for input features (based on the length of df_input)
    time_range_input = pd.date_range(start=start_date_str, periods=len(df_input), freq='H')

    # Create subplots for the predictions and input features
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

    # Plot Predicted and Actual PM2.5 for each site
    axes[0].plot(time_range_predictions, df_predictions["PM2.5_LIVERPOOL_pred"], label='Predicted PM2.5 (Liverpool)', color='tab:blue', linewidth=2)
    axes[0].plot(time_range_input, actual_pm25_data["PM2.5_LIVERPOOL"], label='Actual PM2.5 (Liverpool)', color='tab:cyan', linestyle='--', linewidth=2)
    axes[0].set_title('Predicted vs Actual PM2.5 at Liverpool')
    axes[0].set_ylabel('PM2.5 (µg/m³)')
    axes[0].legend(loc='upper left')

    # Plot Predicted and Actual PM2.5 at other sites
    axes[1].plot(time_range_predictions, df_predictions["PM2.5_BRINGELLY_pred"], label='Predicted PM2.5 (Bringelly)', color='tab:green', linewidth=2)
    axes[1].plot(time_range_input, actual_pm25_data["PM2.5_BRINGELLY"], label='Actual PM2.5 (Bringelly)', color='lime', linestyle='--', linewidth=2)
    axes[1].set_title('Predicted vs Actual PM2.5 at Bringelly')
    axes[1].set_ylabel('PM2.5 (µg/m³)')
    axes[1].legend(loc='upper left')

    axes[2].plot(time_range_predictions, df_predictions["PM2.5_CAMPBELLTOWN_WEST_pred"], label='Predicted PM2.5 (Campbelltown West)', color='tab:red', linewidth=2)
    axes[2].plot(time_range_input, actual_pm25_data["PM2.5_CAMPBELLTOWN_WEST"], label='Actual PM2.5 (Campbelltown West)', color='salmon', linestyle='--', linewidth=2)
    axes[2].set_title('Predicted vs Actual PM2.5 at Campbelltown West')
    axes[2].set_ylabel('PM2.5 (µg/m³)')
    axes[2].legend(loc='upper left')

    axes[3].plot(time_range_predictions, df_predictions["PM2.5_CAMDEN_pred"], label='Predicted PM2.5 (Camden)', color='tab:orange', linewidth=2)
    axes[3].plot(time_range_input, actual_pm25_data["PM2.5_CAMDEN"], label='Actual PM2.5 (Camden)', color='gold', linestyle='--', linewidth=2)
    axes[3].set_title('Predicted vs Actual PM2.5 at Camden')
    axes[3].set_ylabel('PM2.5 (µg/m³)')
    axes[3].legend(loc='upper left')

    # Plot Input Features (WSP, Temp, and RH for Liverpool)
    axes[4].plot(time_range_input, df_input["WSP_LIVERPOOL"][:len(time_range_input)], label="Wind Speed (WSP)", color='tab:purple', linestyle='--', linewidth=2)
    axes[4].set_title('Wind Speed (WSP), Temperature (TEMP), and RH at Liverpool')
    axes[4].set_ylabel('WSP (m/s), Temp (°C), RH (%)')
    axes[4].legend(loc='upper left')

    axes[4].plot(time_range_input, df_input["LIV_Temp"][:len(time_range_input)], label="Temperature (TEMP)", color='cyan', linestyle='-', linewidth=2)
    axes[4].plot(time_range_input, df_input["LIV_RH"][:len(time_range_input)], label="Relative Humidity (RH)", color='pink', linestyle='-.', linewidth=2)

    # Set the x-axis labels and grid
    axes[4].set_xlabel('Time')
    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming actual_pm25_data contains actual PM2.5 values for each site over the forecast period.
# This could be fetched or loaded from a separate data source.

# Create a dictionary for actual PM2.5 values (to be used in the plot)
actual_pm25_data = {
    "PM2.5_LIVERPOOL": df_recent["PM2.5_LIVERPOOL"][:len(df_predictions)],
    "PM2.5_BRINGELLY": df_recent["PM2.5_BRINGELLY"][:len(df_predictions)],
    "PM2.5_CAMPBELLTOWN_WEST": df_recent["PM2.5_CAMPBELLTOWN_WEST"][:len(df_predictions)],
    "PM2.5_CAMDEN": df_recent["PM2.5_CAMDEN"][:len(df_predictions)],
}

# Now call the plot function to visualize both predictions and actual values
plot_predictions_and_actual(df_predictions, df_recent, start_date_str, end_date_str, actual_pm25_data)

