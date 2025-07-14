import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Load the trained CNN-LSTM model and scaler
model = load_model('cnn_lstm_pm25_met_72input.h5')
scaler = joblib.load("pm25_scaler.save")
feature_columns = joblib.load("pm25_columns.save")

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

    y_pred_inv = scaler.inverse_transform(y_pred_padded)

    #y_pred_padded = np.hstack([y_pred_flat, np.zeros((y_pred_flat.shape[0], len(feature_columns) - y_pred_flat.shape[1]))])
    
    #y_pred_inv = scaler.inverse_transform(y_pred_padded)[:, :4]  # Only the PM2.5 columns
    
    # Step 6: Save and print the results
    results_df = pd.DataFrame({
        f"{c}_pred": y_pred_inv[:, i] for i, c in enumerate(pm25_columns)
    })

    #results_df = pd.DataFrame({
    #    "PM2.5_LIVERPOOL_pred": y_pred_inv[:, 0],
    #    "PM2.5_BRINGELLY_pred": y_pred_inv[:, 1],
    #    "PM2.5_CAMPBELLTOWN_WEST_pred": y_pred_inv[:, 2],
    #    "PM2.5_CAMDEN_pred": y_pred_inv[:, 3],
    #})
    
    results_df.to_csv("cnn_lstm_predictions.csv", index=False)
    print("Predictions saved to 'cnn_lstm_predictions.csv'")

# Example call for a forecast
start_date_str = "2024-12-01"
end_date_str = "2024-12-07"
run_forecast(start_date_str, end_date_str, scaler, feature_columns)

