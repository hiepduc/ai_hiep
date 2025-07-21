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
    "MERRIWA": {"lat": -32.139, "lon": 150.356},
    "ALBION_PARK_SOUTH": {"lat": -34.567, "lon": 150.80},
    "KEMBLA_GRANGE": {"lat": -34.47, "lon": 150.796},
    "WOLLONGONG": {"lat": -34.424, "lon": 150.893}
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
        "CE": ["EARLWOOD", "MACQUARIE PARK", "RANDWICK", "ROZELLE"],
        "NW": ["PARRAMATTA NORTH", "RICHMOND", "ROUSE HILL"],
        "UH": ["MUSWELLBROOK", "SINGLETON", "MERRIWA"],
        "LH": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
        "ILLAWARRA": ["ALBION PARK SOUTH", "KEMBLA GRANGE", "WOLLONGONG"]
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

def preprocess_time_index(df, freq='H'):
    df = df.copy()
    df = df.sort_index()
    # Create complete time index
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    # Reindex to ensure all expected times are present
    df = df.reindex(full_range)
    # Interpolate or fill missing data
    df = df.interpolate(method='time')  # You can use ffill or bfill instead
    return df

def pad_to_n_input(array, n_input):
    n_missing = n_input - array.shape[0]
    if n_missing > 0:
        padding = np.tile(array[0], (n_missing, 1))  # or use np.zeros
        array = np.vstack([padding, array])
    return array


# ------------------------------
# Forecast Function
# ------------------------------
def run_forecast(df, model, scaler, feature_columns, n_input, forecast_hours, model_type):
    df_input = df[feature_columns].copy()

    if len(df_input) < n_input:
        pad_len = n_input - len(df_input)
        padding_df = pd.DataFrame([df_input.iloc[0].values] * pad_len, columns=feature_columns)
        df_input = pd.concat([padding_df, df_input])

    df_input = df_input[-n_input:]
    scaled_input = scaler.transform(df_input)

    predictions_scaled = []
    n_total_features = scaled_input.shape[1]

    for _ in range(forecast_hours):
        # Prepare input sample
        n_subseq = 6     # number of subsequences
        n_steps = n_input // n_subseq  # length of each subsequence
        if model_type.lower() == "vmd_cnn_lstm":
            try:
                if n_input % n_subseq != 0:
                    raise ValueError(f"n_input ({n_input}) must be divisible by n_subseq ({n_subseq})")

                X = scaled_input.reshape((1, n_input, n_total_features))
            except Exception as e:
                st.error(f"Reshape failed: {e}")
                return None
        else:
            X = scaled_input.reshape((1, n_subseq, n_steps, n_total_features))


        y_pred_scaled = model.predict(X)

        # If model outputs 2D array: (1, n_features), squeeze it
        if len(y_pred_scaled.shape) == 3:
            y_pred_scaled = y_pred_scaled[:, -1, :]  # (1, 1, n_features) → (1, n_features)
        y_pred_scaled = y_pred_scaled.squeeze()

        # Store prediction
        predictions_scaled.append(y_pred_scaled)

        # Update input with autoregressive step
        scaled_input = np.vstack([scaled_input, y_pred_scaled.reshape(1, -1)])
        scaled_input = scaled_input[-n_input:]

    # Combine and inverse transform
    forecast_array = np.array(predictions_scaled)  # shape: (forecast_hours, n_features)
    forecast_inverse = scaler.inverse_transform(forecast_array)

    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1),
                                   periods=forecast_hours, freq="H")

    return pd.DataFrame(forecast_inverse, index=forecast_index, columns=feature_columns)


# ------------------------------
# App Layout
# ------------------------------
st.set_page_config(layout="wide")
st.title("🌫️ PM2.5 Forecast Using Deep Learning (DL) Dashboard")

main_col, map_col = st.columns([5, 1])

# --- Station Map ---
with map_col:
    st.markdown("### 🧭 Station Map")
    fmap = folium.Map(location=[-33.9, 151.0], zoom_start=8)
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
    model_type = st.selectbox("Model Type", ["CNN_LSTM", "CNN_LSTM_BNN (coming)", "VMD_CNN_LSTM", "VMD_CNN_LSTM_ATTENTION"])
    regions_selected = st.multiselect("Select Regions", ["SW", "NW", "CE", "UH","LH","ILLAWARRA"], default=["SW"])
    n_input = st.radio("Model Input Hours", [72, 120], horizontal=True)
    forecast_days = st.slider("Forecast Horizon (days)", 1, 7, 3)
    start_date = st.date_input("Start Date", datetime(2025, 7, 1))
    end_date = st.date_input("End Date", datetime(2025, 7, 10))

# --- Main Forecast Logic ---
forecast_hours = forecast_days * 24
#all_forecasts = []
#all_observed = []
#all_errors = []

with main_col:
    if st.button("🚀 Run Forecast"):
        for region in regions_selected:
            all_forecasts = []
            all_observed = []
            all_errors = []
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
                elif model_type == "VMD_CNN_LSTM":
                    model = load_model(f"models/cnn_lstm_vmd_model_pm25_{region.lower()}_{n_input}input.h5")
                    scaler = joblib.load(f"scalers/pm25_vmd_{region.lower()}_scaler_{n_input}.save")
                    features = joblib.load(f"scalers/pm25_vmd_{region.lower()}_columns_{n_input}.save")
                else:
                    st.info(f"Model type '{model_type}' is a placeholder.")
                    continue
            except Exception as e:
                st.error(f"Missing model/scaler/columns for {region}: {e}")
                continue

            from vmdpy import VMD

            @st.cache_data(show_spinner="Applying VMD to observations...")
            def apply_vmd_to_df(df, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7):
                """
                Applies VMD to each column in the DataFrame and adds IMF columns.
                Ensures IMF outputs match the DataFrame index length by padding/trimming.
                Returns a DataFrame with aligned IMF columns only.
                """
                df_vmd = pd.DataFrame(index=df.index)

                for col in df.columns:
                    try:
                        signal = df[col].values.astype(float)

                        # Skip if any NaNs
                        if np.isnan(signal).any():
                            st.warning(f"Skipping VMD for {col} due to NaNs.")
                            continue

                        # Apply VMD
                        u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

                        for k in range(K):
                            imf = u[k]
                            if len(imf) < len(df):
                                imf = np.pad(imf, (0, len(df) - len(imf)), mode='edge')
                            elif len(imf) > len(df):
                                imf = imf[:len(df)]

                            df_vmd[f"{col}_IMF{k+1}"] = imf

                    except Exception as e:
                        st.warning(f"VMD failed for {col}: {e}")
                        continue

                return df_vmd.dropna()

            if model_type == "CNN_LSTM":
                df_obs = df_obs[features].dropna()
                forecast_df = run_forecast(df_obs, model, scaler, features, n_input, forecast_hours, model_type)
                forecast_df["Region"] = region
                all_forecasts.append(forecast_df)
                all_observed.append(df_obs)

                # Get actual values during forecast period
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

                # Individual plots
                st.markdown("### 📈 Forecast vs Observed (per site)")
                for col in features:
                    # Create full time index covering both history and forecast
                    full_index = pd.date_range(df_obs.index.min(), forecast_df.index.max(), freq='H')

                    df_combined = pd.DataFrame(index=full_index)
                    df_combined["Observed"] = df_obs[col].reindex(full_index)
                    df_combined["Forecast"] = forecast_df[col].reindex(full_index)
                    df_combined["Observed Future"] = actual_future[col].reindex(full_index)

                    df_plot = df_combined.reset_index().melt(id_vars="index", var_name="Type", value_name="PM2.5")
                    df_plot.rename(columns={"index": "datetime"}, inplace=True)

                    fig = px.line(df_plot, x="datetime", y="PM2.5", color="Type",
                                  title=col.replace("_", " "), markers=True)
                    st.plotly_chart(fig, use_container_width=True)
#########

                # Combined Graph
                if all_forecasts:
                    st.markdown("### 🌍 Combined Forecast and Observed")
                    combined_df = pd.concat(all_forecasts)
                    combined_obs = pd.concat(all_observed)

                    # Combine historical, forecast, and observed future
                    combined_future_obs = []
                    for forecast_df, df_obs in zip(all_forecasts, all_observed):
                        region_sites = forecast_df.columns.drop("Region")
                        future_start = forecast_df.index[0]
                        future_end = forecast_df.index[-1]
                        region = forecast_df["Region"].iloc[0]

                        # Fetch actual future observations again (safer for reindexing)
                        df_future_obs = fetch_pm25_data(future_start.strftime("%Y-%m-%d"),
                                                         future_end.strftime("%Y-%m-%d"), region)
                        df_future_obs = df_future_obs[region_sites].reindex(forecast_df.index)
                        combined_future_obs.append(df_future_obs)

                    combined_obs_long = pd.concat(all_observed).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Observed")
                    combined_forecast_long = pd.concat([df.drop(columns="Region") for df in all_forecasts]).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Forecast")
                    combined_futureobs_long = pd.concat(combined_future_obs).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Observed Future")

                    all_long = pd.concat([combined_obs_long, combined_forecast_long, combined_futureobs_long])
                    all_long["datetime"] = all_long.index

                    fig = px.line(all_long, x="datetime", y="PM2.5", color="Site", line_dash="Type", title="Combined PM2.5 Forecast vs Observed + Future")
                    st.plotly_chart(fig, use_container_width=True)

                    # Download CSV
                    csv = combined_df.drop(columns="Region").to_csv().encode("utf-8")
                    st.download_button("📥 Download Forecast CSV", csv, file_name="pm25_forecast.csv", key=region)

                if all_errors:
                    st.markdown("### 🧪 Forecast Accuracy")
                    err_df = pd.DataFrame(all_errors)
                    st.dataframe(err_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))


#########
            if "VMD" in model_type:
                df_obs_raw = df_obs.copy()  # <-- Keep raw observations

                df_obs = apply_vmd_to_df(df_obs)
                df_obs = df_obs[features].dropna()

                #if model_type == "cnn_lstm":
                #    X = scaled_input.reshape((1, n_subseq, n_steps, scaled_input.shape[1]))
                #elif model_type == "vmd_cnn_lstm":
                #    X = scaled_input.reshape((1, n_input, scaled_input.shape[1]))

                df_obs = preprocess_time_index(df_obs)
                forecast_df = run_forecast(df_obs, model, scaler, features, n_input, forecast_hours, model_type)

                forecast_df["Region"] = region
                all_forecasts.append(forecast_df)

                # --- Load actual PM2.5 data for the forecast period ---
                actual_future = fetch_pm25_data(
                    (df_obs.index[-1] + timedelta(hours=1)).strftime("%Y-%m-%d"),
                    (df_obs.index[-1] + timedelta(hours=forecast_hours)).strftime("%Y-%m-%d"),
                    region
                )
                actual_future = actual_future.reindex(forecast_df.index)

                # --- Reconstruct PM2.5 for all sites ---
                site_base_names = sorted(set([col.split("_IMF")[0] for col in features]))

                #st.write("Available observed columns:", actual_future.columns.tolist())
                #st.write("Available site base names:", site_base_names)

                for site_base_name in site_base_names:
                    recon_col_name = f"{site_base_name}_Reconstructed"
                    imf_cols = [f"{site_base_name}_IMF{i+1}" for i in range(3)]

                    # --- Get Forecast Reconstructed PM2.5 ---
                    if all(col in forecast_df.columns for col in imf_cols):
                        forecast_recon = forecast_df[imf_cols].sum(axis=1).rename(recon_col_name)
                        forecast_df[recon_col_name] = forecast_recon
                    else:
                        st.warning(f"⚠️ Missing IMF forecast columns for {site_base_name}, skipping.")
                        continue

                    # --- Get Observed Historical PM2.5 from API ---
                    historical_obs = None
                    if site_base_name in df_obs_raw.columns:
                        historical_obs = df_obs_raw[[site_base_name]].dropna()
                        historical_obs.columns = ["Observed PM2.5"]
                    else:
                        st.warning(f"⚠️ Observation missing for {site_base_name}, skipping.")
                        continue

                    # --- Combine Historical Observed + Forecast Reconstructed ---
                    forecast_df_subset = forecast_df[[recon_col_name]].rename(columns={recon_col_name: "Reconstructed PM2.5"})
                    df_plot = pd.concat([historical_obs, forecast_df_subset])
                    df_plot = df_plot.reset_index().rename(columns={"index": "datetime"})
                    df_melted = df_plot.melt(id_vars="datetime", var_name="Type", value_name="PM2.5")

                    # --- Plot ---
                    fig = px.line(df_melted, x="datetime", y="PM2.5", color="Type",
                                  title=f"📈 Observed vs Forecast Reconstructed PM2.5: {site_base_name}")
                    st.plotly_chart(fig, use_container_width=True, key=f"obs_vs_recon_{site_base_name}")

                    # --- Forecast Accuracy Metrics ---
                    if site_base_name in actual_future.columns:
                        forecast_range = forecast_df.index
                        actual_trimmed = actual_future[[site_base_name]].reindex(forecast_range).dropna()
                        recon_trimmed = forecast_recon.reindex(actual_trimmed.index)

                        if not actual_trimmed.empty and not recon_trimmed.empty:
                            mae = mean_absolute_error(actual_trimmed, recon_trimmed)
                            rmse = mean_squared_error(actual_trimmed, recon_trimmed, squared=False)
                            st.info(f"📏 {site_base_name} Forecast MAE: {mae:.2f}, RMSE: {rmse:.2f}")
#######

                all_observed.append(df_obs)

                # 🔁 Apply VMD to actual_future as well
                actual_future = apply_vmd_to_df(actual_future)

                # Extract only the IMF features
                actual_future = actual_future[features].reindex(forecast_df.index).dropna()
                common_index = forecast_df.index.intersection(actual_future.index)

                if not actual_future.empty:
                    for col in features:
                        mae = mean_absolute_error(actual_future[col], forecast_df.loc[common_index, col])
                        rmse = mean_squared_error(actual_future[col], forecast_df.loc[common_index, col], squared=False)
                        all_errors.append({"Region": region, "Site": col, "MAE": mae, "RMSE": rmse})

                # Individual plots
                st.markdown("### 📈 Forecast vs Observed (per site)")
                for col in features:
                    # Create full time index covering both history and forecast
                    full_index = pd.date_range(df_obs.index.min(), forecast_df.index.max(), freq='H')

                    df_combined = pd.DataFrame(index=full_index)
                    df_combined["Observed"] = df_obs[col].reindex(full_index)
                    df_combined["Forecast"] = forecast_df[col].reindex(full_index)
                    df_combined["Observed Future"] = actual_future[col].reindex(full_index)

                    df_plot = df_combined.reset_index().melt(id_vars="index", var_name="Type", value_name="PM2.5")
                    df_plot.rename(columns={"index": "datetime"}, inplace=True)

                    fig = px.line(df_plot, x="datetime", y="PM2.5", color="Type",
                                  title=col.replace("_", " "), markers=True)
                    st.plotly_chart(fig, use_container_width=True)

                # Combined Graph of IMFs vs Observed IMFs
                if all_forecasts:
                    st.markdown("### 🌍 Combined Forecast and Observed")
                    combined_df = pd.concat(all_forecasts)
                    combined_obs = pd.concat(all_observed)

                    # Combine historical, forecast, and observed future
                    combined_future_obs = []

                    import re

                    for forecast_df, df_obs in zip(all_forecasts, all_observed):
                        region_sites = forecast_df.columns.drop("Region")
                        future_start = forecast_df.index[0]
                        future_end = forecast_df.index[-1]
                        region = forecast_df["Region"].iloc[0]

                        # Fetch actual future observations again (safer for reindexing)
                        df_future_obs = fetch_pm25_data(future_start.strftime("%Y-%m-%d"),
                                             future_end.strftime("%Y-%m-%d"), region)

                        # Extract IMF columns and reconstruct base site names
                        imf_cols = [col for col in forecast_df.columns if re.search(r'_IMF\d+$', col)]
                        base_sites = sorted(set([re.sub(r'_IMF\d+$', '', col) for col in imf_cols]))

                        # Add reconstructed columns if they exist
                        reconstructed_sites = []
                        for base in base_sites:
                            recon_col = f"{base}_Reconstructed"
                            if recon_col in forecast_df.columns:
                                reconstructed_sites.append(recon_col)

                        # Convert reconstructed site names to match obs column names
                        obs_sites = [site.replace('_Reconstructed', '') for site in reconstructed_sites]

                        # Filter only columns that exist in df_future_obs
                        obs_sites_existing = [s for s in obs_sites if s in df_future_obs.columns]

                        # Warn if missing columns
                        missing_sites = list(set(obs_sites) - set(obs_sites_existing))
                        if missing_sites:
                            st.warning(f"⚠️ These site(s) not found in future obs: {missing_sites}")

                        # Proceed only with existing ones
                        df_future_obs = df_future_obs[obs_sites_existing].reindex(forecast_df.index)
    
                        # Rename obs columns to match forecast_df reconstructed column names
                        df_future_obs.columns = [f"{s}_Reconstructed" for s in obs_sites_existing]

                        combined_future_obs.append(df_future_obs)

                    combined_obs_long = pd.concat(all_observed).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Observed")
                    combined_forecast_long = pd.concat([df.drop(columns="Region") for df in all_forecasts]).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Forecast")
                    combined_futureobs_long = pd.concat(combined_future_obs).melt(ignore_index=False, var_name="Site", value_name="PM2.5").assign(Type="Observed Future")

                    all_long = pd.concat([combined_obs_long, combined_forecast_long, combined_futureobs_long])
                    all_long["datetime"] = all_long.index

                    fig = px.line(all_long, x="datetime", y="PM2.5", color="Site", line_dash="Type", title="Combined PM2.5 Forecast vs Observed + Future")
                    st.plotly_chart(fig, use_container_width=True)

                    for idx, forecast_df in enumerate(all_forecasts):
                        imf_cols = [col for col in forecast_df.columns if re.search(r'_IMF\d+$', col)]
                        base_sites = sorted(set([re.sub(r'_IMF\d+$', '', col) for col in imf_cols]))
    
                        # Reconstruct and insert missing '_Reconstructed' columns
                        for site in base_sites:
                            recon_col = f"{site}_Reconstructed"
                            site_imf_cols = [col for col in forecast_df.columns if col.startswith(site) and '_IMF' in col]
        
                            if recon_col not in forecast_df.columns:
                                if site_imf_cols:
                                    # Sum the IMFs to reconstruct
                                    forecast_df[recon_col] = forecast_df[site_imf_cols].sum(axis=1)
                                else:
                                    st.warning(f"⚠️ No IMF components found for site: {site} to reconstruct.")
    
                     # Update back to all_forecasts list
                     #all_forecasts[idx] = forecast_df


