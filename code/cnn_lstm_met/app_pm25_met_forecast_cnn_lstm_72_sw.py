import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st

# --------------------------
# API Data Fetch Function
# --------------------------
import requests

def get_available_parameters():
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        params = response.json()
        
        # Print the available parameters to check which ones correspond to wind components
        for param in params:
            print(f"Parameter Name: {param.get('ParameterName')}, Code: {param.get('ParameterCode')}")
        
    except Exception as e:
        print(f"Error fetching parameter details: {e}")

# Call the function to print out available parameters
get_available_parameters()

def get_forecast_from_api(start_date_str, end_date_str):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    # Fetch site details
    site_map = {site["SiteName"]: site["Site_Id"]
                for site in requests.get(sites_url, headers=HEADERS).json()}
    
    # Fetch parameter details (PM2.5)
    param_map = {}
    for param in requests.get(params_url, headers=HEADERS).json():
        code = param.get("ParameterCode")
        freq = param.get("Frequency", "").lower()
        if code and "hour" in freq:
            param_map[code] = param.get("ParameterCode")

    # Define target sites
    target_sites = {
        "BRINGELLY": site_map["BRINGELLY"],
        "CAMPBELLTOWN_WEST": site_map["CAMPBELLTOWN WEST"],
        "CAMDEN": site_map["CAMDEN"],  # Use CAMDEN instead of BARGO
        "LIVERPOOL": site_map["LIVERPOOL"]
    }

    parameter_id = param_map["PM2.5"]  # ID for PM2.5 parameter
    site_dfs = []
    present_sites = []

    # Fetch PM2.5 data for all target sites
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

    df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()

    # Handle missing sites by filling with mean
    missing_sites = [s for s in target_sites if s not in present_sites]
    for site in missing_sites:
        col_name = f"PM2.5_{site}"
        df_api[col_name] = df_api.mean(axis=1)
        st.info(f"Interpolated missing site {site} with mean.")

    df_api = df_api.dropna()

    # --------------------------
    # Fetch Meteorological Data for Liverpool (with correct parameters)
    # --------------------------
    met_feats = {
        "TEMP": "Temperature",
        "WSP": "Wind Speed",
        "WDR": "Wind Direction"
    }
    met_records = []
    for feat, param_code in met_feats.items():
        payload = {
            "Sites": [site_map["LIVERPOOL"]],
            "Parameters": [param_map.get(param_code, None)],  # Use correct parameter codes
            "StartDate": start_date_str,
            "EndDate": end_date_str,
            "Categories": ["Averages"],
            "SubCategories": ["Hourly"],
            "Frequency": ["Hourly average"]
        }
        try:
            resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            for r in resp.json():
                if r["Value"] is not None:
                    dt = datetime.strptime(r["Date"], "%Y-%m-%d") + timedelta(hours=r["Hour"])
                    met_records.append({"datetime": dt, feat: r["Value"]})
        except Exception as e:
            st.warning(f"Error fetching meteorological data for Liverpool: {e}")

    met_df = pd.DataFrame(met_records).groupby("datetime").first()

    # --------------------------
    # Convert Wind Speed and Wind Direction to U and V Components
    # --------------------------
    # Ensure that we have the data required to compute U and V components
    if "WSP" in met_df.columns and "WDR" in met_df.columns:
        # Convert wind direction from degrees to radians
        wind_direction_rad = np.radians(met_df["WDR"])
        
        # Calculate U (East-West) and V (North-South) components
        met_df["U"] = -met_df["WSP"] * np.sin(wind_direction_rad)
        met_df["V"] = -met_df["WSP"] * np.cos(wind_direction_rad)

        st.info("🌀 Wind Speed and Wind Direction converted to U and V components.")

    # Combine PM2.5 and meteorological data into a single DataFrame
    df_combined = pd.concat([df_api, met_df], axis=1).dropna()
    return df_combined

