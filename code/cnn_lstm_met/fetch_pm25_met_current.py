import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_recent_data(start_date_str, end_date_str):
    API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    site_map = {site["SiteName"]: site["Site_Id"]
                for site in requests.get(sites_url, headers=HEADERS).json()}

    param_map = {}
    for param in requests.get(params_url, headers=HEADERS).json():
        code = param.get("ParameterCode")
        if code:
            param_map[code] = param.get("ParameterCode")

    target_sites = {
        "CAMPBELLTOWN_WEST": site_map["CAMPBELLTOWN WEST"],
        "BRINGELLY": site_map["BRINGELLY"],
        "LIVERPOOL": site_map["LIVERPOOL"],
        "CAMDEN": site_map["CAMDEN"]
    }

    site_dfs = []
    present_sites = []
    for site_name, site_id in target_sites.items():
        payload = {
            "Sites": [site_id],
            "Parameters": ["WDR", "WSP", "TEMP", "PM2.5"],
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
            print(f"Error fetching data for {site_name}: {e}")
            continue

        records = []
        for rec in data:
            if rec["Value"] is not None:
                # Debug: Print the raw data
                #print(f"Raw data for {site_name}: {rec}")
                dt = datetime.strptime(rec["Date"], "%Y-%m-%d") + timedelta(hours=rec["Hour"])
                records.append({"datetime": dt, f"PM2.5_{site_name}": rec["Value"],
                                f"WDR_{site_name}": rec.get("WDR"),
                                f"WSP_{site_name}": rec.get("WSP"),
                                f"TEMP_{site_name}": rec.get("TEMP")})

        if records:
            df_site = pd.DataFrame(records)
            df_site = df_site.set_index("datetime")  # Set datetime as the index
            # Ensure unique datetime index
            df_site = df_site[~df_site.index.duplicated(keep='first')]  # Remove duplicates, keeping the first occurrence
            site_dfs.append(df_site)
            present_sites.append(site_name)

    if not site_dfs:
        return pd.DataFrame()  # Return empty DataFrame if no valid data found

    # Concatenate all dataframes on datetime index
    try:
        df_api = pd.concat(site_dfs, axis=1, join="outer").sort_index()
    except Exception as e:
        print(f"Error during concatenation or sorting: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

    # Ensure no missing datetime entries
    df_api = df_api.dropna(subset=["PM2.5_LIVERPOOL"])  # Drop rows with missing PM2.5 values

    # Calculate U and V wind components only if valid values for WDR and WSP exist
    for site_name in target_sites:
        wdr_column = f"WDR_{site_name}"
        wsp_column = f"WSP_{site_name}"

        # Check if both WDR and WSP have valid (non-None) values before calculating U and V components
        if wdr_column in df_api and wsp_column in df_api:
            # Replace None values with NaN for proper handling
            df_api[wdr_column] = pd.to_numeric(df_api[wdr_column], errors='coerce')
            df_api[wsp_column] = pd.to_numeric(df_api[wsp_column], errors='coerce')

            # Now safely perform the calculations for U and V components
            valid_data_mask = df_api[wdr_column].notna() & df_api[wsp_column].notna()
            df_api.loc[valid_data_mask, f"U_{site_name}"] = df_api.loc[valid_data_mask, wsp_column] * np.sin(np.deg2rad(df_api.loc[valid_data_mask, wdr_column]))
            df_api.loc[valid_data_mask, f"V_{site_name}"] = df_api.loc[valid_data_mask, wsp_column] * np.cos(np.deg2rad(df_api.loc[valid_data_mask, wdr_column]))

    return df_api

# Example usage
start_date_str = "2025-07-01"
end_date_str = "2025-07-07"
df_recent = fetch_recent_data(start_date_str, end_date_str)
print(df_recent.head(25))  # Inspect the data

