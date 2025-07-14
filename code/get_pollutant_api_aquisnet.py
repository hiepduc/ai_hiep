import streamlit as st
import os
from datetime import datetime, timedelta
from PIL import Image
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

from streamlit_folium import folium_static
import io
import tempfile
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import urllib.parse

import matplotlib.dates as mdates
import time

# ------------------------
# Sidebar: Parameter & Date Selection
# ------------------------
st.sidebar.markdown("### API Aquisnet Parameter & Date Selection")

# Parameter and date selection
parameter = st.sidebar.selectbox("Select Parameter", ["OZONE", "NO", "NO2", "PM10", "PM2.5", "TEMP"])
start_date = st.sidebar.date_input("Start Date", datetime(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 1, 7))

if start_date > end_date:
    st.sidebar.error("End Date must be after Start Date")
    st.stop()

# API configuration
API_URL = "https://data.airquality.nsw.gov.au/api/Data/get_Observations"
HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}

# Helper function to check data existence
def parameter_exists_api(site_id, parameter_id, start_date, end_date):
    payload = {
        #"Sites": [site_id],
        #"Parameters": [parameter_id],
        "Sites": site_id,    # should be inside [] but deliberately make it wrong to get default resutls
        "Parameters": parameter_id,    # should be inside [] but deliberately make it wrong to get default resutls
        "StartDate": start_date.strftime("%Y-%m-%d"),
        "EndDate": end_date.strftime("%Y-%m-%d"),
        "Categories": ["Averages"],
        "SubCategories": ["Hourly"],
        "Frequency": ["Hourly average"]
    }
    try:
        with st.spinner("Please wait, fetching sites... Once finished. site is available to select"):
            # Simulate a slow API call
            #time.sleep(5)  # Replace this with your real API request

            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return len(data) > 0
        st.success("Data loaded successfully!")

    except Exception as e:
        st.warning(f"API error for site ID {site_id}: {e}")
    return False


# Load available sites and parameter IDs
def load_sites_and_params():
    sites_url = "https://data.airquality.nsw.gov.au/api/Data/get_SiteDetails"
    params_url = "https://data.airquality.nsw.gov.au/api/Data/get_ParameterDetails"

    sites = requests.get(sites_url, headers=HEADERS).json()
    params = requests.get(params_url, headers=HEADERS).json()

    # Map: Site name -> Site ID
    site_map = {site["SiteName"]: site["Site_Id"] for site in sites}

    # Map: ParameterCode -> first valid ParameterId (prefer hourly average)
    param_map = {}
    for param in params:
        code = param.get("ParameterCode")
        freq = param.get("Frequency", "").lower()
        if code and code not in param_map:
            if "hour" in freq:  # prefer hourly
                param_map[code] = param.get("ParameterCode")

    return site_map, param_map

site_map, param_map = load_sites_and_params()

# Get the parameter ID from name
parameter_id = param_map.get(parameter)
if parameter_id is None:
    st.error(f"Parameter '{parameter}' not found in API.")
    st.stop()

# Check which sites have the parameter data
available_sites = [
    site_name for site_name, site_id in site_map.items()
    if parameter_exists_api(site_id, parameter_id, start_date, end_date)
]

if not available_sites:
    st.warning(f"No data found for {parameter} between {start_date} and {end_date} at any site.")
    st.stop()

# Let user select from available sites
selected_site = st.sidebar.selectbox("Select Site", available_sites)

# Get the selected site ID
selected_site_id = site_map[selected_site]

#st.write(f"Sending Site_Id (type {type(selected_site_id)}): {selected_site_id}")

# Prepare the API request payload with correct fields
payload = {
    "Sites": [selected_site_id],
    "Parameters": [parameter_id],
    "StartDate": start_date.strftime("%Y-%m-%d"),
    "EndDate": end_date.strftime("%Y-%m-%d"),
    "Categories": ["Averages"],
    "SubCategories": ["Hourly"],
    "Frequency": ["Hourly average"]
}

#st.write(f"Selected site ID: {selected_site_id}, Parameter ID: {parameter_id}")
#st.subheader("Payload sent to API:")
#st.json(payload)

# Fetch data from the API
try:
    with st.spinner("Please wait, fetching data..."):
        # Simulate a slow API call
        #time.sleep(5)  # Replace this with your real API request

        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

records = []
units = None  # store the units

for rec in data:
    # Convert everything to correct type before comparison
    rec_site_id = int(rec.get("Site_Id", -1))
    rec_param_code = str(rec.get("Parameter", {}).get("ParameterCode", "")).upper()

    if rec_site_id == selected_site_id and rec_param_code == parameter_id.upper():
        date_str = rec["Date"]
        hour = rec["Hour"]
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=hour)
        value = rec["Value"]

        if value is not None:
            if units is None:
                units = rec["Parameter"].get("Units", "")
            records.append({"datetime": dt, "value": value})

#st.write(f"Total records returned by API: {len(data)}")
st.write(f"selected_site_id: {selected_site_id} ({type(selected_site_id)})")
st.write(f"parameter_id: {parameter_id} ({type(parameter_id)})")

df = pd.DataFrame(records)

# Assuming df contains 'datetime' and 'value' columns
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["datetime"], df["value"], marker="o", linestyle="-", label=parameter)

# Set axis titles
ax.set_title(f"{parameter} Time Series at {selected_site}", fontsize=14)
ax.set_xlabel("Datetime")
ax.set_ylabel(f"{parameter} ({units})")
#ax.set_ylabel(f"{parameter} ({rec['Parameter']['Units']})")
ax.grid(True)
ax.legend()

# Format x-axis to show hourly ticks or auto-adjust
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

# Rotate x-tick labels for readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show in Streamlit
st.pyplot(fig)

# CSV downloads
st.download_button("Download CSV", data=df.to_csv(index=False), file_name=f"{selected_site}_{parameter}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")


