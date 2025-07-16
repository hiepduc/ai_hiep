
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from utils.forecasting_utils import create_sequences
import plotly.express as px
import geopandas as gpd

st.sidebar.title("PM2.5 Forecast Dashboard")
region = st.sidebar.selectbox("Select Region", ["SW Sydney", "NW Sydney", "Upper Hunter"])
input_hours = st.sidebar.selectbox("Input Hours for Model", [72, 120])
forecast_date = st.sidebar.date_input("Forecast Start Date", datetime.date(2020, 1, 1))

region_key = {"SW Sydney": "SW", "NW Sydney": "NW", "Upper Hunter": "UH"}
region_id = region_key[region]

data_path = f"data/Imputed_data_{region_id}_PM2.5.csv"
df = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")
stations = df.columns.tolist()

start_time = pd.to_datetime(forecast_date) - pd.Timedelta(hours=input_hours)
input_df = df.loc[start_time: pd.to_datetime(forecast_date) - pd.Timedelta(hours=1)]

model_path = f"models/cnn_lstm_pm25_{region_id.lower()}_{input_hours}.h5"
scaler_path = f"scalers/{region_id.lower()}_scaler_{input_hours}.save"
model = load_model(model_path)
scaler = joblib.load(scaler_path)

X_input, _ = create_sequences(scaler.transform(input_df), input_hours, 6)
X_input = X_input[-1:]
y_pred_scaled = model.predict(X_input)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, len(stations)))

st.title(f"PM2.5 Forecast for {region} on {forecast_date}")
for i, site in enumerate(stations):
    st.subheader(f"{site}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_pred[:, i], x=pd.date_range(forecast_date, periods=6, freq="H"),
                             mode='lines+markers', name='Predicted'))
    st.plotly_chart(fig)

st.subheader("Station Map")
gdf = gpd.read_file("stations.geojson")
gdf_region = gdf[gdf["Region"] == region]
fig_map = px.scatter_mapbox(
    gdf_region, lat="lat", lon="lon", text="Station", zoom=8, height=400
)
fig_map.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig_map)
