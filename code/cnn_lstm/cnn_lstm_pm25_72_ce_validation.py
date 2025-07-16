import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, RepeatVector
from tensorflow.keras.layers import Reshape
import matplotlib.pyplot as plt
import joblib

# === PARAMETERS ===
INPUT_FILE = "/home/duch/ai_hiep/data/Imputed_data_CE_PM2.5.csv"
N_INPUT = 72     # Use 72 hours as input
N_OUTPUT = 6     # Forecast 6 hours ahead
N_SUBSEQ = 6     # Break 72 into 6 subsequences
EPOCHS = 30
BATCH_SIZE = 64

# === STEP 1: LOAD DATA ===
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# === STEP 2: SCALING ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
joblib.dump(scaler, "pm25_ce_scaler_72.save")
joblib.dump(df.columns.tolist(), "pm25_ce_columns_72.save")  # Save feature order
n_features = scaled_data.shape[1]

# === STEP 3: CREATE SEQUENCES ===
def create_sequences(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i+n_input])
        y.append(data[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, N_INPUT, N_OUTPUT)

# === STEP 4: RESHAPE FOR CNN-LSTM ===
n_steps_per_subseq = N_INPUT // N_SUBSEQ
X = X.reshape((X.shape[0], N_SUBSEQ, n_steps_per_subseq, n_features))

# === STEP 5: TRAIN / VAL / TEST SPLIT ===
n_total = len(X)
train_end = int(n_total * 0.7)
val_end = int(n_total * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X[val_end:], y[val_end:]

# === STEP 6: MODEL ===
model = Sequential([
    TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'),
                    input_shape=(N_SUBSEQ, n_steps_per_subseq, n_features)),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(100, activation='relu'),
    RepeatVector(N_OUTPUT),
    LSTM(100, activation='relu', return_sequences=True),
    TimeDistributed(Dense(n_features))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# === STEP 7: TRAIN MODEL ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)
model.save("cnn_lstm_model_pm25_ce_72input.h5")

# === STEP 8: EVALUATE ON TEST SET ===
y_pred_test = model.predict(X_test)

# Flatten sequences
y_test_flat = y_test.reshape(-1, n_features)
y_pred_flat = y_pred_test.reshape(-1, n_features)

# Inverse scaling
y_test_inv = scaler.inverse_transform(y_test_flat)
y_pred_inv = scaler.inverse_transform(y_pred_flat)

# Overall metrics
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = mean_squared_error(y_test_inv, y_pred_inv, squared=False)
print(f"\n📊 Test MAE: {mae:.2f} µg/m³")
print(f"📊 Test RMSE: {rmse:.2f} µg/m³")

# Save summary to CSV
summary_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})
summary_df.to_csv("test_summary_metrics.csv", index=False)

# Per-site metrics
site_names = df.columns.tolist()
site_mae_list = []
for i, site in enumerate(site_names):
    site_mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
    site_mae_list.append({"Site": site, "MAE": site_mae})
    print(f"{site:<30}: {site_mae:.2f}")

pd.DataFrame(site_mae_list).to_csv("test_site_mae.csv", index=False)

# === STEP 9: Save actual vs predicted as DataFrame
test_index = pd.date_range(start='2025-01-01', periods=len(y_test_inv), freq='H')  # Replace with real timestamps if known
df_actual = pd.DataFrame(y_test_inv, columns=[s + "_actual" for s in site_names], index=test_index)
df_pred = pd.DataFrame(y_pred_inv, columns=[s + "_predicted" for s in site_names], index=test_index)
df_combined = pd.concat([df_actual, df_pred], axis=1)
df_combined.to_csv("test_predictions_vs_actual.csv")

# === STEP 10: Dashboard-style interactive plot using Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sites_to_plot = ['PM2.5_LIVERPOOL', 'PM2.5_BARGO', 'PM2.5_BRINGELLY', 'PM2.5_CAMPBELLTOWN_WEST']

fig = make_subplots(rows=len(sites_to_plot), cols=1, shared_xaxes=True,
                    subplot_titles=[s.replace("PM2.5_", "") for s in sites_to_plot])

for i, site in enumerate(sites_to_plot):
    actual = df_actual[f"{site}_actual"]
    predicted = df_pred[f"{site}_predicted"]
    fig.add_trace(go.Scatter(x=test_index, y=actual, name=f"{site} Actual", mode='lines'), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=predicted, name=f"{site} Predicted", mode='lines'), row=i+1, col=1)

fig.update_layout(height=300*len(sites_to_plot), title_text="PM2.5 Forecast vs Actual on Test Set", showlegend=True)
fig.write_html("test_forecast_dashboard.html")
print("✅ Interactive dashboard saved to test_forecast_dashboard.html")


