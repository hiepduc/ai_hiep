# cnn_lstm_pm25_met.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, RepeatVector
import matplotlib.pyplot as plt
import joblib
import os

# === PARAMETERS ===
INPUT_FILE = "/home/duch/ai_hiep/data/PM25_RH_T_Wind_SW_2018_2022.csv"
N_INPUT = 72     # 72 hours input
N_OUTPUT = 6     # 6-hour forecast
N_SUBSEQ = 6     # 6 subsequences of 12 hours
EPOCHS = 30
BATCH_SIZE = 64
MODEL_NAME = "cnn_lstm_pm25_met_72input.h5"

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'], dayfirst=True)
df = df.rename(columns={
    'BRINGELLY': 'PM2.5_BRINGELLY',
    'CAMPBELLTOWN_WEST': 'PM2.5_CAMPBELLTOWN_WEST',
    'CAMDEN': 'PM2.5_CAMDEN',
    'LIVERPOOL': 'PM2.5_LIVERPOOL'
})
df = df.set_index("datetime")
df = df.dropna()

# === FEATURE COLUMNS ===
feature_columns = [
    "PM2.5_LIVERPOOL",
    "PM2.5_BRINGELLY",
    "PM2.5_CAMPBELLTOWN_WEST",
    "PM2.5_CAMDEN",
    "LIV_Temp",
    "LIV_RH",
    "LIV_U",
    "LIV_V"
]

target_columns = [
    "PM2.5_LIVERPOOL",
    "PM2.5_BRINGELLY",
    "PM2.5_CAMPBELLTOWN_WEST",
    "PM2.5_CAMDEN"
]

# === SCALE DATA ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[feature_columns])
joblib.dump(scaler, "pm25_scaler.save")
joblib.dump(feature_columns, "pm25_columns.save")

# === CREATE SEQUENCES ===
def create_sequences(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i+n_input])
        y.append(data[i+n_input:i+n_input+n_output, :4])  # only PM2.5 for output
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, N_INPUT, N_OUTPUT)

# === RESHAPE INPUT FOR CNN-LSTM ===
n_steps_per_subseq = N_INPUT // N_SUBSEQ
n_features = X.shape[2]
X = X.reshape((X.shape[0], N_SUBSEQ, n_steps_per_subseq, n_features))

# === TRAIN/VAL SPLIT ===
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# === MODEL ===
model = Sequential([
    TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'),
                    input_shape=(N_SUBSEQ, n_steps_per_subseq, n_features)),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(100, activation='relu'),
    RepeatVector(N_OUTPUT),
    LSTM(100, activation='relu', return_sequences=True),
    TimeDistributed(Dense(len(target_columns)))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# === TRAIN ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

model.save(MODEL_NAME)

# === EVALUATE ===
y_pred = model.predict(X_val)
y_val_flat = y_val.reshape(-1, len(target_columns))
y_pred_flat = y_pred.reshape(-1, len(target_columns))

# Reconstruct full feature matrix for inverse scaling
pad_shape = (y_pred_flat.shape[0], len(feature_columns) - len(target_columns))
y_val_padded = np.hstack([y_val_flat, np.zeros(pad_shape)])
y_pred_padded = np.hstack([y_pred_flat, np.zeros(pad_shape)])

y_val_inv = scaler.inverse_transform(y_val_padded)[:, :4]
y_pred_inv = scaler.inverse_transform(y_pred_padded)[:, :4]

# Save predictions and actuals
results_df = pd.DataFrame({
    "PM2.5_LIVERPOOL_actual": y_val_inv[:, 0],
    "PM2.5_LIVERPOOL_pred": y_pred_inv[:, 0],
    "PM2.5_BRINGELLY_actual": y_val_inv[:, 1],
    "PM2.5_BRINGELLY_pred": y_pred_inv[:, 1],
    "PM2.5_CAMPBELLTOWN_WEST_actual": y_val_inv[:, 2],
    "PM2.5_CAMPBELLTOWN_WEST_pred": y_pred_inv[:, 2],
    "PM2.5_CAMDEN_actual": y_val_inv[:, 3],
    "PM2.5_CAMDEN_pred": y_pred_inv[:, 3],
})
results_df.to_csv("cnn_lstm_predictions.csv", index=False)

# === MAE ===
print("\nMAE per site (µg/m³):")
mae_dict = {}
for i, site in enumerate(target_columns):
    mae_site = mean_absolute_error(y_val_inv[:, i], y_pred_inv[:, i])
    mae_dict[site] = mae_site
    print(f"{site:<30}: {mae_site:.2f}")

# Save MAE to CSV
mae_df = pd.DataFrame.from_dict(mae_dict, orient='index', columns=["MAE"])
mae_df.to_csv("cnn_lstm_mae.csv")

# === PLOT ===
for site in target_columns:
    col_actual = f"{site}_actual"
    col_pred = f"{site}_pred"
    plt.figure(figsize=(10, 5))
    plt.plot(results_df[col_actual].values[:100], label="Actual")
    plt.plot(results_df[col_pred].values[:100], label="Predicted")
    plt.title(f"PM2.5 Forecast at {site.replace('PM2.5_', '')}")
    plt.xlabel("Time Step (hour)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{site}.png")
    plt.close()

