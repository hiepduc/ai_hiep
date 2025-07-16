import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, RepeatVector
import matplotlib.pyplot as plt
import joblib

# === PARAMETERS ===
INPUT_FILE = "/home/duch/ai_hiep/data/Imputed_data_UH_PM2.5.csv"
N_INPUT = 120      # Input: past 72 hours (3 days)
N_OUTPUT = 6      # Output: next 6 hours
N_SUBSEQ = 10      # Number of subsequences
EPOCHS = 30
BATCH_SIZE = 64

# === STEP 1: LOAD DATA ===
df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'], index_col='datetime')
print("Loaded data shape:", df.shape)

# === STEP 2: SCALING ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
n_features = scaled_data.shape[1]
joblib.dump(scaler, "pm25_uh_scaler_120.save")
joblib.dump(df.columns.tolist(), "pm25_uh_columns_120.save")

# === STEP 3: CREATE SEQUENCES ===
def create_sequences(data, n_input, n_output):
    X, y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        X.append(data[i:i+n_input])
        y.append(data[i+n_input:i+n_input+n_output])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, N_INPUT, N_OUTPUT)
print("X shape:", X.shape)
print("y shape:", y.shape)

# === STEP 4: RESHAPE FOR CNN-LSTM ===
n_steps_per_subseq = N_INPUT // N_SUBSEQ
X = X.reshape((X.shape[0], N_SUBSEQ, n_steps_per_subseq, n_features))

# === STEP 5: TRAIN/VAL SPLIT ===
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

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
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)
model.save("cnn_lstm_model_pm25_uh_120input.h5")

# === STEP 8: PREDICT & INVERSE SCALE ===
y_pred = model.predict(X_val)

# Flatten for comparison
y_val_flat = y_val.reshape(-1, n_features)
y_pred_flat = y_pred.reshape(-1, n_features)

y_val_inv = scaler.inverse_transform(y_val_flat)
y_pred_inv = scaler.inverse_transform(y_pred_flat)

# === STEP 9: Evaluation
site_names = df.columns.tolist()
print("\nMAE per site (µg/m³):")
for i, site in enumerate(site_names):
    mae_site = mean_absolute_error(y_val_inv[:, i], y_pred_inv[:, i])
    print(f"{site:<30}: {mae_site:.2f}")

print(f"\nOverall MAE (µg/m³): {mean_absolute_error(y_val_inv, y_pred_inv):.2f}")

# === STEP 10: PLOTS
for site in site_names:
    site_index = df.columns.get_loc(site)
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_inv[:, site_index], label='Actual')
    plt.plot(y_pred_inv[:, site_index], label='Predicted')
    plt.title(f'PM2.5 Forecast at {site}')
    plt.xlabel('Time Step (hour)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.tight_layout()
    plt.show()

