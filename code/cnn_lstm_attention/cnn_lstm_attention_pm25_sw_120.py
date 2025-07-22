import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, RepeatVector
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM, Dense, Attention, Permute, Multiply, Lambda, Concatenate
import tensorflow.keras.backend as K
import tensorflow as tf

# === PARAMETERS ===
INPUT_FILE = "/home/duch/ai_hiep/data/Imputed_data_SW_PM2.5.csv"
N_INPUT = 120     # 3 days
N_OUTPUT = 6     # 6-hour forecast
N_SUBSEQ = 10     # 120 = 10 subsequences of 12 hours each
EPOCHS = 30
BATCH_SIZE = 64

# === STEP 1: LOAD DATA ===
df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'], index_col='datetime')
print("Loaded data shape:", df.shape)

# === STEP 2: SCALING ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
joblib.dump(scaler, "pm25_cnn_lstm_attention_sw_scaler_120.save")
joblib.dump(df.columns.tolist(), "pm25_cnn_lstm_attention_sw_columns_120.save")

n_features = scaled_data.shape[1]

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

# === STEP 6: MODEL with Attention ===

# Input layer
input_layer = Input(shape=(N_SUBSEQ, n_steps_per_subseq, n_features))

# CNN block (TimeDistributed)
cnn = TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'))(input_layer)
cnn = TimeDistributed(MaxPooling1D(pool_size=2))(cnn)
cnn = TimeDistributed(Flatten())(cnn)

# LSTM encoder
encoder_output = LSTM(100, return_sequences=True)(cnn)

# Attention mechanism
attention_scores = Dense(1, activation='tanh')(encoder_output)               # Shape: (batch, time, 1)
attention_weights = tf.nn.softmax(attention_scores, axis=1)                  # Shape: (batch, time, 1)
context_vector = tf.reduce_sum(attention_weights * encoder_output, axis=1)   # Shape: (batch, features)

# Decoder: Repeat context and pass through LSTM
context_repeated = RepeatVector(N_OUTPUT)(context_vector)
decoder_lstm = LSTM(100, return_sequences=True)(context_repeated)
output = TimeDistributed(Dense(n_features))(decoder_lstm)

# Build and compile
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()
y_pred = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val.reshape(-1, n_features), y_pred.reshape(-1, n_features)))

# === STEP 7: TRAIN MODEL ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)
model.save("cnn_lstm_attention_model_pm25_sw_120input.h5")

# === STEP 8: PREDICT & INVERSE SCALE ===
y_pred = model.predict(X_val)
y_val_flat = y_val.reshape(-1, n_features)
y_pred_flat = y_pred.reshape(-1, n_features)
y_val_inv = scaler.inverse_transform(y_val_flat)
y_pred_inv = scaler.inverse_transform(y_pred_flat)

# === STEP 9: EVALUATION ===
site_names = df.columns.tolist()
print("\nMAE per site (µg/m³):")
for i, site in enumerate(site_names):
    mae_site = mean_absolute_error(y_val_inv[:, i], y_pred_inv[:, i])
    print(f"{site:<30}: {mae_site:.2f}")

print(f"\nOverall MAE (µg/m³): {mean_absolute_error(y_val_inv, y_pred_inv):.2f}")

# === STEP 10: PLOT ===
for site in site_names:
    idx = df.columns.get_loc(site)
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_inv[:, idx], label='Actual')
    plt.plot(y_pred_inv[:, idx], label='Predicted')
    plt.title(f'PM2.5 Forecast at {site}')
    plt.xlabel('Time Step (hour)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.tight_layout()
    plt.show()


