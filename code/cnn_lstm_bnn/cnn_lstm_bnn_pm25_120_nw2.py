import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, RepeatVector
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl

print("TF version:", tf.__version__)
print("TFP version:", tfp.__version__)


# === PARAMETERS ===
INPUT_FILE = "/home/duch/ai_hiep/data/Imputed_data_NW_PM2.5.csv"
N_INPUT = 120     # 3 days
N_OUTPUT = 6      # 6-hour forecast
N_SUBSEQ = 10     # 120 = 10 subsequences of 12 hours each
EPOCHS = 30
BATCH_SIZE = 64

# === STEP 1: LOAD DATA ===
df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'], index_col='datetime')
print("Loaded data shape:", df.shape)

# === STEP 2: SCALING ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
joblib.dump(scaler, "pm25_cnn_lstm_bnn_nw_scaler_120.save")
joblib.dump(df.columns.tolist(), "pm25_cnn_lstm_bnn_nw_columns_120.save")

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

# === STEP 6: MODEL WITH BNN ===
tfd = tfp.distributions

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)
        )
    ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),
                                      reinterpreted_batch_ndims=1))
    ])

#def negloglik(y_true, y_pred):
#    return -y_pred.log_prob(y_true)
def negloglik(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)


# Define model
#model = Sequential([
#    TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'),
#                    input_shape=(N_SUBSEQ, n_steps_per_subseq, n_features)),
#    TimeDistributed(MaxPooling1D(pool_size=2)),
#    TimeDistributed(Flatten()),
#    LSTM(100, activation='relu'),
#    RepeatVector(N_OUTPUT),
#    LSTM(100, activation='relu', return_sequences=True),
#    TimeDistributed(tfpl.DenseVariational(
#        units=n_features * 2,  # mean + stddev
#        make_prior_fn=prior_trainable,
#        make_posterior_fn=posterior_mean_field,
#        kl_weight=1.0 / X_train.shape[0],
#        activation=None
#    )),
#    TimeDistributed(tfpl.IndependentNormal(n_features))
#])

from tensorflow.keras import Input, Model

# === Define inputs ===
inputs = Input(shape=(N_SUBSEQ, n_steps_per_subseq, n_features))

# === CNN + LSTM part ===
x = TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'))(inputs)
x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(100, activation='relu')(x)
x = RepeatVector(N_OUTPUT)(x)
x = LSTM(100, activation='relu', return_sequences=True)(x)

# === DenseVariational layer (mean + stddev) ===
x = TimeDistributed(tfpl.DenseVariational(
    units=n_features * 2,
    make_prior_fn=prior_trainable,
    make_posterior_fn=posterior_mean_field,
    kl_weight=1.0 / X_train.shape[0],
    activation=None
))(x)

# === Wrap as distribution ===
outputs = TimeDistributed(tfpl.IndependentNormal(n_features))(x)

# === Compile the model ===
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss=negloglik)

# Compile model
#model.compile(optimizer='adam', loss=negloglik)
#model.compile(optimizer='adam', loss='mse')
model.summary()

# === STEP 7: TRAIN MODEL ===
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

model.save_weights("cnn_lstm_bnn_pm25_nw_120input_weights.h5")
#model.save("cnn_lstm_bnn_pm25_nw_120input")

# === STEP 8: PREDICT & INVERSE SCALE ===
y_dist = model(X_val)
#y_dist = model.predict(X_test)  # probably returns a tensor already
print(type(y_dist))  # check if it's a Tensor or Distribution

y_pred_mean = y_dist.mean().numpy()
#y_pred_mean = y_dist.numpy()
y_val_flat = y_val.reshape(-1, n_features)
y_pred_flat = y_pred_mean.reshape(-1, n_features)
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
    plt.plot(y_pred_inv[:, idx], label='Predicted Mean')
    plt.title(f'PM2.5 Forecast at {site} (with BNN)')
    plt.xlabel('Time Step (hour)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.tight_layout()
    plt.show()

y_std = y_dist.stddev().numpy()
for site in site_names:
    idx = df.columns.get_loc(site)
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_inv[:, idx], label='Actual')
    plt.plot(y_pred_inv[:, idx], label='Predicted Mean')
    plt.fill_between(
        range(len(y_pred_inv[:, idx])),
        y_pred_inv[:, idx] - y_std[:, idx],
        y_pred_inv[:, idx] + y_std[:, idx],
        color='gray', alpha=0.3, label='Uncertainty ±1σ'
    )
    plt.title(f'PM2.5 Forecast at {site} (with BNN)')
    plt.xlabel('Time Step (hour)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.tight_layout()
    plt.show()

