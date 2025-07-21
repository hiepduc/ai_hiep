import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vmdpy import VMD
import matplotlib.pyplot as plt

# === Parameters ===
INPUT_FILE = "/home/duch/ai_hiep/data/Imputed_data_NW_PM2.5.csv"
LOOK_BACK = 120
K_IMF = 3  # Number of VMD components
TEST_SIZE = 500
BATCH_SIZE = 64
EPOCHS = 30

# === 1. Read and preprocess data ===
df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"], index_col="datetime")

def apply_vmd_to_all_sites(df, K=3):
    vmd_components = []
    for col in df.columns:
        signal = df[col].values
        u, _, _ = VMD(signal, alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-7)
        for k in range(K):
            df[f"{col}_IMF{k+1}"] = u[k]
            vmd_components.append(f"{col}_IMF{k+1}")
    return df[vmd_components]

df_vmd = apply_vmd_to_all_sites(df, K=K_IMF)

# === 2. Normalize and save scaler ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_vmd)
joblib.dump(scaler, "pm25_vmd_nw_scaler_120.save")
joblib.dump(df_vmd.columns.tolist(), "pm25_vmd_nw_columns_120.save")

# === 3. Create input/output sequences ===
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOK_BACK)

# === 4. Train/test split ===
X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
y_train, y_test = y[:-TEST_SIZE], y[-TEST_SIZE:]

# === 5. Build CNN-LSTM model ===
n_timesteps = X.shape[1]
n_features = X.shape[2]

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    LSTM(100),
    Dense(n_features)  # Predicting all features at once
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# === 6. Train ===
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test), verbose=1)

# === 7. Predict ===
y_pred = model.predict(X_test)

# === 8. Inverse transform predictions ===
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# === 9. Evaluate ===
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")

# === 10. Reconstruct PM2.5 by summing IMFs for each site ===
site_names = df.columns  # Original site names before VMD
imfs_per_site = K_IMF

reconstructed_actual = {}
reconstructed_pred = {}

for i, site in enumerate(site_names):
    site_imf_indices = [j for j, name in enumerate(df_vmd.columns) if name.startswith(site)]
    actual_sum = np.sum(y_test_inv[:, site_imf_indices], axis=1)
    pred_sum = np.sum(y_pred_inv[:, site_imf_indices], axis=1)
    reconstructed_actual[site] = actual_sum
    reconstructed_pred[site] = pred_sum

# === 11. Plot all sites ===
import matplotlib.pyplot as plt
n_sites = len(site_names)
fig, axs = plt.subplots(n_sites, 1, figsize=(12, 4 * n_sites), sharex=True)

if n_sites == 1:
    axs = [axs]

for i, site in enumerate(site_names):
    axs[i].plot(reconstructed_actual[site], label="Actual", color="black")
    axs[i].plot(reconstructed_pred[site], label="Predicted", color="red")
    axs[i].set_title(f"PM2.5 Prediction at {site}")
    axs[i].set_ylabel("PM2.5 (µg/m³)")
    axs[i].legend()
    axs[i].grid(True)

plt.xlabel("Time Step")
plt.tight_layout()
plt.savefig("cnn_lstm_vmd_all_sites_prediction_120.png")
plt.show()

# === 11. Save model ===
model.save("cnn_lstm_vmd_model_pm25_nw_120input.h5")

# === 12. Compute MAE and RMSE per site ===
metrics_list = []

for site in site_names:
    actual = reconstructed_actual[site]
    predicted = reconstructed_pred[site]
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    metrics_list.append({"Site": site, "MAE": mae, "RMSE": rmse})
    print(f"{site}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("cnn_lstm_vmd_nw_site_metrics_120.csv", index=False)
print("\nPer-site MAE/RMSE saved to: cnn_lstm_vmd_nw_site_metrics_120.csv")

# === 13. Save full predictions to CSV ===
output_df = pd.DataFrame()
for site in site_names:
    output_df[f"{site}_actual"] = reconstructed_actual[site]
    output_df[f"{site}_predicted"] = reconstructed_pred[site]

output_df.to_csv("cnn_lstm_vmd_nw_predictions_120.csv", index=False)
print("Predictions saved to: cnn_lstm_vmd_nw_predictions_120.csv")

