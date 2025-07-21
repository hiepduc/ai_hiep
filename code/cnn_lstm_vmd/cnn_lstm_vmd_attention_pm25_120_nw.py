import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vmdpy import VMD
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from tensorflow.keras.models import Model

class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_attention:
            return [K.sum(output, axis=1), a]
        else:
            return K.sum(output, axis=1)

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
joblib.dump(scaler, "pm25_vmd_attention_nw_scaler_120.save")
joblib.dump(df_vmd.columns.tolist(), "pm25_vmd_attention_nw_columns_120.save")

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

input_layer = Input(shape=(n_timesteps, n_features))
conv = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
lstm_out = LSTM(64, return_sequences=True)(conv)
#attn_out = Attention()(lstm_out)
#output = Dense(n_features)(attn_out)
#attn_out, attn_weights = Attention(return_attention=True)(lstm_out)
attn_out, attn_weights = Attention(return_attention=True, name="attention_layer")(lstm_out)
output = Dense(n_features)(attn_out)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()


# === 6. Train ===
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test), verbose=1)

# After model.fit()
attention_model = Model(inputs=model.input, outputs=model.get_layer("attention_layer").output[1])
attention_scores = attention_model.predict(X_test)
np.save("attention_scores_120.npy", attention_scores)

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
#site_names = df.columns  # Original site names before VMD
site_names = [col for col in df.columns if not "_IMF" in col]
imfs_per_site = K_IMF

reconstructed_actual = {}
reconstructed_pred = {}

for site in site_names:
    imf_cols = [col for col in df_vmd.columns if site in col]
    print(f"{site}: {imf_cols}")

for i, site in enumerate(site_names):
    #site_imf_indices = [j for j, name in enumerate(df_vmd.columns) if name.startswith(site)]
    #site_imf_indices = [df_vmd.columns.get_loc(f"{site}_IMF{k+1}") for k in range(K_IMF)]
    site_imf_cols = sorted([col for col in df_vmd.columns if col.startswith(f"{site}_IMF")])
    site_imf_indices = [df_vmd.columns.get_loc(col) for col in site_imf_cols]

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
plt.savefig("cnn_lstm_vmd_attention_all_sites_prediction_120.png")
plt.show()

# === 11. Save model ===
model.save("cnn_lstm_vmd_attention_model_pm25_nw_120input.h5")

# === 12. Compute MAE and RMSE per site ===
metrics_list = []

for site in site_names:
    try:
        site_imf_cols = [f"{site}_IMF{k+1}" for k in range(K_IMF)]
        site_imf_indices = [df_vmd.columns.get_loc(col) for col in site_imf_cols]
        actual = np.sum(y_test_inv[:, site_imf_indices], axis=1)
        predicted = np.sum(y_pred_inv[:, site_imf_indices], axis=1)
        mae = mean_absolute_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted, squared=False)
        metrics_list.append({"Site": site, "MAE": mae, "RMSE": rmse})
        print(f"{site}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")
    except KeyError as e:
        print(f"Skipping site {site} due to missing IMF columns: {e}")

avg_mae = np.mean([m["MAE"] for m in metrics_list])
avg_rmse = np.mean([m["RMSE"] for m in metrics_list])
metrics_list.append({"Site": "Average", "MAE": avg_mae, "RMSE": avg_rmse})

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("cnn_lstm_vmd_attention_site_metrics_120.csv", index=False)
print("\nPer-site MAE/RMSE saved to: cnn_lstm_vmd_attention_site_metrics_120.csv")

# === 13. Save full predictions to CSV ===
output_df = pd.DataFrame()
for site in site_names:
    output_df[f"{site}_actual"] = reconstructed_actual[site]
    output_df[f"{site}_predicted"] = reconstructed_pred[site]

output_df.to_csv("cnn_lstm_vmd_attention_predictions_120.csv", index=False)
print("Predictions saved to: cnn_lstm_vmd_attention_predictions_120.csv")

import seaborn as sns

# Example: visualize attention for first sample and first IMF
sample_idx = 0
feature_idx = 0

plt.figure(figsize=(10, 4))
sns.heatmap(attention_scores[sample_idx, :, feature_idx].reshape(1, -1), cmap="viridis", cbar=True)
plt.title(f"Attention Weights - Sample {sample_idx}, Feature {feature_idx}")
plt.xlabel("Time Step")
plt.yticks([])
plt.tight_layout()
plt.savefig("attention_sample0_feature0.png")
plt.show()

# Shape: (n_samples, time_steps, n_features)
mean_attention_over_features = np.mean(attention_scores, axis=2)  # shape: (samples, time_steps)
mean_attention = np.mean(mean_attention_over_features, axis=0)    # shape: (time_steps,)

plt.figure(figsize=(12, 4))
plt.plot(mean_attention)
plt.title("Average Attention Across All Samples and Features")
plt.xlabel("Time Step")
plt.ylabel("Mean Attention")
plt.grid(True)
plt.savefig("average_attention_over_time.png")
plt.show()

# Contribution of each IMF
for site in site_names:
    plt.figure(figsize=(10, 4))
    for k in range(K_IMF):
        col_name = f"{site}_IMF{k+1}"
        idx = df_vmd.columns.get_loc(col_name)
        plt.plot(y_pred_inv[:, idx], label=f"Predicted {col_name}")
    plt.title(f"{site}: Predicted IMFs (not yet summed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{site}_imf_contributions.png")
    plt.close()

