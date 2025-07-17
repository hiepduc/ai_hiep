from tensorflow.keras.models import load_model

model = load_model("cnn_lstm_model_pm25_sw_72input.h5")
model.summary()

