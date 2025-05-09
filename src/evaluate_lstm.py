import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def evaluate_lstm():
    df = pd.read_csv("data/preprocessed_data.csv", index_col='Date')
    close_prices = df['Close'].values.reshape(-1, 1)

    # Load scaler config
    scaler = joblib.load("models/lstm_scaler.pkl")

    scaled_data = scaler.transform(close_prices)

    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y[split:]

    model = load_model("models/lstm_model.h5")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n📊 LSTM Model Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    evaluate_lstm()
