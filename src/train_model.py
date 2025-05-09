import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_lstm_model():
    df = pd.read_csv("data/preprocessed_data.csv", index_col='Date')
    close_prices = df['Close'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare sequences
    seq_length = 60
    X, y = [], []

    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model and scaler
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save("models/lstm_model.h5")
    joblib.dump(scaler, "models/lstm_scaler.pkl")
    print("✅ LSTM model trained and saved.")

    # Predict on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 LSTM Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

def train_xgboost_model():
    df = pd.read_csv("data/preprocessed_data.csv", index_col='Date')
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Close']]
    y = df['Target']

    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Save model
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\n📊 XGBoost Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_lstm_model()
    train_lstm_model()
    train_xgboost_model()
