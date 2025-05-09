import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/preprocessed_data.csv", parse_dates=["Date"])
    return df

df = load_data()
st.subheader("📊 Raw Stock Data")
st.line_chart(df.set_index("Date")["Close"])

# Linear Regression Prediction
st.subheader("📈 Linear Regression - Next Day Prediction")
model_lr = pickle.load(open("models/linear_model.pkl", "rb"))
latest_price = df["Close"].values[-1]
predicted_lr = model_lr.predict([[latest_price]])
st.metric(label="Next Day Predicted Price (Linear)", value=f"${predicted_lr[0]:.2f}")

# LSTM Prediction
st.subheader("🔮 LSTM Model - Next Day Prediction")
scaler = joblib.load("models/lstm_scaler.pkl")
seq_length = 60
close_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))
last_seq = close_scaled[-seq_length:]
X_input = np.reshape(last_seq, (1, seq_length, 1))
model_lstm = load_model("models/lstm_model.h5")
pred_scaled = model_lstm.predict(X_input)
predicted_lstm = scaler.inverse_transform(pred_scaled)
st.metric(label="Next Day Predicted Price (LSTM)", value=f"${predicted_lstm[0][0]:.2f}")

# XGBoost Prediction
st.subheader("🚀 XGBoost - Next Day Prediction")
model_xgb = pickle.load(open("models/xgboost_model.pkl", "rb"))
predicted_xgb = model_xgb.predict([[latest_price]])
st.metric(label="Next Day Predicted Price (XGBoost)", value=f"${predicted_xgb[0]:.2f}")

# Prophet Forecast
st.subheader("📅 Prophet Forecast - Next 7 Days")
try:
    forecast_df = pd.read_csv("output/prophet_forecast.csv")
    st.dataframe(forecast_df.tail(7))
    st.image("output/prophet_plot.png", caption="Prophet Forecast Plot")
except:
    st.warning("⚠️ Prophet forecast files not found. Please run `train_prophet.py` first.")

# Moving Averages
st.subheader("📉 20-Day and 50-Day Moving Averages")
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Date"], df["Close"], label="Close Price")
ax.plot(df["Date"], df["SMA_20"], label="20-Day SMA")
ax.plot(df["Date"], df["SMA_50"], label="50-Day SMA")
ax.set_title("Stock Price with Moving Averages")
ax.legend()
st.pyplot(fig)
