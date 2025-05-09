import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def predict_lstm():
    df = pd.read_csv("data/preprocessed_data.csv", index_col='Date')
    close_prices = df['Close'].values.reshape(-1, 1)

    # Load scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    scaled_data = scaler.transform(close_prices)

    # Prepare last sequence
    seq_length = 60
    last_seq = scaled_data[-seq_length:]
    X_test = np.reshape(last_seq, (1, seq_length, 1))

    # Load model
    model = load_model("models/lstm_model.h5")
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)

    print(f"📈 Next Day Predicted Price: ${pred[0][0]:.2f}")

if __name__ == "__main__":
    predict_lstm()
