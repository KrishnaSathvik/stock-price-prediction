import pandas as pd
import pickle

def predict():
    df = pd.read_csv("data/preprocessed_data.csv", index_col='Date')
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    model = pickle.load(open("models/linear_model.pkl", "rb"))
    df['Predicted'] = model.predict(df[['Close']])

    df[['Close', 'Predicted']].tail(10).to_csv("output/predictions.csv")
    print("📈 Predictions saved to output/predictions.csv")

if __name__ == "__main__":
    predict()
