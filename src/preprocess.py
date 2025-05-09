import pandas as pd

def preprocess_stock_data(filename="data/AAPL.csv"):
    df = pd.read_csv(filename)

    # Rename first column if needed
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)

    if "Date" not in df.columns:
        raise ValueError("❌ 'Date' column is missing and could not be recovered.")

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Ensure 'Close' is numeric (drop rows where it's not)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df[['Close']].dropna()

    df['Return'] = df['Close'].pct_change()
    df['Close_SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Close_SMA_50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()

    df.to_csv("data/preprocessed_data.csv")
    print("✅ Preprocessing complete. Output saved to data/preprocessed_data.csv")

if __name__ == "__main__":
    preprocess_stock_data()
