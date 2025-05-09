import yfinance as yf

def fetch_stock_data(ticker="AAPL", start="2015-01-01", end="2024-12-31"):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)  # Converts index (Date) to a column
    data.to_csv(f"data/{ticker}.csv", index=False)
    print(f"✅ {ticker} data saved to data/{ticker}.csv")

if __name__ == "__main__":
    fetch_stock_data()
