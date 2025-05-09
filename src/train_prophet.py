from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def train_prophet_model():
    df = pd.read_csv("data/preprocessed_data.csv", parse_dates=["Date"])
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet(daily_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_csv("output/prophet_forecast.csv", index=False)

    # Plot
    fig = model.plot(forecast)
    fig.savefig("output/prophet_plot.png")
    print("📈 Prophet forecast saved (CSV + Plot)")

if __name__ == "__main__":
    train_prophet_model()
