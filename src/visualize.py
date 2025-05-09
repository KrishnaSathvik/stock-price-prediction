import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions():
    df = pd.read_csv("output/predictions.csv")
    df.tail(100).plot(figsize=(12,6), title="Actual vs Predicted Stock Price")
    plt.xlabel("Date Index")
    plt.ylabel("Price")
    plt.grid()
    plt.savefig("output/prediction_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_predictions()
