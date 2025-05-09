# 📈 Stock Price Prediction Pipeline

A complete end-to-end machine learning project for forecasting stock prices using traditional ML (Linear Regression, XGBoost) and deep learning (LSTM) models, including time-series forecasting using Facebook Prophet. An interactive Streamlit dashboard makes model outputs and analysis accessible and user-friendly.

---

## 🚀 Features

- 🔁 Historical stock data fetching via `yfinance`
- 🧹 Preprocessing and scaling of stock data
- 🤖 Models:
  - Linear Regression
  - LSTM (Deep Learning)
  - XGBoost (Boosted Trees)
  - Prophet (7-day time series forecast)
- 📊 Moving averages and time-series visualizations
- 📈 Streamlit dashboard for real-time model outputs
- ✅ Metrics: MAE, RMSE, R² for performance comparison

---

## 🛠 Tech Stack

- Python 3.10+
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `tensorflow`, `prophet`
  - `streamlit`, `yfinance`, `joblib`

---

## 📁 Folder Structure

```
stock-price-prediction/
├── data/                       # Raw and preprocessed CSV files
├── models/                     # Saved ML & DL models
├── notebooks/                  # Jupyter Notebooks (optional)
├── output/                     # Plots, predictions, Prophet outputs
├── src/
│   ├── dashboard.py            # Streamlit UI
│   ├── fetch_data.py           # Download stock data
│   ├── preprocess.py           # Clean and scale data
│   ├── train_model.py          # Train Linear, LSTM, XGBoost
│   ├── train_prophet.py        # Train Prophet model
│   ├── predict_lstm.py         # Run LSTM predictions
│   └── visualize.py            # Generate charts (optional)
├── README.md
└── requirements.txt
```

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Fetch and Preprocess Stock Data

```bash
python src/fetch_data.py
python src/preprocess.py
```

### 4. Train All Models

```bash
python src/train_model.py
python src/train_prophet.py
```

### 5. Run Streamlit Dashboard

```bash
streamlit run src/dashboard.py
```

---

## 📊 Dashboard Overview

The Streamlit UI shows:
- 📉 Raw stock data
- 📈 20-day and 50-day moving averages
- 🔮 Predictions from:
  - Linear Regression
  - LSTM
  - XGBoost
  - Prophet (7-day forecast)

---

## 📁 Outputs

| File | Purpose |
|------|---------|
| `models/*.pkl` | Saved ML models |
| `models/lstm_model.h5` | LSTM model |
| `output/prophet_forecast.csv` | 7-day forecast |
| `output/prophet_plot.png` | Prophet plot |
| `output/predictions.csv` | Final predictions (optional) |

---

## 📈 Example Output

### Prophet Forecast Table

| Date       | yhat     | yhat_lower | yhat_upper |
|------------|----------|------------|------------|
| 2025-05-01 | 185.12   | 183.75     | 187.31     |

### LSTM & Linear vs Actual Plot  
✅ Displayed in the Streamlit app with moving averages.

---

## 🔮 Future Improvements

- Add multi-stock selector in dashboard
- Add volume, RSI, MACD indicators
- Dockerize app for deployment
- Streamlit Cloud or Render hosting
- Backtesting framework for Prophet & LSTM

---

## 🤝 Contributing

Pull requests are welcome. Fork it, improve it, and send a PR 🚀

---

> Built with ❤️ by Krishna Sathvik
