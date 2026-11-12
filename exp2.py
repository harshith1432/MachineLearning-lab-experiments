# Experiment: EXP2
# Laboratory Program for Machine Learning in Cybersecurity

# !pip install statsmodels



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")


np.random.seed(42)

dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

trend = np.linspace(500, 600, 120)

seasonal = 40 * np.sin(np.arange(120) * 2 * np.pi / 7)

noise = np.random.normal(0, 15, 120)

traffic = trend + seasonal + noise



traffic[-10:] += 150



data = pd.DataFrame({
    "Date": dates,
    "Traffic": traffic
})

data.set_index("Date", inplace=True)



model = ARIMA(data["Traffic"], order=(2,1,2))
model_fit = model.fit()



forecast = model_fit.forecast(steps=7)

future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast.values
})

forecast_df.set_index("Date", inplace=True)



plt.figure(figsize=(12,6))

plt.plot(
    data.index,
    data["Traffic"],
    label="Historical Traffic",
    linewidth=2
)

plt.plot(
    forecast_df.index,
    forecast_df["Forecast"],
    color="red",
    marker="o",
    linewidth=2,
    label="ARIMA Forecast"
)

plt.title("Time-Series Forecasting for DDoS Attack Prediction (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Network Traffic")
plt.legend()
plt.grid(True)

plt.show()



threshold = 620

print("="*50)
print("Forecasted Traffic")
print("="*50)
print(forecast_df)

print("\nMaximum Forecast Value :", round(forecast_df["Forecast"].max(),2))
print("Threshold              :", threshold)

if forecast_df["Forecast"].max() > threshold:
    print("\n⚠ Potential DDoS Attack Predicted")
else:
    print("\n✅ Normal Network Traffic")



