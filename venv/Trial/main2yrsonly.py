import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 2: Data Gathering and Preprocessing
data = pd.read_csv("sample_stock_data.csv")
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime format
data = data.set_index('Date')  # Set 'Date' column as the index

# Step 3: Calculating Returns
returns = (data["Adj Close"] - data["Adj Close"].shift(1)) / data["Adj Close"].shift(1)
data["Returns"] = returns

# Step 4: Regression Analysis and Estimating Alpha and Beta
Y = data["Returns"].dropna()
X = data["Adj Close"].shift(1).dropna()  # Use the lagged 'Adj Close' as the independent variable
X = sm.add_constant(X)  # Add a constant for the intercept term in the regression

model = sm.OLS(Y, X).fit()

alpha = model.params[0]
beta = model.params[1]

# Step 5: Exponential Smoothing and Forecasting
def exponential_smoothing(data, alpha, beta, forecast_steps):
    n = len(data)
    smoothed_values = [data[0]]

    for i in range(1, n):
        if i == 1:
            smoothed = data[0] + alpha * (data[1] - data[0])
        else:
            smoothed = alpha * data[i] + (1 - alpha) * (smoothed_values[i - 1] + beta * (data[i - 1] - smoothed_values[i - 1]))
        smoothed_values.append(smoothed)

    # Extend the forecast for the next forecast_steps time steps
    n = len(data)
    forecasted_values = [data[0]]

    for i in range(1, n + forecast_steps):
        if i == 1:
            forecasted = data[0] + alpha * (data[1] - data[0])
        else:
            forecasted = alpha * data[i - forecast_steps] + (1 - alpha) * (forecasted_values[i - 1] + beta * (data[i - forecast_steps] - forecasted_values[i - 1]))
        forecasted_values.append(forecasted)

    return forecasted_values

# Convert forecast_steps from months to days (30 days per month)
forecast_months = 24
forecast_steps = forecast_months * 30

# Apply exponential smoothing to forecast the stock prices for the next 2 years (forecast_months)
stock_prices = data["Adj Close"].tolist()
smoothed_prices, forecasted_stock_prices = exponential_smoothing(stock_prices, alpha, beta, forecast_steps)

# Step 6: Visualize the Results for the next 2 years (in months)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Adj Close"], label="Actual Stock Prices", color='blue')
plt.plot(data.index, smoothed_prices[:len(data)], label="Smoothed Stock Prices", color='green', linestyle='dashed')
plt.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'),
         forecasted_stock_prices, label=f"Forecasted Stock Prices ({forecast_months} Months)", color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Actual, Smoothed, and Forecasted Stock Prices for the Next {forecast_months} Months")
plt.legend()
plt.show()
