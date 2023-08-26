import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Import necessary libraries

# Step 2: Data Gathering and Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def calculate_returns(data):
    returns = (data["Close"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data["Returns"] = returns
    return data

# Step 3: Calculating Returns

# Step 4: Regression Analysis and Estimating Alpha and Beta
def estimate_alpha_beta(data):
    Y = data["Returns"].dropna()
    X = data["Adj Close"].dropna()
    X = sm.add_constant(X)  # Add a constant for the intercept term in the regression

    model = sm.OLS(Y, X).fit()

    alpha = model.params[0]
    beta = model.params[1]
    return alpha, beta

# Step 5: Exponential Smoothing and Forecasting
def exponential_smoothing(data, alpha, beta):
    smoothed_values = [data["Adj Close"].iloc[0]]
    for i in range(1, len(data)):
        smoothed = alpha * data["Adj Close"].iloc[i] + (1 - alpha) * (smoothed_values[i - 1] + beta * (data["Adj Close"].iloc[i - 1] - smoothed_values[i - 1]))
        smoothed_values.append(smoothed)
    return smoothed_values

# Step 6: Visualize the Results
def plot_forecast(data, forecasted_stock_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(data["Date"], data["Adj Close"], label="Actual Stock Prices", color='blue')
    plt.plot(data["Date"].iloc[1:], forecasted_stock_prices, label="Forecasted Stock Prices", color='red')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Actual vs. Forecasted Stock Prices")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load the data from CSV
    file_path = "path/to/your/csv/file.csv"  # Replace this with the actual path to your CSV file
    data = load_data(file_path)

    # Calculate returns
    data = calculate_returns(data)

    # Estimate alpha and beta through regression analysis
    alpha, beta = estimate_alpha_beta(data)

    # Apply exponential smoothing to forecast the stock prices
    forecasted_stock_prices = exponential_smoothing(data, alpha, beta)

    # Plot the actual and forecasted stock prices
    plot_forecast(data, forecasted_stock_prices)
