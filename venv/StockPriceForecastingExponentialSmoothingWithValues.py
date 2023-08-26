import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Ask the user for input
file_name = input("Enter the CSV file name (in uppercase without .csv extension): ")

try:
    # Step 2: Data Gathering and Preprocessing
    data = pd.read_csv(f"{file_name}.csv")
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
    def exponential_smoothing(stock_prices, alpha, beta, periods):
        n = len(stock_prices)
        smoothed_values = [stock_prices[0]]

        for i in range(1, n):
            smoothed = alpha * stock_prices[i] + (1 - alpha) * (smoothed_values[i - 1] + beta * (stock_prices[i - 1] - smoothed_values[i - 1]))
            smoothed_values.append(smoothed)

        forecasted_values = []
        last_date = data.index[-1]
        for i in range(1, periods + 1):
            next_date = last_date + pd.DateOffset(months=i)
            forecasted = alpha * stock_prices[-1] + (1 - alpha) * (smoothed_values[-1] + beta * (stock_prices[-1] - smoothed_values[-1]))
            forecasted_values.append((next_date, forecasted))
            smoothed_values.append(forecasted)

        return smoothed_values, forecasted_values

    # Apply exponential smoothing to forecast the stock prices for the next 2 years (24 months)
    stock_prices = data["Adj Close"].tolist()
    smoothed_prices, forecasted_stock_prices = exponential_smoothing(stock_prices, alpha, beta, periods=24)

    # Convert forecasted_stock_prices to DataFrame for tabular display
    forecast_df = pd.DataFrame(forecasted_stock_prices, columns=["Date", "Forecasted Price"])

    # Add alpha and beta values to the forecast_df
    forecast_df["Alpha"] = alpha
    forecast_df["Beta"] = beta

    # Create a folder for forecasted values if it doesn't exist
    folder_name = "forecasted_values"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the forecasted values to a CSV file in the folder
    forecast_output_file = os.path.join(folder_name, f"{file_name}_forecasted_values.csv")
    forecast_df.to_csv(forecast_output_file, index=False)

    # Step 6: Visualize the Results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Adj Close"], label="Actual Stock Prices", color='blue')
    plt.plot(data.index, smoothed_prices[:len(data)], label="Smoothed Stock Prices", color='green', linestyle='dashed')
    forecast_dates = pd.date_range(start=data.index[-1], periods=25, freq='M')[1:]
    forecasted_prices = [price for _, price in forecasted_stock_prices]
    plt.plot(forecast_dates, forecasted_prices, label="Forecasted Stock Prices", color='red')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Actual vs. Forecasted Stock Prices")
    plt.legend()

    # Create a folder for visualized results if it doesn't exist
    visualized_folder = "visualized_result"
    if not os.path.exists(visualized_folder):
        os.makedirs(visualized_folder)

    # Save the visualized result plot in the folder
    visualized_output_file = os.path.join(visualized_folder, f"{file_name}_visualized_result.png")
    plt.savefig(visualized_output_file)

    plt.show()

    # Display the forecasted values and alpha/beta in a table
    print("\nForecasted Stock Prices:")
    print(forecast_df)

except FileNotFoundError:
    print(f"Error: {file_name}.csv not found.")
